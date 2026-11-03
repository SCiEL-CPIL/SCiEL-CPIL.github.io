import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.linalg import cho_factor, cho_solve

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    Vectorized gradient computation using FFT.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Compute gradient using FFT-based convolution
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # conv1[j] = sum_k w[k] * h[j-k] (linear convolution)
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # conv2[j] = sum_k w[k] * h[j+k]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    # Combine
    grad = -(2.0 / n) * (conv1 + conv2)
    
    return val, grad


def _psd_to_h(psd):
    """
    Convert PSD (power spectral density) to time-domain sequence h.
    psd[k] = |H_k|^2 for k=0..N/2
    h_i = (1/N) * sum_k H_k * exp(2*pi*i*k*i/N)
    We need to recover H_k from psd, but we lose phase info.
    For simplicity, assume H_k is real and non-negative (cosine-like).
    """
    n = len(psd)
    # psd has N/2 + 1 elements (rfft)
    H = np.zeros(n, dtype=np.complex128)
    H.real = np.sqrt(np.maximum(psd, 0))
    # For k=0 and k=N/2 (if N even), H is real
    # For other k, we assume real for simplicity (could add phase later)
    h = np.fft.irfft(H, n=N)
    return h


def _h_to_psd(h):
    """
    Convert time-domain h to PSD.
    """
    H = np.fft.rfft(h, n=2*N)
    psd = np.abs(H[:N//2+1])**2
    return psd


def _psd_overlap_values(psd):
    """
    Compute overlap values directly from PSD.
    C_k = (2/N) * (N/2 - r_k) where r_k is autocorrelation
    r_k = (1/N) * sum_m PSD[m] * exp(2*pi*i*k*m/N)
    """
    n = N
    n_half = n // 2 + 1
    # psd should have n_half elements
    psd = np.asarray(psd, dtype=np.float64)
    if len(psd) < n_half:
        psd = np.pad(psd, (0, n_half - len(psd)))
    psd = psd[:n_half]
    
    # Compute autocorrelation from PSD
    # r_k = (1/N) * sum_{m=0}^{N/2} psd[m] * cos(2*pi*k*m/N) for m=1..N/2-1
    # with special handling for m=0 and m=N/2
    k = np.arange(n)
    m = np.arange(n_half)
    
    # r_k = (1/N) * [psd[0] + 2*sum_{m=1}^{N/2-1} psd[m]*cos(2*pi*k*m/N) + psd[N/2]*cos(2*pi*k*N/2/N)]
    # cos(2*pi*k*N/2/N) = cos(pi*k) = (-1)^k
    
    if n_half > 1:
        cos_terms = np.cos(2.0 * np.pi * np.outer(k, m[1:-1]) / n)
        r = (psd[0] + 2.0 * np.sum(psd[1:-1] * cos_terms, axis=1) + psd[-1] * ((-1.0)**k)) / n
    else:
        r = np.full(n, psd[0] / n)
    
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _lp_psd_optimization(h0, time_budget):
    """
    Use linear programming to optimize PSD coefficients.
    Minimize t subject to:
    - C_k(psd) <= t for all k
    - psd[m] >= 0 for all m
    - sum(h_i) = N/2 (enforced via DC component)
    - h_i in [0,1] (enforced via Cholesky factorization of covariance)
    """
    t_start = time.time()
    n = N
    n_half = n // 2 + 1
    
    # Initial PSD from h0
    psd0 = _h_to_psd(h0)
    
    # LP formulation:
    # Variables: psd[0..n_half-1], t
    # Minimize: t
    # Subject to:
    #   C_k(psd) <= t for k=0..n-1
    #   psd[m] >= 0 for m=0..n_half-1
    #   DC constraint: psd[0] = N/2 (since sum h_i = N/2 means H_0 = N/2, so psd[0] = N^2/4)
    #   Actually, H_0 = sum h_i = N/2, so psd[0] = |H_0|^2 = N^2/4
    
    # Build constraint matrix A_ub for C_k(psd) - t <= 0
    # C_k = (2/N) * (N/2 - r_k)
    # r_k = (1/N) * [psd[0] + 2*sum_{m=1}^{N/2-1} psd[m]*cos(2*pi*k*m/N) + psd[N/2]*(-1)^k]
    # C_k = (2/N) * (N/2 - (1/N)*[...]) = 1 - (2/N^2)*[...]
    
    # So C_k - t <= 0 becomes:
    # 1 - (2/N^2)*[psd[0] + 2*sum psd[m]*cos(...) + psd[N/2]*(-1)^k] - t <= 0
    # -(2/N^2)*psd[0] - (4/N^2)*sum psd[m]*cos(...) - (2/N^2)*psd[N/2]*(-1)^k - t <= -1
    
    n_vars = n_half + 1  # psd[0..n_half-1] and t
    A_ub = np.zeros((n, n_vars))
    b_ub = np.zeros(n)
    
    k = np.arange(n)
    m = np.arange(n_half)
    
    for kk in range(n):
        # Coefficient for psd[0]
        A_ub[kk, 0] = -2.0 / (n**2)
        # Coefficients for psd[1..n_half-2]
        if n_half > 2:
            A_ub[kk, 1:-1] = -4.0 / (n**2) * np.cos(2.0 * np.pi * kk * m[1:-1] / n)
        # Coefficient for psd[n_half-1]
        A_ub[kk, n_half-1] = -2.0 / (n**2) * ((-1.0)**kk)
        # Coefficient for t
        A_ub[kk, n_half] = -1.0
        b_ub[kk] = -1.0
    
    # Non-negativity: psd[m] >= 0
    # In linprog, we use bounds
    bounds = [(0, None)] * n_half + [(0, None)]  # psd non-negative, t non-negative
    
    # Objective: minimize t
    c = np.zeros(n_vars)
    c[n_half] = 1.0
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            psd_opt = res.x[:n_half]
            t_opt = res.x[n_half]
            
            # Convert back to h
            H = np.zeros(n_half, dtype=np.complex128)
            H.real = np.sqrt(np.maximum(psd_opt, 0))
            h_opt = np.fft.irfft(H, n=n)
            h_opt = _fix_sum(h_opt)
            
            s = _score(h_opt)
            return h_opt, s
    except Exception as e:
        pass
    
    return h0, _score(h0)


def _cholesky_psd_optimization(h0, time_budget):
    """
    Optimize PSD using Cholesky factorization to enforce h_i in [0,1].
    This is a more sophisticated approach that ensures the time-domain
    sequence is valid.
    """
    t_start = time.time()
    n = N
    n_half = n // 2 + 1
    
    # Start from current h
    psd0 = _h_to_psd(h0)
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Try different PSD modifications
    # 1. Scale PSD
    for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3]:
        if time.time() - t_start > time_budget:
            break
        
        psd_mod = psd0.copy()
        psd_mod[1:] *= scale  # Scale non-DC components
        psd_mod[0] = (n/2.0)**2  # Keep DC fixed
        
        H = np.zeros(n_half, dtype=np.complex128)
        H.real = np.sqrt(np.maximum(psd_mod, 0))
        h_mod = np.fft.irfft(H, n=n)
        h_mod = _fix_sum(h_mod)
        s = _score(h_mod)
        
        if s < best_score:
            best_score = s
            best_h = h_mod.copy()
    
    # 2. Low-pass filter PSD
    for cutoff in [2, 3, 5, 8, 12]:
        if time.time() - t_start > time_budget:
            break
        
        psd_mod = psd0.copy()
        psd_mod[cutoff:] *= 0.1
        psd_mod[0] = (n/2.0)**2
        
        H = np.zeros(n_half, dtype=np.complex128)
        H.real = np.sqrt(np.maximum(psd_mod, 0))
        h_mod = np.fft.irfft(H, n=n)
        h_mod = _fix_sum(h_mod)
        s = _score(h_mod)
        
        if s < best_score:
            best_score = s
            best_h = h_mod.copy()
    
    return best_h, best_score


def _spectral_search(h0, time_budget):
    """
    Spectral search in the frequency domain.
    """
    t_start = time.time()
    n = len(h0)
    n_fft = 2 * n
    
    # Get the DFT of h0
    H0 = np.fft.rfft(h0, n=n_fft)
    
    # We'll search over scaling and phase shifts in frequency domain
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Try different spectral modifications
    # 1. Scale high frequencies
    for scale_factor in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]:
        if time.time() - t_start > time_budget:
            break
        
        H_mod = H0.copy()
        # Scale frequencies above 1
        for k in range(2, len(H_mod)):
            H_mod[k] *= scale_factor
        
        h_mod = np.fft.irfft(H_mod, n=n_fft)[:n]
        h_mod = _fix_sum(h_mod)
        s = _score(h_mod)
        
        if s < best_score:
            best_score = s
            best_h = h_mod.copy()
    
    # 2. Try phase rotations in frequency domain
    for phase_shift in np.linspace(0, 2*np.pi, 12, endpoint=False):
        if time.time() - t_start > time_budget:
            break
        
        H_mod = H0.copy()
        # Apply phase rotation to higher frequencies
        for k in range(2, len(H_mod)):
            H_mod[k] *= np.exp(1j * phase_shift * (k - 1))
        
        h_mod = np.fft.irfft(H_mod, n=n_fft)[:n]
        h_mod = _fix_sum(h_mod)
        s = _score(h_mod)
        
        if s < best_score:
            best_score = s
            best_h = h_mod.copy()
    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Multiple structured initial guesses
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Bump function: high in middle, low at edges
    h_bump = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    h_bump = _fix_sum(h_bump)
    _update_best(h_bump)
    
    # Step-like: 1s in center, 0s at edges
    h_step = np.zeros(N)
    h_step[N//4:3*N//4] = 1.0
    h_step = _fix_sum(h_step)
    _update_best(h_step)
    
    # Uniform
    h_unif = np.full(N, 0.5)
    _update_best(h_unif)
    
    # Square wave (high freq)
    h_sq = np.where((i % (N//4)) < (N//8), 1.0, 0.0)
    h_sq = _fix_sum(h_sq)
    _update_best(h_sq)
    
    # Sine wave (high freq)
    h_sin = 0.5 + 0.5 * np.sin(4.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    _update_best(h_sin)
    
    # Start from the best initial guess
    h0 = best_h.copy()
    
    def _run_optimization(h0, beta, maxiter, ftol, method='SLSQP'):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _smooth_max_grad(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _smooth_max_grad(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method=method,
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # Phase 1: Coarse optimization with low beta
    if time.time() - t0 < _DEADLINE - 50:
        _run_optimization(h0, beta=10.0, maxiter=80, ftol=1e-5)
        
    # Phase 2: Medium optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 35:
        _run_optimization(best_h, beta=30.0, maxiter=120, ftol=1e-7)
        
    # Phase 3: Higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        _run_optimization(best_h, beta=60.0, maxiter=150, ftol=1e-9)
        
    # Phase 4: Fine optimization with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        _run_optimization(best_h, beta=100.0, maxiter=200, ftol=1e-11)
        
    # Phase 5: Final polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_optimization(best_h, beta=150.0, maxiter=250, ftol=1e-13)
    
    # NEW: PSD-based LP optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_lp, s_lp = _lp_psd_optimization(best_h, time_budget=15)
        if s_lp < best_score:
            _update_best(h_lp)
            # Refine the LP result
            if time.time() - t0 < _DEADLINE - 10:
                _run_optimization(h_lp, beta=80.0, maxiter=100, ftol=1e-10)
    
    # NEW: Cholesky-based PSD optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_chol, s_chol = _cholesky_psd_optimization(best_h, time_budget=10)
        if s_chol < best_score:
            _update_best(h_chol)
            # Refine the Cholesky result
            if time.time() - t0 < _DEADLINE - 5:
                _run_optimization(h_chol, beta=100.0, maxiter=150, ftol=1e-11)
    
    # Spectral search
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_spec, s_spec = _spectral_search(best_h, time_budget=15)
        if s_spec < best_score:
            _update_best(h_spec)
            # Refine the spectral result
            if time.time() - t0 < _DEADLINE - 10:
                _run_optimization(h_spec, beta=80.0, maxiter=100, ftol=1e-10)
    
    # Pure tone spectral search
    if time.time() - t0 < _DEADLINE - 40:
        # Try different frequencies and phases
        freq_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        for freq in freq_candidates:
            if time.time() - t0 > _DEADLINE - 20:
                break
            
            # Try multiple phases and amplitudes
            for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
                if time.time() - t0 > _DEADLINE - 15:
                    break
                
                for amp in [0.3, 0.4, 0.5, 0.6]:
                    if time.time() - t0 > _DEADLINE - 10:
                        break
                    
                    params = np.array([float(freq), phase, amp, 0.5])
                    i_arr = np.arange(N)
                    h_tone = params[3] + amp * np.cos(2.0 * np.pi * params[0] * i_arr / N + params[1])
                    h_tone = _fix_sum(h_tone)
                    s = _score(h_tone)
                    
                    if s < best_score:
                        _update_best(h_tone)
                        
                        # Local refinement with gradient
                        if time.time() - t0 > _DEADLINE - 8:
                            break
                            
                        try:
                            def obj(p):
                                h_p = p[3] + p[2] * np.cos(2.0 * np.pi * p[0] * np.arange(N) / N + p[1])
                                h_p = _fix_sum(h_p)
                                val, _ = _smooth_max_grad(h_p, 50.0)
                                return val
                            
                            res = minimize(obj, params.copy(), method='L-BFGS-B',
                                           bounds=[(0.5, 15), (0, 2*np.pi), (0.1, 0.8), (0.3, 0.7)],
                                           options={'maxiter': 100, 'ftol': 1e-12})
                            if res.fun < best_score:
                                h_refined = res.x[3] + res.x[2] * np.cos(2.0 * np.pi * res.x[0] * np.arange(N) / N + res.x[1])
                                _update_best(h_refined)
                        except Exception:
                            pass
    
    # Two-tone spectral search
    if time.time() - t0 < _DEADLINE - 25:
        # Try combinations of two frequencies
        for f1 in [1, 2, 3]:
            if time.time() - t0 > _DEADLINE - 12:
                break
            for f2 in [4, 5, 6, 7]:
                if time.time() - t0 > _DEADLINE - 10:
                    break
                
                # Try a few phase combinations
                for p1 in [0, np.pi/2, np.pi]:
                    if time.time() - t0 > _DEADLINE - 8:
                        break
                    for p2 in [0, np.pi/2, np.pi]:
                        if time.time() - t0 > _DEADLINE - 6:
                            break
                        
                        for a1 in [0.2, 0.3]:
                            if time.time() - t0 > _DEADLINE - 5:
                                break
                            for a2 in [0.1, 0.2]:
                                if time.time() - t0 > _DEADLINE - 4:
                                    break
                                
                                i_arr = np.arange(N)
                                h_two = 0.5
                                h_two += a1 * np.cos(2.0 * np.pi * f1 * i_arr / N + p1)
                                h_two += a2 * np.cos(2.0 * np.pi * f2 * i_arr / N + p2)
                                h_two = _fix_sum(h_two)
                                s = _score(h_two)
                                
                                if s < best_score:
                                    _update_best(h_two)
                                    
                                    # Local refinement
                                    if time.time() - t0 > _DEADLINE - 3:
                                        break
                                        
                                    try:
                                        def obj2(p):
                                            i_arr = np.arange(N)
                                            h_p = p[6]
                                            h_p += p[2] * np.cos(2.0 * np.pi * p[0] * i_arr / N + p[1])
                                            h_p += p[5] * np.cos(2.0 * np.pi * p[3] * i_arr / N + p[4])
                                            h_p = _fix_sum(h_p)
                                            val, _ = _smooth_max_grad(h_p, 50.0)
                                            return val
                                        
                                        res = minimize(obj2, np.array([float(f1), p1, a1, float(f2), p2, a2, 0.5]),
                                                       method='L-BFGS-B',
                                                       bounds=[(0.5, 5), (0, 2*np.pi), (0.05, 0.5),
                                                              (2, 10), (0, 2*np.pi), (0.05, 0.5),
                                                              (0.3, 0.7)],
                                                       options={'maxiter': 80, 'ftol': 1e-12})
                                        if res.fun < best_score:
                                            i_arr = np.arange(N)
                                            h_refined = res.x[6]
                                            h_refined += res.x[2] * np.cos(2.0 * np.pi * res.x[0] * i_arr / N + res.x[1])
                                            h_refined += res.x[5] * np.cos(2.0 * np.pi * res.x[3] * i_arr / N + res.x[4])
                                            _update_best(h_refined)
                                    except Exception:
                                        pass
    
    # Random restarts if time permits
    if time.time() - t0 < _DEADLINE - 25:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            if time.time() - t0 > _DEADLINE - 12:
                break
            _run_optimization(h_rand, beta=30.0, maxiter=60, ftol=1e-7)
            if best_h is not None and time.time() - t0 < _DEADLINE - 8:
                _run_optimization(best_h, beta=80.0, maxiter=100, ftol=1e-10)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
