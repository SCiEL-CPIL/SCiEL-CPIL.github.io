import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
_DEADLINE = time.time() + 110

def _linear_overlap_values(h):
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
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_score(h, tau=5.0):
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    if m < 1e-9:
        return 0.0
    return m + tau * np.log(np.sum(np.exp((cs - m) / tau)))

def _fix_sum(h):
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

def _h_from_spectral(magnitudes, phases, support):
    """
    Build h from spectral representation.
    magnitudes: array of |X[k]| for k in support
    phases: array of arg(X[k]) for k in support
    support: list of frequency indices (including 0 for DC)
    
    We use the fact that for real h, X[N-k] = conj(X[k]), so we only
    need to specify X[k] for k in support where k <= N/2.
    """
    X = np.zeros(N, dtype=np.complex128)
    for i, k in enumerate(support):
        if k == 0:
            X[0] = magnitudes[0]
        elif k == N // 2:
            X[k] = magnitudes[i] * np.exp(1j * phases[i])
        else:
            val = magnitudes[i] * np.exp(1j * phases[i])
            X[k] = val
            X[N - k] = np.conj(val)
    
    y = np.fft.ifft(X)
    h_raw = np.abs(y) ** 2
    total = np.sum(h_raw)
    if total < 1e-15:
        return np.full(N, 0.5)
    # Scale so that sum(h) = N/2
    h = h_raw / total * (N / 2.0)
    return h

def _spectral_objective(params, support, target_sum=N/2.0):
    """
    Objective function for spectral optimization.
    params layout: [mag_0, mag_1, ..., mag_{K-1}, phase_1, ..., phase_{K-1}]
    (phase_0 is fixed to 0 since X[0] is real for real h)
    """
    K = len(support)
    magnitudes = params[:K]
    phases = params[K:]  # K-1 phases (phase for DC is 0)
    
    # Ensure non-negative magnitudes
    magnitudes = np.maximum(magnitudes, 0.0)
    
    h = _h_from_spectral(magnitudes, phases, support)
    
    # Penalty for exceeding [0,1] bounds
    penalty = 0.0
    over = h[h > 1.0]
    if len(over) > 0:
        penalty += np.sum((over - 1.0) ** 2) * 50.0
    under = h[h < 0.0]
    if len(under) > 0:
        penalty += np.sum(under ** 2) * 50.0
    
    s = _smooth_max_score(h, tau=2.0)
    return s + penalty

def _polish_time_domain(h0, max_iter=300, ftol=1e-12):
    h0 = _fix_sum(h0)
    
    def obj(x):
        return _smooth_max_score(x, tau=1.0)
        
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    try:
        res = minimize(obj, h0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': max_iter, 'ftol': ftol})
        if res.success and res.fun < _score(h0):
            return _fix_sum(res.x)
    except Exception:
        pass
    
    for tau in [0.5, 2.0, 3.0]:
        def obj_t(x):
            return _smooth_max_score(x, tau=tau)
        try:
            res = minimize(obj_t, h0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 100, 'ftol': 1e-10})
            if res.success and res.fun < _score(h0):
                return _fix_sum(res.x)
        except Exception:
            pass
    
    return h0

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        if np.any(h > 1.0 + 1e-9) or np.any(h < -1e-9):
            return
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Fallbacks
    _update_best(np.full(N, 0.5))
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Spectral optimization with various supports
    # Strategy: try different cardinalities and frequency selections
    
    def _optimize_support(support, n_starts=4, use_de=False):
        """Optimize spectral coefficients for a given support."""
        K = len(support)
        # params: K magnitudes + (K-1) phases
        n_params = 2 * K - 1
        
        best_local = None
        best_local_score = float('inf')
        
        def _eval_params(params):
            h = _h_from_spectral(params[:K], params[K:], support)
            if np.any(h > 1.0 + 1e-6) or np.any(h < -1e-6):
                return float('inf')
            return _score(h)
        
        # Generate initial guesses
        starts = []
        
        # Start 0: DC dominant, small random AC
        p0 = np.zeros(n_params)
        p0[0] = np.sqrt(N / 2.0)  # DC magnitude for uniform h
        for j in range(K - 1):
            p0[K + j] = np.random.rand() * 2 * np.pi
        # Small AC magnitudes
        for j in range(1, K):
            p0[j] = 0.01 * np.random.rand()
        starts.append(p0)
        
        # Start 1: Two-tone (DC + one strong AC)
        p1 = np.zeros(n_params)
        p1[0] = np.sqrt(N / 2.0) * 0.9
        if K >= 2:
            p1[1] = np.sqrt(N / 2.0) * 0.3
            p1[K] = np.random.rand() * 2 * np.pi
        for j in range(2, K):
            p1[j] = 0.005 * np.random.rand()
            if j >= 1:
                p1[K + j - 1] = np.random.rand() * 2 * np.pi
        starts.append(p1)
        
        # Start 2: Random magnitudes, random phases
        p2 = np.zeros(n_params)
        p2[0] = np.sqrt(N / 2.0) * (0.8 + 0.4 * np.random.rand())
        for j in range(1, K):
            p2[j] = np.random.rand() * 0.3
            p2[K + j - 1] = np.random.rand() * 2 * np.pi
        starts.append(p2)
        
        # Start 3: Structured - geometric decay in magnitudes
        p3 = np.zeros(n_params)
        p3[0] = np.sqrt(N / 2.0) * 0.85
        for j in range(1, K):
            p3[j] = 0.25 * (0.5 ** j)
            p3[K + j - 1] = 2.0 * np.pi * j / K
        starts.append(p3)
        
        # Try each start with L-BFGS-B
        for p_start in starts:
            if time.time() - t0 > _DEADLINE - 15:
                break
            
            try:
                res = minimize(
                    _spectral_objective,
                    p_start,
                    args=(support,),
                    method='L-BFGS-B',
                    bounds=[(0.0, None)] * K + [(0.0, 2.0 * np.pi)] * (K - 1),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.fun < best_local_score:
                    h_cand = _h_from_spectral(res.x[:K], res.x[K:], support)
                    h_cand = _fix_sum(h_cand)
                    s_cand = _score(h_cand)
                    if s_cand < best_local_score:
                        best_local_score = s_cand
                        best_local = h_cand
            except Exception:
                pass
        
        # If K is small enough, also try differential evolution
        if use_de and K <= 8 and time.time() - t0 < _DEADLINE - 20:
            try:
                bounds_de = [(0.0, np.sqrt(N) * 2.0)] * K + [(0.0, 2.0 * np.pi)] * (K - 1)
                res_de = differential_evolution(
                    _spectral_objective,
                    bounds_de,
                    args=(support,),
                    seed=42,
                    maxiter=100,
                    tol=1e-8,
                    popsize=10,
                    polish=True
                )
                if res_de.fun < best_local_score:
                    h_cand = _h_from_spectral(res_de.x[:K], res_de.x[K:], support)
                    h_cand = _fix_sum(h_cand)
                    s_cand = _score(h_cand)
                    if s_cand < best_local_score:
                        best_local_score = s_cand
                        best_local = h_cand
            except Exception:
                pass
        
        return best_local, best_local_score
    
    # Define various supports to try
    supports_to_try = []
    
    # Consecutive low frequencies (including DC)
    for K in [3, 4, 5, 6, 8, 10]:
        supports_to_try.append(list(range(K)))
    
    # DC + symmetric pairs around specific frequencies
    for center in [1, 2, 3, 5, 8, 10, 15, 20]:
        for width in [1, 2, 3]:
            freqs = [0]
            for d in range(width):
                if center - d > 0:
                    freqs.append(center - d)
                if center + d < N // 2:
                    freqs.append(center + d)
            freqs = sorted(set(freqs))
            if 0 in freqs:
                supports_to_try.append(freqs)
    
    # Random sparse supports
    np.random.seed(99)
    for trial in range(15):
        K = np.random.randint(3, 12)
        if K == 1:
            supports_to_try.append([0])
        else:
            freqs = np.random.choice(np.arange(1, N // 2 + 1), size=K - 1, replace=False)
            supports_to_try.append(sorted([0] + list(freqs)))
    
    # Harmonic supports (multiples of a base frequency)
    for base in [1, 2, 3, 4]:
        for n_harm in [2, 3, 4, 5]:
            freqs = [0]
            for m in range(1, n_harm + 1):
                f = base * m
                if f < N // 2:
                    freqs.append(f)
            supports_to_try.append(sorted(set(freqs)))
    
    # Optimize each support
    for support in supports_to_try:
        if time.time() - t0 > _DEADLINE - 10:
            break
        
        K = len(support)
        use_de = (K <= 6)
        
        h_opt, s_opt = _optimize_support(support, n_starts=4, use_de=use_de)
        
        if h_opt is not None and s_opt < best_score:
            best_h = h_opt
            best_score = s_opt
    
    # Final time-domain polish on the best spectral solution
    if best_h is not None:
        for attempt in range(2):
            if time.time() - t0 > _DEADLINE - 5:
                break
            polished = _polish_time_domain(best_h, max_iter=150, ftol=1e-11)
            s_pol = _score(polished)
            if s_pol < best_score:
                best_h = polished
                best_score = s_pol
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    return best_h
