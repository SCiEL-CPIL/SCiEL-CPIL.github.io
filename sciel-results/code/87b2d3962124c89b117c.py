import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import eigh_tridiagonal

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _build_conv_matrix():
    """Build the N x N matrix C such that conv(h,h)[k] = h^T C_k h for each lag k.
    
    For full convolution of length 2N-1:
    conv(h,h)[m] = sum_i h[i]*h[i+m-N+1] for appropriate ranges.
    
    More precisely, conv(h,h)[m] = sum_{i} h[i] * h[m-i] where valid indices.
    For m = 0..2N-2:
      conv[m] = sum_{i=max(0,m-N+1)}^{min(N-1,m)} h[i]*h[m-i]
    
    We can write conv[m] = h^T Q_m h where Q_m is a symmetric matrix.
    """
    # We'll build the matrix for the maximum lag.
    # Actually, let's think differently.
    # conv(h,h) is the autoconvolution. In Fourier domain:
    # conv(h,h) = ifft(fft(h)^2)
    # So max(conv(h,h)) is a function of h.
    # 
    # For the QP approach, we want to minimize max_k (h^T Q_k h) subject to sum(h)=1
    # where Q_k are symmetric matrices.
    #
    # A simpler approach: use the fact that the peak of autoconvolution relates to
    # the L-infinity norm of the autoconvolution.
    #
    # Let's build the matrix for the DC component (lag N-1):
    # conv[N-1] = sum_i h[i]^2 = h^T h
    # This is the "peak" when h is constant.
    #
    # For general h, the max over all lags is what matters.
    # We can use a trust-region approach where we approximate the max as a
    # smooth function and solve a QP subproblem.
    
    pass  # Placeholder - will use direct optimization instead


def _eigen_init():
    """Generate initial candidate using eigenstructure of the convolution operator."""
    # The autoconvolution in Fourier domain: ifft(|fft(h)|^2)
    # To minimize the peak, we want |fft(h)|^2 to be as flat as possible.
    # This means h should be close to a white noise sequence (in frequency domain).
    #
    # Strategy: start with a chirp or random phase in frequency domain,
    # then inverse FFT to get h.
    
    rng = np.random.default_rng(42)
    
    # Method 1: Random phases in frequency domain
    best_h = None
    best_val = 1e18
    
    for trial in range(50):
        # Random phase mask
        freq = np.fft.rfft(np.zeros(N))
        freq[0] = 0  # Zero DC
        for k in range(1, N//2 + 1):
            freq[k] = np.exp(1j * rng.uniform(0, 2 * np.pi))
        
        h = np.fft.irfft(freq, N)
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Method 2: Flat spectrum with controlled DC
    for dc_level in np.linspace(-0.5, 0.5, 11):
        freq = np.ones(N//2 + 1, dtype=complex)
        freq[0] = dc_level
        h = np.fft.irfft(freq, N)
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _smooth_objective(h, beta=30.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _trust_region_solve(h0, t0, radius, beta=50.0):
    """Solve a trust-region subproblem to find a better h within a ball around h0."""
    # This is a simplified approach: use L-BFGS-B with bounds
    # to stay within a trust region.
    
    h0_center = h0.copy()
    s0 = np.sum(h0)
    
    # Define bounds to keep h within radius of h0
    lower = h0_center - radius
    upper = h0_center + radius
    
    try:
        res = minimize(
            lambda h: _smooth_objective(h, beta=beta),
            h0,
            jac=lambda h: _smooth_grad(h, beta=beta)[0],
            method='L-BFGS-B',
            bounds=list(zip(lower, upper)),
            options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            v = _value(res.x)
            if v < t0:
                return res.x, v
    except Exception:
        pass
    
    return h0, t0


def _smooth_grad(h, beta=30.0):
    """Gradient of the smooth objective."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    obj = abs(2.0 * N * soft_max / (s * s))
    
    # d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0
    d_soft_max = np.zeros(N)
    for j in range(N):
        k_start = j
        k_end = min(j + N - 1, 2 * N - 2)
        if k_start > k_end:
            continue
        d_soft_max[j] = 2.0 * np.dot(weights[k_start:k_end+1], h[0:k_end-k_start+1])
    
    sign = np.sign(2.0 * N * soft_max / (s * s))
    if sign == 0:
        sign = 1.0
    
    grad = np.zeros(N)
    for j in range(N):
        grad[j] = sign * 2.0 * N * (d_soft_max[j] / (s * s) - 2.0 * soft_max / (s ** 3))
    
    return grad, obj


def _qp_solve_direct(h0, beta=50.0):
    """Direct QP-inspired solve using projected gradient."""
    # Use a projected gradient method with step size adaptation
    h = h0.copy()
    best_h = h0.copy()
    best_val = _value(h0)
    
    step_size = 0.1
    for it in range(200):
        grad, obj = _smooth_grad(h, beta=beta)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Projected gradient step
        h_new = h - step_size * grad
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-12:
            step_size *= 0.5
            continue
        h_new = h_new / s_new
        
        v_new = _value(h_new)
        if v_new < best_val:
            best_val = v_new
            best_h = h_new.copy()
            h = h_new
            step_size = min(step_size * 1.1, 1.0)
        else:
            step_size *= 0.5
    
    return best_h, best_val


def _chirp_h(freq, chirp, dc):
    """Generate chirp signal: h[i] = dc + cos(2*pi*freq*i/N + 0.5*chirp*(i/N)^2)"""
    x = np.arange(N, dtype=float)
    t = x / N
    phase = 2 * np.pi * freq * t + 0.5 * chirp * t * t
    h = dc + np.cos(phase)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return None
    return h / s


def _flat_spectrum_init(max_time=15.0):
    """Initialize using flat-spectrum design with controlled DC null."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    
    # Design h such that its DFT has approximately flat magnitude
    # except for a controlled DC component.
    # This minimizes the peak of the autoconvolution.
    
    for dc_factor in np.linspace(-0.8, 0.8, 17):
        if time.time() - t0 > max_time * 0.8:
            break
        
        # Create frequency domain signal with flat magnitude and controlled DC
        freq_mag = np.ones(N//2 + 1)
        freq_mag[0] = abs(dc_factor)
        
        # Random phases to break symmetry
        rng = np.random.default_rng(int(abs(dc_factor * 1000)))
        freq_phase = rng.uniform(0, 2 * np.pi, N//2 + 1)
        
        freq = freq_mag * np.exp(1j * freq_phase)
        h = np.fft.irfft(freq, N)
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Also try deterministic flat spectrum
    for dc_factor in np.linspace(-0.5, 0.5, 11):
        if time.time() - t0 > max_time:
            break
        
        freq = np.ones(N//2 + 1, dtype=complex)
        freq[0] = dc_factor
        h = np.fft.irfft(freq, N)
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _chirp_grid_search(max_time=20.0):
    """Grid search over chirp parameters."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    
    # Grid over frequency, chirp rate, and DC bias
    freqs = np.arange(1, 80, 1)
    chirps = np.arange(-200, 201, 5)
    dcs = np.linspace(-0.3, 0.3, 13)
    
    for freq in freqs:
        if time.time() - t0 > max_time * 0.7:
            break
        for chirp in chirps:
            if time.time() - t0 > max_time * 0.9:
                break
            for dc in dcs:
                h = _chirp_h(freq, chirp, dc)
                if h is None:
                    continue
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _multi_freq_cosine_init(max_time=15.0):
    """Initialize using sum of multiple cosine modes with optimized amplitudes."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    
    x = np.arange(N)
    
    # Try different combinations of cosine modes
    mode_sets = [
        [1, 2, 3],
        [1, 3, 5],
        [2, 4, 6],
        [1, 2, 3, 4],
        [1, 3, 5, 7],
        [2, 5, 8, 11],
        [1, 4, 7, 10, 13],
    ]
    
    for modes in mode_sets:
        if time.time() - t0 > max_time * 0.8:
            break
        
        # Grid search over amplitudes
        n_modes = len(modes)
        amp_range = np.linspace(-1.0, 1.0, 5)
        
        for amps in np.ndindex(*[5] * n_modes):
            if time.time() - t0 > max_time:
                break
            
            h = np.ones(N) * 0.1  # Small DC offset
            for m, a_idx in zip(modes, amps):
                h += amp_range[a_idx] * np.cos(2 * np.pi * m * x / N)
            
            s = np.sum(h)
            if abs(s) < 1e-12:
                continue
            h = h / s
            v = _value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _chirp_refine(h0, max_time=15.0):
    """Refine chirp parameters using local optimization."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    # Try L-BFGS-B on the full h vector starting from chirp init
    try:
        def obj_fn(h):
            return _smooth_objective(h, beta=40.0)
        
        def grad_fn(h):
            g, _ = _smooth_grad(h, beta=40.0)
            return g
        
        res = minimize(
            obj_fn,
            best_h,
            jac=grad_fn,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            v = _value(res.x)
            if v < best_val:
                best_val = v
                best_h = res.x.copy()
    except Exception:
        pass
    
    if time.time() - t0 < max_time * 0.6:
        try:
            def obj_fn2(h):
                return _smooth_objective(h, beta=80.0)
            
            def grad_fn2(h):
                g, _ = _smooth_grad(h, beta=80.0)
                return g
            
            res = minimize(
                obj_fn2,
                best_h,
                jac=grad_fn2,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 800, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _diverse_init(max_time=20.0):
    """Generate diverse initial candidates."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    rng = np.random.default_rng(123)
    
    x = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Single cosine modes
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Chirp signals
    for freq in [5, 10, 20, 30, 50, 75]:
        for chirp in [-100, -50, 0, 50, 100]:
            h = _chirp_h(freq, chirp, 0.0)
            if h is not None:
                candidates.append(h)
    
    # Random Gaussian
    for trial in range(20):
        if time.time() - t0 > max_time * 0.5:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Block structure
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(5):
            if time.time() - t0 > max_time * 0.8:
                break
            rng_block = np.random.default_rng(seed + 1000)
            h = np.zeros(N)
            for b in range(nblocks):
                val = rng_block.normal(0, 1)
                h[b*block_size:(b+1)*block_size] = val
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Evaluate all candidates
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _local_refine(h0, max_time=25.0):
    """Local refinement using smooth objective with multiple betas."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in [30.0, 50.0, 80.0]:
        if time.time() - t0 > max_time * 0.8:
            break
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=beta)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=beta)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, n_starts=8, max_time=30.0):
    """Multi-start local refinement."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        scale = 0.1 + 0.05 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=40.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=40.0)[0],
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 300, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Phase 1: Flat spectrum initialization (key insight from inspirations)
    h_flat, val_flat = _flat_spectrum_init(max_time=10.0)
    
    # Phase 2: Multi-frequency cosine initialization
    h_multi, val_multi = _multi_freq_cosine_init(max_time=10.0)
    
    # Phase 3: Eigen-based initialization
    h_eig, val_eig = _eigen_init()
    
    # Phase 4: Chirp grid search
    h_chirp, val_chirp = _chirp_grid_search(max_time=10.0)
    
    # Phase 5: Diverse initialization
    h_div, val_div = _diverse_init(max_time=10.0)
    
    # Pick best from all initializations
    candidates = [
        (h_flat, val_flat),
        (h_multi, val_multi),
        (h_eig, val_eig),
        (h_chirp, val_chirp),
        (h_div, val_div),
    ]
    
    best_h = candidates[0][0].copy()
    best_val = candidates[0][1]
    for h_c, v_c in candidates[1:]:
        if v_c < best_val:
            best_val = v_c
            best_h = h_c.copy()
    
    # Include seed
    h_seed = np.ones(N)
    v_seed = _value(h_seed)
    if v_seed < best_val:
        best_val = v_seed
        best_h = h_seed.copy()
    
    # Phase 6: QP-inspired direct solve
    if time_left() > 30:
        h_qp, v_qp = _qp_solve_direct(best_h, beta=50.0)
        if v_qp < best_val:
            best_val = v_qp
            best_h = h_qp.copy()
    
    # Phase 7: Trust region refinement
    if time_left() > 25:
        h_tr, v_tr = _trust_region_solve(best_h, best_val, radius=0.5, beta=50.0)
        if v_tr < best_val:
            best_val = v_tr
            best_h = h_tr.copy()
    
    # Phase 8: Chirp-specific refinement
    if time_left() > 20:
        best_h, best_val = _chirp_refine(best_h, max_time=min(15.0, time_left() - 10.0))
    
    # Phase 9: General local refinement
    if time_left() > 25:
        best_h, best_val = _local_refine(best_h, max_time=min(25.0, time_left() - 10.0))
    
    # Phase 10: Multi-start
    if time_left() > 20:
        best_h, best_val = _multi_start(best_h, n_starts=10, max_time=min(30.0, time_left() - 10.0))
    
    # Phase 11: Final high-precision polish
    if time_left() > 10:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=100.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=100.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h


if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
