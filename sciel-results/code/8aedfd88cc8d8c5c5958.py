import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _soft_max(conv, beta=50.0):
    """Compute soft-max of conv and its gradient."""
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    sm = max_val + np.log(sum_exp) / beta
    grad_sm = exp_shifted / sum_exp
    return sm, grad_sm

def _ratio(h, beta=50.0):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution).
    h is assumed to be non-negative. The ratio is scale-invariant."""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    sm, _ = _soft_max(conv, beta)
    if sm < 1e-15:
        return 0.0
    return l2_sq / sm

def _dft_parameterize(mags, phases, active_bins):
    """Reconstruct h from spectral magnitudes and phases on active bins.
    mags: array of magnitudes for active bins (length K)
    phases: array of phases for active bins (length K)
    active_bins: array of bin indices
    Returns h (length N), guaranteed non-negative via |IDFT|^2."""
    spectrum = np.zeros(N, dtype=complex)
    for i, bin_idx in enumerate(active_bins):
        spectrum[bin_idx] = mags[i] * np.exp(1j * phases[i])
    # Ensure Hermitian symmetry for real-valued inverse
    # For a real signal: X[k] = conj(X[N-k])
    # We'll handle this by only using positive frequency bins and mirroring
    # Actually, let's just do IDFT and take the real part, then square
    x = np.fft.ifft(spectrum).real
    # h = x^2 ensures non-negativity
    h = x ** 2
    return np.maximum(h, 0)

def _optimize_from_z(z, active_bins, beta=50.0, maxiter=80):
    """Optimize magnitudes and phases of active spectral bins.
    z: initial complex spectrum values for active bins (length K)
    active_bins: array of bin indices
    Returns optimized h."""
    K = len(active_bins)
    mags_init = np.abs(z)
    phases_init = np.angle(z)
    
    def _objective(params):
        mags = np.maximum(params[:K], 1e-10)
        phases = params[K:]
        h = _dft_parameterize(mags, phases, active_bins)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 1e6
        h_norm = h / l1
        l2_sq = np.sum(h_norm ** 2)
        conv = _autoconv(h_norm)
        sm, _ = _soft_max(conv, beta)
        if sm < 1e-15:
            return 1e6
        return -l2_sq / sm
    
    x0 = np.concatenate([mags_init, phases_init])
    bounds = [(1e-10, 10.0)] * K + [(-np.pi, np.pi)] * K
    
    try:
        result = minimize(
            _objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
    except Exception:
        return None
    
    mags_opt = np.maximum(result.x[:K], 1e-10)
    phases_opt = result.x[K:]
    h_opt = _dft_parameterize(mags_opt, phases_opt, active_bins)
    return h_opt

def _local_optimize(h_init, beta=50.0, maxiter=60):
    """Local optimization in h-space using g = sqrt(h) parameterization."""
    g_init = np.sqrt(np.maximum(h_init, 0))
    
    def _obj(g):
        h = g ** 2
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 1e6
        h_norm = h / l1
        l2_sq = np.sum(h_norm ** 2)
        conv = _autoconv(h_norm)
        sm, _ = _soft_max(conv, beta)
        if sm < 1e-15:
            return 1e6
        return -l2_sq / sm
    
    try:
        result = minimize(
            _obj, g_init,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
    except Exception:
        return None
    
    h_opt = result.x ** 2
    return h_opt

def _gaussian_profile(sigma, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return h

def _lorentzian_profile(gamma, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return h

def _power_profile(alpha, center=0.0, width=0.15):
    """Power-law decaying profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = np.abs(x - center)
    h = np.exp(-alpha * t / width)
    return h

def construct():
    start_time = time.time()
    best_heights = np.ones(N).copy()
    best_score = _ratio(best_heights)
    
    rng = np.random.RandomState(42)
    beta = 50.0
    
    # Strategy 1: Sparse spectral representation with K active bins
    # Try different numbers of active bins and different bin selections
    K_values = [2, 3, 4, 5, 6, 8]
    
    for K in K_values:
        if _time_left(start_time) < 20:
            break
        
        # Try different sets of active bins
        # Focus on low-frequency bins for smooth profiles
        all_bins = np.arange(N)
        
        for trial in range(3):
            if _time_left(start_time) < 10:
                break
            
            if trial == 0:
                # Low-frequency bins: 0, 1, N-1, N-2, ...
                active_bins = np.array([0, 1, N-1, N-2, N-3, N-4, N-5, N-6][:K])
            elif trial == 1:
                # Random low-mid frequency bins
                low_bins = np.arange(0, N//2 + 1)
                active_bins = rng.choice(low_bins, size=K, replace=False)
            else:
                # Symmetric pairs around 0 and N/2
                bins_list = []
                for b in range(1, K//2 + 1):
                    bins_list.append(b)
                    bins_list.append(N - b)
                if K % 2 == 1:
                    bins_list.append(0)
                active_bins = np.array(bins_list[:K])
            
            # Initialize with random magnitudes and phases
            mags_init = rng.uniform(0.1, 1.0, size=K)
            phases_init = rng.uniform(-np.pi, np.pi, size=K)
            
            # Build initial spectrum
            z_init = mags_init * np.exp(1j * phases_init)
            
            h_opt = _optimize_from_z(z_init, active_bins, beta=beta, maxiter=60)
            
            if h_opt is not None:
                l1 = np.sum(h_opt)
                if l1 > 1e-15:
                    h_opt = h_opt / l1
                    score = _ratio(h_opt, beta)
                    if score > best_score:
                        best_score = score
                        best_heights = h_opt.copy()
            
            # Also try with symmetric phases (real-valued spectrum)
            if _time_left(start_time) > 5:
                mags_init2 = rng.uniform(0.1, 1.0, size=K)
                phases_init2 = np.zeros(K)  # All phases zero -> real spectrum
                z_init2 = mags_init2 * np.exp(1j * phases_init2)
                h_opt2 = _optimize_from_z(z_init2, active_bins, beta=beta, maxiter=40)
                if h_opt2 is not None:
                    l1 = np.sum(h_opt2)
                    if l1 > 1e-15:
                        h_opt2 = h_opt2 / l1
                        score2 = _ratio(h_opt2, beta)
                        if score2 > best_score:
                            best_score = score2
                            best_heights = h_opt2.copy()
    
    # Strategy 2: Local optimization from diverse initializations
    initializations = []
    
    for sigma in [0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
        for center in [-0.15, -0.05, 0.0, 0.05, 0.15]:
            h = _gaussian_profile(sigma, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    for gamma in [0.02, 0.05, 0.1]:
        for center in [-0.1, 0.0, 0.1]:
            h = _lorentzian_profile(gamma, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    for alpha in [1.0, 2.0, 3.0, 5.0]:
        h = _power_profile(alpha)
        if np.sum(h) > 0:
            initializations.append(h.copy())
    
    initializations.append(np.ones(N).copy())
    
    for h_init in initializations:
        if _time_left(start_time) < 8:
            break
        
        l1 = np.sum(h_init)
        if l1 == 0:
            continue
        h_init = h_init / l1
        
        h_opt = _local_optimize(h_init, beta=beta, maxiter=50)
        if h_opt is not None:
            l1_opt = np.sum(h_opt)
            if l1_opt > 1e-15:
                h_opt = h_opt / l1_opt
                score = _ratio(h_opt, beta)
                if score > best_score:
                    best_score = score
                    best_heights = h_opt.copy()
    
    # Strategy 3: Local optimization from best found so far with perturbations
    if _time_left(start_time) > 15:
        g_best = np.sqrt(best_heights)
        for _ in range(5):
            if _time_left(start_time) < 5:
                break
            g_pert = g_best + 0.05 * rng.randn(N)
            h_pert = g_pert ** 2
            h_opt = _local_optimize(h_pert, beta=beta, maxiter=40)
            if h_opt is not None:
                l1_opt = np.sum(h_opt)
                if l1_opt > 1e-15:
                    h_opt = h_opt / l1_opt
                    score = _ratio(h_opt, beta)
                    if score > best_score:
                        best_score = score
                        best_heights = h_opt.copy()
    
    # Final safety: ensure we return valid non-negative heights
    if best_score <= 0:
        best_heights = np.ones(N).copy()
    
    # Ensure all values are non-negative
    best_heights = np.maximum(best_heights, 0)
    
    return best_heights.tolist()
