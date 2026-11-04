"""Second autocorrelation inequality: maximize squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def _autoconvolution_peak(h):
    """Compute the supremum norm of the discrete autoconvolution of h."""
    size = 2 * N - 1
    h_pad = np.zeros(2 * size)
    h_pad[:N] = h
    H = np.fft.fft(h_pad, n=2 * size)
    auto = np.fft.ifft(H * H).real[:size]
    return np.max(np.abs(auto))

def _compute_ratio(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    peak = _autoconvolution_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _two_lobe_profile(delta, sigma, center=24.5):
    """Two narrow Gaussian lobes at center +/- delta with width sigma."""
    x = np.arange(N, dtype=float)
    h1 = np.exp(-0.5 * ((x - (center - delta)) / sigma) ** 2)
    h2 = np.exp(-0.5 * ((x - (center + delta)) / sigma) ** 2)
    h = h1 + h2
    h = np.maximum(h, 0)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def _two_lobe_ratio(delta, sigma):
    """Ratio for two-lobe profile with given parameters."""
    h = _two_lobe_profile(delta, sigma)
    return _compute_ratio(h)

def _optimize_two_lobe(deadline):
    """Optimize the two-lobe structure by varying delta and sigma."""
    best_val = -1
    best_params = (10.0, 2.0)
    
    # Grid search over delta and sigma
    deltas = np.arange(3.0, 24.0, 0.5)
    sigmas = np.arange(0.8, 8.0, 0.3)
    
    for d in deltas:
        if time.time() > deadline - 30:
            break
        for s in sigmas:
            if time.time() > deadline - 30:
                break
            val = _two_lobe_ratio(d, s)
            if val > best_val:
                best_val = val
                best_params = (d, s)
    
    # Refine with local optimization around best grid point
    d0, s0 = best_params
    
    def neg_ratio(params):
        d, s = params
        d = max(0.5, min(24.0, d))
        s = max(0.3, min(10.0, s))
        return -_two_lobe_ratio(d, s)
    
    try:
        result = minimize(
            neg_ratio,
            [d0, s0],
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < neg_ratio([d0, s0]):
            d_opt, s_opt = result.x
            d_opt = max(0.5, min(24.0, d_opt))
            s_opt = max(0.3, min(10.0, s_opt))
            val_opt = _two_lobe_ratio(d_opt, s_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_params = (d_opt, s_opt)
    except Exception:
        pass
    
    h_best = _two_lobe_profile(best_params[0], best_params[1])
    return h_best, best_val

def _sparse_spikes_profile(positions, amplitudes=None):
    """Create a sparse spike profile at given positions."""
    h = np.zeros(N)
    for p in positions:
        p_int = int(round(p))
        if 0 <= p_int < N:
            h[p_int] = 1.0
    if amplitudes is not None:
        for i, p in enumerate(positions):
            p_int = int(round(p))
            if 0 <= p_int < N and i < len(amplitudes):
                h[p_int] = amplitudes[i]
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def _sidon_like_positions(k, seed=42):
    """Generate k positions that form a Sidon-like set (all pairwise sums distinct)."""
    rng = np.random.RandomState(seed)
    best_positions = None
    best_score = -1
    
    for attempt in range(50):
        positions = sorted(rng.choice(N, size=k, replace=False))
        # Check Sidon property: all pairwise sums i+j (i<=j) should be distinct
        sums = set()
        valid = True
        for i in range(len(positions)):
            for j in range(i, len(positions)):
                s = positions[i] + positions[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        if valid:
            # Score by minimum pairwise distance
            min_dist = min(abs(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i+1, len(positions)))
            if min_dist > best_score:
                best_score = min_dist
                best_positions = list(positions)
    
    if best_positions is None:
        # Fallback: evenly spaced
        best_positions = list(np.linspace(2, N-3, k).astype(int))
    
    return best_positions

def _optimize_sparse(deadline):
    """Try sparse spike configurations with Sidon-like positions."""
    best_val = -1
    best_h = None
    
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 15:
            break
        
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            positions = _sidon_like_positions(k, seed=seed)
            h = _sparse_spikes_profile(positions)
            val = _compute_ratio(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
    
    return best_h, best_val

def _optimize_from_start(h0, deadline):
    """Run L-BFGS-B from a starting point, respecting deadline."""
    h0 = np.asarray(h0, dtype=float).copy()
    bounds = [(0, None)] * N
    
    best_h = h0.copy()
    best_val = _compute_ratio(h0)
    
    try:
        def obj(h):
            return -_compute_ratio(h)
        
        result = minimize(
            obj,
            h0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'maxfun': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        val = _compute_ratio(result.x)
        if val > best_val:
            best_val = val
            best_h = result.x.copy()
    except Exception:
        pass
    
    return best_h, best_val

def _polish(h, deadline):
    """High-precision polish pass with tighter tolerances."""
    h = np.asarray(h, dtype=float).copy()
    bounds = [(0, None)] * N
    
    try:
        def obj(h):
            return -_compute_ratio(h)
        
        result = minimize(
            obj,
            h,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
        )
        val = _compute_ratio(result.x)
        val_orig = _compute_ratio(h)
        if val > val_orig:
            return result.x, val
    except Exception:
        pass
    return h, _compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = -1
    
    # Strategy 1: Two-lobe Gaussian structure (directive)
    if time.time() < deadline - 60:
        h_lobe, val_lobe = _optimize_two_lobe(deadline - 40)
        if val_lobe > best_val:
            best_val = val_lobe
            best_h = h_lobe.copy()
    
    # Strategy 2: Sparse Sidon-like spikes
    if time.time() < deadline - 40:
        h_sparse, val_sparse = _optimize_sparse(deadline - 20)
        if val_sparse > best_val:
            best_val = val_sparse
            best_h = h_sparse.copy()
    
    # Strategy 3: Optimize from diverse smooth initializations
    if best_h is not None and time.time() < deadline - 20:
        h_opt, val_opt = _optimize_from_start(best_h, deadline - 10)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Also try optimizing from a couple of hand-crafted starts
    inits = []
    # Single narrow Gaussian
    x = np.arange(N, dtype=float)
    h_gauss = np.exp(-0.5 * ((x - 24.5) / 3.0) ** 2)
    h_gauss = h_gauss / np.sum(h_gauss)
    inits.append(h_gauss)
    
    # Two well-separated Gaussians
    h_bi1 = np.exp(-0.5 * ((x - 8) / 2.5) ** 2) + np.exp(-0.5 * ((x - 42) / 2.5) ** 2)
    h_bi1 = h_bi1 / np.sum(h_bi1)
    inits.append(h_bi1)
    
    # Three peaks
    h_tri = (np.exp(-0.5 * ((x - 6) / 2.0) ** 2) + 
             np.exp(-0.5 * ((x - 25) / 2.0) ** 2) + 
             np.exp(-0.5 * ((x - 44) / 2.0) ** 2))
    h_tri = h_tri / np.sum(h_tri)
    inits.append(h_tri)
    
    for h0 in inits:
        if time.time() > deadline - 15:
            break
        h_opt, val = _optimize_from_start(h0, deadline - 10)
        if val > best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Polish the best result
    if best_h is not None and time.time() < deadline - 5:
        best_h, best_val = _polish(best_h, deadline - 2)
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = _compute_ratio(best_h)
    
    # Ensure nonnegativity and return as list
    best_h = np.maximum(best_h, 0)
    return list(best_h)
