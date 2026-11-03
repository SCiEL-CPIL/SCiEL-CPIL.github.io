"""Erdos minimum overlap: vectorized smooth-surrogate + bisection on t.
Uses log-sum-exp soft-max of (C(h) - t) as a differentiable surrogate,
with bisection on t and projected gradient / L-BFGS-B for h.
Gradient of the smooth max is computed via FFT-convolution for speed.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N

# Precompute FFT of basis for gradient
# d r_k / d h_j: for each k, the "kernel" is:
#   for j in [k, N-1]: h[j-k]  -> shift of h
#   for j in [0, N-k-1]: h[j+k] -> shift of h
# We can compute g_h = sum_k g_c[k] * (shifted contributions)
# More efficiently: note that the full gradient is a convolution-like operation.
# Let's think: g_h[j] = factor * sum_k g_c[k] * (h[j-k] * 1_{j>=k} + h[j+k] * 1_{j+k<N})
# This is: factor * sum_k g_c[k] * h[j-k] * 1_{j>=k} + factor * sum_k g_c[k] * h[j+k] * 1_{j+k<N}
# Let a[k] = g_c[k] for k=0..N-1
# First term: sum_{k=0}^{j} a[k] * h[j-k]  = (a * h)[j] (linear convolution, but only up to j)
# Second term: sum_{k=0}^{N-1-j} a[k] * h[j+k]
# Actually let's just use the direct approach but vectorized with FFT.

def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=_NFFT)
    return corr[:N]

def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    r = _autocorr(h)
    return (2.0 / N) * (N / 2.0 - r)

def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))

def _project(h):
    """Project h to sum=N/2, values in [0,1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    h = np.clip(h, 0.0, 1.0)
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        cnt = np.sum(mask)
        if cnt > 0:
            h[mask] += diff / cnt
            h = np.clip(h, 0.0, 1.0)
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _spectral_start(frequencies, amplitudes, phases, rng):
    """Generate a diverse start from spectral parameters."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    for f, a, p in zip(frequencies, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)

def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)

def _perturb_start(h, rng, sigma=0.05):
    """Add small random perturbation and project."""
    h_pert = h + sigma * rng.normal(0, 1, N)
    return _project(h_pert)

def _reduce_active_c(h, active_indices, step=0.01):
    """Try to reduce C_k for active indices by adjusting h coordinates.
    
    For each active k, C_k = (2/N)*(N/2 - sum_{i=k}^{N-1} h[i]*h[i-k])
    To reduce C_k, we need to increase r_k = sum_{i=k}^{N-1} h[i]*h[i-k].
    
    Strategy: for the most active k, find pairs (i, i-k) and try to increase
    their product while maintaining constraints.
    """
    cs = _overlap_values(h)
    best_score = float(np.max(cs))
    
    for k in active_indices:
        if k == 0:
            continue  # C_0 is fixed by sum constraint
        
        # r_k = sum_{i=k}^{N-1} h[i]*h[i-k]
        # d(r_k)/d(h[j]) = h[j-k] (if j>=k) + h[j+k] (if j+k<N)
        # To increase r_k, increase h[j] where gradient is positive
        
        # Compute gradient of r_k w.r.t. h
        grad_rk = np.zeros(N)
        for j in range(k, N):
            grad_rk[j] += h[j - k]
        for j in range(0, N - k):
            grad_rk[j] += h[j + k]
        
        # Find the coordinate with largest positive gradient
        pos_mask = grad_rk > 0
        if not np.any(pos_mask):
            continue
        
        # Find best coordinate to increase
        best_j = np.argmax(np.where(pos_mask, grad_rk, -np.inf))
        
        # Try increasing h[best_j] by step
        h_trial = h.copy()
        h_trial[best_j] = min(1.0, h_trial[best_j] + step)
        
        # Project to maintain sum constraint
        h_trial = _project(h_trial)
        
        cs_trial = _overlap_values(h_trial)
        new_max = float(np.max(cs_trial))
        
        if new_max < best_score - 1e-12:
            h = h_trial
            best_score = new_max
    
    return h, best_score

def _coordinate_descent(h0, max_iter=5000, tol=1e-10, time_limit=None, t_start=None):
    """Coordinate descent on active set to minimize max C_k.
    
    Iteratively identifies the most active C_k values and tries to reduce them
    via single-coordinate moves.
    """
    h = h0.copy()
    h = _project(h)
    best_h = h.copy()
    best_score = _score(h)
    
    if t_start is None:
        t_start = time.time()
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - t_start) > time_limit:
            break
        
        cs = _overlap_values(h)
        current_max = float(np.max(cs))
        
        if current_max < best_score:
            best_score = current_max
            best_h = h.copy()
        
        # Find active indices (within tolerance of max)
        threshold = current_max - 1e-6
        active_mask = cs >= threshold
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            active_indices = np.array([np.argmax(cs)])
        
        # Sort by C_k value (most active first)
        active_indices = active_indices[np.argsort(cs[active_indices])[::-1]]
        
        # Try to reduce the most active C_k values
        h_new, new_score = _reduce_active_c(h, active_indices, step=0.01)
        
        if new_score >= best_score - tol:
            # Try smaller steps
            h_new, new_score = _reduce_active_c(h, active_indices, step=0.001)
        
        if new_score < best_score - 1e-12:
            h = h_new
            best_score = new_score
            best_h = h.copy()
        else:
            # Try perturbations
            rng = np.random.default_rng(it)
            h_pert = _perturb_start(h, rng, sigma=0.02)
            h_pert, s_pert = _reduce_active_c(h_pert, active_indices, step=0.005)
            if s_pert < best_score - 1e-12:
                h = h_pert
                best_score = s_pert
                best_h = h.copy()
            else:
                # Stagnation detected, break early
                if it > 100:
                    break
    
    return best_h, best_score

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s kill
    rng = np.random.default_rng(42)

    best_h = None
    best_score = float('inf')

    def _update(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        return s

    # Generate diverse spectral starts
    i = np.arange(N)
    freq_sets = [
        ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06], [0.0, 0.5, 1.0, 1.5]),
        ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05], [0.2, 0.7, 1.2, 1.7]),
        ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04], [0.1, 0.6, 1.1, 1.6, 2.1]),
        ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04], [0.3, 0.8, 1.3, 1.8, 2.3]),
        ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
        ([3, 5, 7, 11], [0.12, 0.10, 0.08, 0.06], [0.4, 0.9, 1.4, 1.9]),
        ([1, 2, 3, 4, 5], [0.12, 0.10, 0.08, 0.06, 0.05], [0.15, 0.65, 1.15, 1.65, 2.15]),
        ([2, 4, 6, 8, 10], [0.10, 0.09, 0.08, 0.07, 0.06], [0.25, 0.75, 1.25, 1.75, 2.25]),
    ]
    
    # Phase 1: Evaluate all spectral starts
    for freqs, amps, phases in freq_sets:
        if time.time() > deadline - 30:
            break
        h_start = _spectral_start(freqs, amps, phases, rng)
        _update(h_start)

    # Add random starts
    for _ in range(4):
        if time.time() > deadline - 30:
            break
        h_rand = _random_start(rng)
        _update(h_rand)

    # Phase 2: Coordinate descent from each promising start
    # Sort starts by initial score and refine the best ones
    candidate_starts = []
    if best_h is not None:
        candidate_starts.append(('best', best_h.copy(), best_score))
    
    for freqs, amps, phases in freq_sets:
        if time.time() > deadline - 20:
            break
        h_start = _spectral_start(freqs, amps, phases, rng)
        s_start = _score(h_start)
        candidate_starts.append(('spec', h_start, s_start))
    
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        h_rand = _random_start(rng)
        s_rand = _score(h_rand)
        candidate_starts.append(('rand', h_rand, s_rand))
    
    # Sort by score (best first)
    candidate_starts.sort(key=lambda x: x[2])
    
    # Refine top candidates with coordinate descent
    time_per_cd = 15.0  # seconds per coordinate descent run
    for name, h_start, s_start in candidate_starts[:6]:
        if time.time() > deadline - 10:
            break
        h_cd, s_cd = _coordinate_descent(
            h_start, 
            max_iter=3000, 
            tol=1e-10,
            time_limit=time_per_cd,
            t_start=time.time()
        )
        _update(h_cd)
    
    # Phase 3: Final polish on best with more iterations
    if best_h is not None and time.time() < deadline - 5:
        h_polish, s_polish = _coordinate_descent(
            best_h,
            max_iter=5000,
            tol=1e-12,
            time_limit=10.0,
            t_start=time.time()
        )
        _update(h_polish)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
