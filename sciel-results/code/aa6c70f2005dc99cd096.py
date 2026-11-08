"""Spectral phase-rotation + coordinate descent for signed autocorrelation minimization.
Objective: minimize max|autoconv(h)| / (sum h)^2 over signed step heights h of length N.
For ±1 signals, lag-0 autoconv = N, so we want all other lags near zero.
New approach: use FFT-based spectral initialization (minimize L2 of sidelobes),
then coordinate descent with fast delta updates.
"""
import time
import numpy as np

N = 400
_DEADLINE = 128.0  # safety margin under 150s kill


def _fft_autoconv(h):
    """Linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    C = np.fft.irfft(H * H, n=fft_size)
    return C[:size]


def _eval(h):
    """Return (value, c) where value = max|c| / (sum h)^2."""
    c = _fft_autoconv(h)
    m = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    denom = s * s if s * s > 1e-12 else 1e-12
    return m / denom, c


def _spectral_init(n, seed=0):
    """
    Spectral initialization: find ±1 sequence minimizing L2 of sidelobe energy.
    This is equivalent to finding the real part of a random phase in frequency domain,
    then thresholding to ±1. We iterate a few times to refine.
    """
    np.random.seed(seed)
    # Start with random ±1
    h = np.random.choice([-1.0, 1.0], size=n)
    
    # Iterate: compute autoconv, find the "ideal" signal that would have flat spectrum
    # (except DC), threshold to ±1
    for it in range(30):
        c = _fft_autoconv(h)
        # The sidelobes are c[1:] and c[:0] (i.e., all except lag 0)
        # We want to reduce the max/energy of sidelobes
        # Compute the "error" at each position and try to flip
        # Simple spectral approach: in frequency domain, flatten
        H = np.fft.rfft(h, n=2 * n)
        # Keep DC, flatten the rest
        H_flat = H.copy()
        H_flat[1:] = np.mean(np.abs(H_flat[1:])) * np.exp(1j * np.angle(H_flat[1:]))
        h_new = np.real(np.fft.irfft(H_flat, n=2 * n))[:n]
        # Threshold to ±1
        h = np.where(h_new >= 0, 1.0, -1.0)
    
    return h


def _golay_pair(bits=None):
    """
    Golay complementary pair via recursive construction.
    """
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < N:
        a_new = np.concatenate([a, b])
        b_new = np.concatenate([a, -b])
        a, b = a_new, b_new
    return a[:N].copy(), b[:N].copy()


def _rudin_shapiro(n):
    """Rudin-Shapiro sequence."""
    r = np.array([1.0])
    s = np.array([1.0])
    while len(r) < n:
        r_new = np.concatenate([r, s])
        s_new = np.concatenate([r, -s])
        r, s = r_new, s_new
    return r[:n].copy()


def _delta_update(h, idx):
    """
    Compute the delta in autoconvolution if we flip h[idx].
    Returns (delta_array, d) where d = new_h[idx] - old_h[idx].
    For ±1, d = ±2.
    delta_c[L] = d * (h[idx+L] if valid + h[idx-L] if valid) for L != 0
    delta_c[0] = 0 (since h[idx]^2 doesn't change for ±1)
    """
    old = h[idx]
    target = -old
    d = target - old  # ±2
    size = 2 * N - 1
    delta = np.zeros(size)
    
    # L ranges from -(N-1) to (N-1)
    Ls = np.arange(-(N - 1), N)
    # For each L, delta_c at position (L + N - 1)
    term1 = np.zeros(len(Ls))
    term2 = np.zeros(len(Ls))
    valid1 = (idx + Ls >= 0) & (idx + Ls < N)
    valid2 = (idx - Ls >= 0) & (idx - Ls < N)
    # Use safe indexing
    idx1 = np.where(valid1, idx + Ls, 0)
    idx2 = np.where(valid2, idx - Ls, 0)
    term1[valid1] = h[idx1[valid1]]
    term2[valid2] = h[idx2[valid2]]
    delta_vals = d * (term1 + term2)
    # At L=0, the two terms are the same, so we double-counted.
    # But h[idx]^2 doesn't change, so delta_c[0] should be 0.
    zero_pos = N - 1
    delta_vals[zero_pos] = 0.0
    delta[:] = delta_vals
    return delta, d


def _coordinate_descent(h, c, max_iter=200000, deadline=128.0):
    """
    Coordinate descent: greedily flip signs to reduce max|c|/(sum h)^2.
    Use fast delta updates.
    """
    start = time.time()
    s = float(np.sum(h))
    current_val, _ = _eval(h)
    current_max = float(np.max(np.abs(c)))
    denom = s * s if s * s > 1e-12 else 1e-12
    
    # Track the positions with high |c| to prioritize
    # Use a simple approach: cycle through indices, try flip, accept if improves
    
    # For speed, we'll use a "hot" list of indices that are likely to help
    # (those where flipping reduces the max)
    
    np.random.seed(123)
    order = np.random.permutation(N)
    oi = 0
    n_flips = 0
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        if oi >= N:
            oi = 0
            order = np.random.permutation(N)
        
        idx = int(order[oi])
        oi += 1
        
        delta, d = _delta_update(h, idx)
        new_c = c + delta
        new_s = s + d
        new_denom = new_s * new_s if new_s * new_s > 1e-12 else 1e-12
        new_max = float(np.max(np.abs(new_c)))
        new_val = new_max / new_denom
        
        if new_val < current_val - 1e-12:
            h[idx] = -h[idx]
            c = new_c
            s = new_s
            denom = new_denom
            current_max = new_max
            current_val = new_val
            n_flips += 1
    
    return h, current_val, n_flips


def _multi_start_cd(h_init, max_iter=50000, deadline=128.0):
    """Coordinate descent with early stopping on stagnation."""
    h, val, flips = _coordinate_descent(h_init, _fft_autoconv(h_init), max_iter, deadline)
    return h, val


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    best_h = None
    best_val = float('inf')
    best_c = None
    
    # Generate multiple initial candidates
    candidates = []
    
    # Golay pair
    a, b = _golay_pair()
    candidates.append(a)
    candidates.append(b)
    
    # Rudin-Shapiro
    candidates.append(_rudin_shapiro(N))
    
    # Spectral init with different seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        candidates.append(_spectral_init(N, seed=seed))
    
    # Random
    np.random.seed(42)
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates
    for h_cand in candidates:
        if time.time() > deadline - 10:
            break
        val, c = _eval(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
            best_c = c.copy()
    
    # Coordinate descent on the best
    if best_h is not None and time.time() < deadline - 5:
        h_cd, val_cd = _multi_start_cd(best_h.copy(), max_iter=100000, deadline=deadline)
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd
    
    # Try coordinate descent on a few other good candidates
    # Sort candidates by their initial eval
    cand_scores = []
    for h_cand in candidates:
        val, c = _eval(h_cand)
        cand_scores.append((val, h_cand))
    cand_scores.sort(key=lambda x: x[0])
    
    # Try CD on top 3 distinct candidates (excluding the best already tried)
    tried = [id(best_h)]
    for val_c, h_c in cand_scores[:5]:
        if time.time() > deadline - 5:
            break
        if id(h_c) in tried:
            continue
        h_try, val_try = _multi_start_cd(h_c.copy(), max_iter=50000, deadline=deadline)
        tried.append(id(h_try))
        if val_try < best_val:
            best_val = val_try
            best_h = h_try
    
    # Final pass of CD on the overall best
    if time.time() < deadline - 5:
        _, c_final = _eval(best_h)
        h_final, val_final = _coordinate_descent(best_h.copy(), c_final.copy(), max_iter=200000, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
