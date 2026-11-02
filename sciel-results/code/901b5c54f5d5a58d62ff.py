import numpy as np
import time

N = 50
_TIME_LIMIT = 145.0  # safety margin below 150s kill
_start_time = time.time()

def _autoconv_max(h):
    """Compute max of linear autoconvolution via FFT (fast)."""
    # Linear convolution length is 2*N - 1
    size = 2 * N - 1
    # Use FFT size that's a power of 2 for speed
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    return np.max(conv)

def _score(h):
    """(sum h_i^2) / ((sum h_i) * max(autoconv))"""
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    max_conv = _autoconv_max(h)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _score_fast(h):
    """Score with early-out for very bad configurations."""
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    s2 = np.dot(h, h)
    # Quick lower bound on max_conv: at least h_max^2
    h_max = np.max(h)
    if h_max < 1e-12:
        return 0.0
    max_conv = _autoconv_max(h)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _coordinate_descent(h, max_rounds=200, tol=1e-10):
    """
    Coordinate descent on nonnegative simplex.
    For each coordinate i, fix others and maximize f(x) = (s2 + 2*x*h_i_old + x^2) / ((s1 + x - h_i_old) * M(x))
    where M(x) is max autoconv with h_i = x.
    
    We use a simpler approach: for each coordinate, do a 1-D search.
    """
    h = h.copy()
    best_score = _score(h)
    
    for rnd in range(max_rounds):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        improved = False
        for i in range(N):
            if time.time() - _start_time > _TIME_LIMIT:
                break
            old_val = h[i]
            # Try a range of values for h[i]
            # Scale: try values from 0 to 2*mean
            mean_val = np.sum(h) / N
            # Candidate values: 0, old, and some perturbations
            candidates = [0.0, old_val]
            if old_val > 0:
                candidates += [old_val * 0.5, old_val * 0.75, old_val * 1.25, old_val * 1.5]
            candidates += [mean_val * 0.5, mean_val, mean_val * 1.5, mean_val * 2.0]
            # Filter to nonnegative
            candidates = [max(0.0, c) for c in candidates]
            # Also try small values
            candidates += [1e-4, 1e-3, 1e-2]
            
            best_cand = old_val
            best_cand_score = best_score
            
            for c in candidates:
                if abs(c - old_val) < 1e-12:
                    continue
                h[i] = c
                s = _score(h)
                if s > best_cand_score:
                    best_cand_score = s
                    best_cand = c
            
            h[i] = best_cand
            if best_cand != old_val:
                improved = True
                if best_cand_score > best_score:
                    best_score = best_cand_score
        
        if not improved:
            break
    
    return h, best_score

def _refine_1d(h, idx, n_trials=20):
    """Refine a single coordinate using golden-section-like search."""
    old_val = h[idx]
    # Determine search range
    s1 = np.sum(h)
    scale = max(s1 / N, 1e-3)
    lo = 0.0
    hi = 3.0 * scale
    
    # First, find a bracket
    best_val = old_val
    best_score = _score(h)
    
    # Try a grid
    grid = np.linspace(0, hi, n_trials)
    for g in grid:
        h[idx] = g
        s = _score(h)
        if s > best_score:
            best_score = s
            best_val = g
    
    h[idx] = best_val
    return best_val, best_score

def _multi_start_cd(init, rounds=100):
    """Run coordinate descent from an initialization."""
    h = np.array(init, dtype=float)
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-12:
        return h, 0.0
    
    best_h, best_score = _coordinate_descent(h, max_rounds=rounds)
    
    # Additional 1-D refinement pass
    for _ in range(3):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        for i in range(N):
            if time.time() - _start_time > _TIME_LIMIT:
                break
            val, sc = _refine_1d(best_h, i, n_trials=15)
            if sc > best_score:
                best_score = sc
                best_h[i] = val
    
    return best_h, best_score

def _random_init(seed):
    """Generate a random nonnegative vector."""
    rng = np.random.default_rng(seed)
    # Random Dirichlet-like
    alpha = rng.uniform(0.1, 2.0, N)
    h = rng.dirichlet(alpha) * rng.uniform(0.5, 2.0)
    return h

def _structured_inits():
    """Generate diverse structured initializations."""
    inits = []
    
    # Sparse: few nonzero
    for k in [2, 3, 4, 5, 7, 10]:
        h = np.zeros(N)
        # Place mass on first k positions with varying weights
        weights = np.ones(k) / k
        h[:k] = weights
        inits.append(h)
        
        # Uneven weights
        weights2 = np.linspace(0.1, 1.0, k)
        weights2 /= weights2.sum()
        h2 = np.zeros(N)
        h2[:k] = weights2
        inits.append(h2)
    
    # Spread: all nonzero
    h = np.ones(N) / N
    inits.append(h)
    
    # Gaussian-like
    idx = np.arange(N)
    center = N / 2
    h = np.exp(-0.5 * ((idx - center) / (N / 4)) ** 2)
    h /= h.sum()
    inits.append(h)
    
    # Two bumps
    h = np.zeros(N)
    h[5:10] = 0.2
    h[30:35] = 0.2
    h /= h.sum()
    inits.append(h)
    
    # Frequency-motivated: alternating
    h = np.zeros(N)
    h[::2] = 1.0 / (N // 2 + 1)
    inits.append(h)
    
    # Triangular
    h = np.abs(np.sin(np.pi * np.arange(N) / N))
    h /= h.sum()
    inits.append(h)
    
    # Sparse with gaps (Sidon-like)
    positions = [0, 3, 7, 12, 18, 25, 33, 42]
    h = np.zeros(N)
    for p in positions:
        h[p] = 1.0 / len(positions)
    inits.append(h)
    
    # More Sidon-like
    positions = [0, 4, 9, 15, 22, 30, 39, 49]
    h = np.zeros(N)
    for p in positions:
        h[p] = 1.0 / len(positions)
    inits.append(h)
    
    return inits

def construct():
    best_h = None
    best_score = 0.0
    
    # Structured inits
    inits = _structured_inits()
    
    # Add some random inits
    for seed in range(10):
        inits.append(_random_init(seed))
    
    # Run coordinate descent from each init
    for init in inits:
        if time.time() - _start_time > _TIME_LIMIT:
            break
        h_opt, s_opt = _multi_start_cd(init, rounds=80)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Additional random restarts around best
    if best_h is not None:
        rng = np.random.default_rng(123)
        for _ in range(30):
            if time.time() - _start_time > _TIME_LIMIT:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = np.clip(h_pert, 0, None)
            h_opt, s_opt = _multi_start_cd(h_pert, rounds=50)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
    
    # Final polish: more coordinate descent rounds
    if best_h is not None and time.time() - _start_time < _TIME_LIMIT - 5:
        h_opt, s_opt = _coordinate_descent(best_h, max_rounds=100)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    if best_h is None:
        # Fallback
        best_h = np.zeros(N)
        best_h[0] = 2.0 / 3.0
        best_h[1] = 1.0 / 3.0
    
    return [float(x) for x in best_h]
