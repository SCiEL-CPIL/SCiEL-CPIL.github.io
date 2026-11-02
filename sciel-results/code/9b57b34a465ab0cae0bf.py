"""Optimized construction for the signed autocorrelation C3-prime task.
Uses epigraph formulation with penalty + coordinate descent polish + multiple beta soft-max sweeps.
"""
import time
import numpy as np

N = 400


def _autoconv(x):
    """Full linear autoconvolution of x via FFT (zero-padded)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(x, size)
    c = np.fft.irfft(F * np.conj(F), size)
    return c[:2 * n - 1]


def _score(x):
    """C3' = max|autoconv| / (integral)^2. Integral = sum(x)."""
    s = x.sum()
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv(x)
    return float(np.max(np.abs(c)) / (s * s))


def _epigraph_objective(x, beta, penalty=100.0):
    """Epigraph-style objective: soft-max of |autoconv| + penalty * (max(0, t - |sum|))^2.
    We minimize over x and implicitly t is the soft-max value."""
    s = x.sum()
    c = _autoconv(x)
    ac_abs = np.abs(c)
    m = np.max(ac_abs)
    # Soft max approximation
    soft_max = m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (ac_abs - m))))
    # Penalize small |sum| to keep denominator large
    sum_penalty = penalty * max(0.0, 1.0 - abs(s)) ** 2
    return soft_max + sum_penalty


def _epigraph_coordinate_descent(x, deadline, beta, iters=50000):
    """Coordinate descent on the epigraph objective for a fixed beta."""
    x = x.copy().astype(np.float64)
    n = len(x)
    rng = np.random.default_rng(42)
    best_obj = _epigraph_objective(x, beta)
    
    # Precompute autoconv for current x
    c = _autoconv(x)
    s = x.sum()
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # Pick a random coordinate
        j = int(rng.integers(n))
        
        # Try flipping sign at j
        x[j] = -x[j]
        
        # Recompute autoconv efficiently using the fact that flipping one element
        # changes the autoconv in a structured way
        # c_new[l] = c_old[l] - 2*x_old[j]*sum_{k} x[k+l-j] ... this is complex
        # Just recompute for simplicity since N=400 is small
        obj = _epigraph_objective(x, beta)
        
        if obj < best_obj:
            best_obj = obj
            c = _autoconv(x)
            s = x.sum()
        else:
            x[j] = -x[j]  # revert
    
    return x, best_obj


def _soft_max_sweep(x, deadline, betas=None):
    """Sweep over different beta values for the soft-max approximation."""
    if betas is None:
        betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    best_x = x.copy()
    best_score = _score(x)
    
    for beta in betas:
        if time.time() > deadline:
            break
        x_cur, _ = _epigraph_coordinate_descent(x, deadline, beta, iters=20000)
        sc = _score(x_cur)
        if sc < best_score:
            best_score = sc
            best_x = x_cur.copy()
    
    return best_x, best_score


def _local_search_signs(x, deadline, max_steps=50000):
    """Sign-flip local search minimizing the actual score."""
    x = x.copy().astype(np.float64)
    best = _score(x)
    n = len(x)
    rng = np.random.default_rng(12345)
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        
        j = int(rng.integers(n))
        x[j] = -x[j]
        sc = _score(x)
        if sc < best:
            best = sc
        else:
            x[j] = -x[j]  # revert
    
    return x, best


def _structured_seeds(rng):
    """Generate diverse structured seeds."""
    seeds = []
    
    # All ones (baseline)
    seeds.append(np.ones(N, dtype=np.float64))
    
    # Alternating
    seeds.append(np.tile([1.0, -1.0], N // 2).astype(np.float64))
    
    # Random ±1
    for _ in range(10):
        x = rng.choice([-1.0, 1.0], size=N).astype(np.float64)
        seeds.append(x)
    
    # Block patterns with varying block sizes
    for block_size in [2, 4, 8, 16, 32, 64]:
        if N % block_size == 0:
            n_blocks = N // block_size
            block_signs = rng.choice([-1.0, 1.0], size=n_blocks)
            x = np.repeat(block_signs, block_size).astype(np.float64)
            seeds.append(x)
    
    # Barker-like: try to minimize autocorrelation side lobes
    # Use a known low-autocorrelation pattern as inspiration
    for _ in range(5):
        # Start with random, then do a few greedy improvements
        x = rng.choice([-1.0, 1.0], size=N).astype(np.float64)
        # Greedy: flip the element that most reduces max|autoconv|
        for _ in range(100):
            c = _autoconv(x)
            ac_abs = np.abs(c)
            # Find which single flip would most reduce the peak
            best_j = 0
            best_improvement = 0
            for j in range(0, N, max(1, N // 50)):  # Sample some positions
                x[j] = -x[j]
                c2 = _autoconv(x)
                ac_abs2 = np.abs(c2)
                improvement = np.max(ac_abs) - np.max(ac_abs2)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_j = j
                x[j] = -x[j]
            x[best_j] = -x[best_j]
        seeds.append(x)
    
    return seeds


def construct():
    """Return the sequence of signed step heights minimizing C3'."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    rng = np.random.default_rng(2024)
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)
    
    seeds = _structured_seeds(rng)
    
    for s in seeds:
        if time.time() > deadline - 10:
            break
        
        s = s.astype(np.float64)
        
        # Phase 1: Epigraph coordinate descent with beta sweep
        x_ep, sc_ep = _soft_max_sweep(s, deadline - 10)
        if sc_ep < best_score:
            best_score = sc_ep
            best_x = x_ep.copy()
        
        # Phase 2: Sign-flip local search polish
        x_ls, sc_ls = _local_search_signs(x_ep, deadline - 5, max_steps=30000)
        if sc_ls < best_score:
            best_score = sc_ls
            best_x = x_ls.copy()
    
    # Final refinement on the best found
    if time.time() < deadline - 5:
        x_final, sc_final = _local_search_signs(best_x, deadline - 2, max_steps=100000)
        if sc_final < best_score:
            best_x = x_final
            best_score = sc_final
    
    return best_x
