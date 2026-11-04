"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_value(h):
    """Compute the actual evaluation value."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))


def _local_search(h, max_iters=500, step_size=0.5, rng=None):
    """Perform local search by randomly perturbing one element at a time."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    best_h = h.copy()
    best_val = _compute_value(best_h)
    
    for i in range(max_iters):
        # Pick a random index to perturb
        idx = rng.integers(0, N)
        # Try a few random perturbations
        for _ in range(5):
            delta = rng.normal(0, step_size)
            trial = best_h.copy()
            trial[idx] += delta
            val = _compute_value(trial)
            if val < best_val:
                best_val = val
                best_h = trial.copy()
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    rng = np.random.default_rng(123)
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Try several initial strategies
    strategies = []
    
    # Strategy 1: All ones (seed)
    strategies.append(np.ones(N))
    
    # Strategy 2: Alternating signs
    alt = np.ones(N)
    alt[1::2] = -1
    strategies.append(alt)
    
    # Strategy 3: Random Gaussian
    strategies.append(rng.normal(0, 1, N))
    
    # Strategy 4: Low-frequency cosine
    t = np.linspace(0, 4*np.pi, N)
    strategies.append(np.cos(t))
    
    # Strategy 5: Sine
    strategies.append(np.sin(t))
    
    # Strategy 6: Mixed
    strategies.append(np.cos(t) + 0.5*np.sin(2*t))
    
    # Strategy 7: Block pattern
    block = np.ones(N//4)
    block[1::2] = -1
    strategies.append(np.tile(block, 4))
    
    # Strategy 8: Triangular wave
    tri = 2 * (np.abs(2 * (np.arange(N) / N - 0.5)) - 0.5)
    strategies.append(tri)
    
    # Strategy 9: Random with low-frequency bias
    low_freq = np.cos(2*np.pi*np.random.default_rng(99).integers(1,5) * np.arange(N)/N)
    high_freq = rng.normal(0, 0.3, N)
    strategies.append(low_freq + high_freq)
    
    # Strategy 10: Another random
    strategies.append(rng.normal(0, 1, N))
    
    # Evaluate all initial strategies
    for h in strategies:
        if time.time() > deadline:
            break
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Perform local search from the best initial point
    if time.time() < deadline:
        best_h, best_val = _local_search(best_h, max_iters=200, step_size=0.3, rng=rng)
    
    # Try more local searches from different starting points if time permits
    for h in strategies[:5]:
        if time.time() > deadline:
            break
        candidate, val = _local_search(h, max_iters=100, step_size=0.2, rng=rng)
        if val < best_val:
            best_val = val
            best_h = candidate
    
    # Final refinement
    if time.time() < deadline - 5:
        best_h, best_val = _local_search(best_h, max_iters=150, step_size=0.1, rng=rng)
    
    # Ensure finite and non-zero sum
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
