"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(h):
    """Compute max |autoconv| / (sum(h))^2. If sum(h) is 0, return inf."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    autoconv = np.correlate(h, h, mode='full')
    max_abs = np.max(np.abs(autoconv))
    return max_abs / (s * s)


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin
    
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    # Try simple constructions
    # 1. All ones (baseline)
    # 2. Try adding small perturbations
    # 3. Local search
    
    rng = np.random.default_rng(42)
    
    # Try a few random sign flips to see if we can improve
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        h = best_h.copy()
        # Flip a random subset
        n_flips = rng.integers(1, N // 4)
        indices = rng.choice(N, size=n_flips, replace=False)
        h[indices] *= -1
        
        score = _compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h
    
    return best_h
# EVOLVE-BLOCK-END
