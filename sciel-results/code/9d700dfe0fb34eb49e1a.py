"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _objective(h):
    """Compute the evaluation value for a given height array."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _random_zero_sum_signs(n, rng):
    """Generate a random ±1 sequence with exactly zero sum (even n)."""
    h = rng.choice([-1.0, 1.0], size=n)
    # Ensure zero sum: swap pairs to balance
    pos_count = np.sum(h == 1.0)
    neg_count = n - pos_count
    if pos_count != neg_count:
        # Need to flip some to balance
        target_pos = n // 2
        if pos_count > target_pos:
            # Flip (pos_count - target_pos) +1s to -1
            pos_indices = np.where(h == 1.0)[0]
            to_flip = rng.choice(pos_indices, size=pos_count - target_pos, replace=False)
            h[to_flip] = -1.0
        else:
            neg_indices = np.where(h == -1.0)[0]
            to_flip = rng.choice(neg_indices, size=target_pos - pos_count, replace=False)
            h[to_flip] = 1.0
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    best_h = None
    best_val = float('inf')
    
    # Initial population: diverse random zero-sum sign sequences
    n_candidates = 200
    candidates = []
    for _ in range(n_candidates):
        h = _random_zero_sum_signs(N, rng)
        val = _objective(h)
        candidates.append((val, h))
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Sort by value
    candidates.sort(key=lambda x: x[0])
    
    # Local search: hill climbing with random restarts
    max_iterations = 200000
    it = 0
    
    while it < max_iterations and time.time() < deadline:
        # Pick a candidate from top portion (biased toward best)
        idx = int(rng.power(2.0) * min(50, len(candidates)))
        if idx >= len(candidates):
            idx = len(candidates) - 1
        val, h = candidates[idx]
        
        # Try random single-flip moves
        n_tries = 10
        improved = False
        for _ in range(n_tries):
            if time.time() > deadline:
                break
            i = rng.integers(0, N)
            h_new = h.copy()
            h_new[i] = -h_new[i]
            
            # Check sum isn't zero
            s = np.sum(h_new)
            if abs(s) < 1e-12:
                continue
            
            val_new = _objective(h_new)
            
            if val_new < val:
                # Accept improvement
                candidates[idx] = (val_new, h_new)
                h = h_new
                val = val_new
                improved = True
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                # Re-sort occasionally
                if it % 1000 == 0:
                    candidates.sort(key=lambda x: x[0])
            elif val_new < float('inf') and rng.random() < 0.1:
                # Occasional random walk
                candidates[idx] = (val_new, h_new)
                h = h_new
                val = val_new
        
        it += 1
    
    # If we still have time, do more focused local search on best
    if time.time() < deadline - 5.0 and best_h is not None:
        h = best_h.copy()
        val = best_val
        for _ in range(100000):
            if time.time() > deadline - 2.0:
                break
            i = rng.integers(0, N)
            h_new = h.copy()
            h_new[i] = -h_new[i]
            s = np.sum(h_new)
            if abs(s) < 1e-12:
                continue
            val_new = _objective(h_new)
            if val_new < val:
                h = h_new
                val = val_new
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
