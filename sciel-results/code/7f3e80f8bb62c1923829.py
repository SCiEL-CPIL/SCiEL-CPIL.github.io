"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    n = len(h)
    max_ov = 0.0
    for k in range(n):
        shifted = np.roll(h, -k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _local_search(h, max_iter=50000):
    """Local search: try flipping individual elements or small perturbations."""
    n = len(h)
    current = h.copy()
    current_score = _compute_max_overlap(current)
    
    for it in range(max_iter):
        # Pick a random index to perturb
        i = np.random.randint(n)
        # Try a small perturbation
        delta = np.random.uniform(-0.1, 0.1)
        new_val = np.clip(current[i] + delta, 0.0, 1.0)
        
        old_val = current[i]
        current[i] = new_val
        new_score = _compute_max_overlap(current)
        
        if new_score < current_score:
            current_score = new_score
        else:
            current[i] = old_val
    
    return current, current_score


def _construct_binary_pattern():
    """Create an initial binary pattern with sum = N/2."""
    h = np.zeros(N)
    # Use a pattern: first N/2 are 1, rest are 0, then shuffle
    h[:N//2] = 1.0
    np.random.shuffle(h)
    # Ensure exact sum
    # Count 1s
    n_ones = np.sum(h == 1.0)
    target = N // 2
    if n_ones > target:
        # Flip some 1s to 0s
        ones_idx = np.where(h == 1.0)[0]
        np.random.shuffle(ones_idx)
        for i in ones_idx[:n_ones - target]:
            h[i] = 0.0
    elif n_ones < target:
        zeros_idx = np.where(h == 0.0)[0]
        np.random.shuffle(zeros_idx)
        for i in zeros_idx[:target - n_ones]:
            h[i] = 1.0
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try multiple random binary initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        h0 = _construct_binary_pattern()
        h, score = _local_search(h0, max_iter=20000)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final refinement on the best solution
    if time.time() < deadline:
        h, score = _local_search(best_h, max_iter=30000)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Ensure sum is exactly N/2
    current_sum = np.sum(best_h)
    diff = N / 2.0 - current_sum
    if abs(diff) > 1e-10:
        # Distribute the correction
        best_h += diff / N
        # Clip and re-normalize if needed
        best_h = np.clip(best_h, 0.0, 1.0)
        # Fix sum again
        current_sum = np.sum(best_h)
        if abs(current_sum - N/2.0) > 1e-10:
            # Simple correction
            best_h[N-1] += (N/2.0 - current_sum)
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
