"""Low discrepancy on all arithmetic progressions - Local search with vectorized evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _max_ap_discrepancy_v2(signs, n=256):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: array of +1/-1 indexed 0..n-1.
    Returns the max absolute sum (int).
    """
    s = np.asarray(signs, dtype=np.int16)
    max_abs = 0
    for d in range(1, n + 1):
        for i in range(d):
            if i >= n:
                break
            idx = np.arange(i, n, d)
            if len(idx) < 2:
                continue
            ap = s[idx]
            cs = np.cumsum(ap)
            cmin = int(cs.min())
            cmax = int(cs.max())
            if cmin > 0:
                cmin = 0
            if cmax < 0:
                cmax = 0
            val = cmax - cmin
            if val > max_abs:
                max_abs = val
    return max_abs

def _compute_influence(n):
    """
    For each position p in [0, n), count how many APs (with length >= 2) pass through p.
    """
    influence = np.zeros(n, dtype=np.int32)
    for d in range(1, n + 1):
        for i in range(d):
            if i >= n:
                break
            L = (n - 1 - i) // d + 1
            if L < 2:
                continue
            idx = np.arange(i, n, d)
            influence[idx] += 1
    return influence

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin out of 90s
    
    # Precompute influence
    influence = _compute_influence(n)
    
    # Initial random assignment
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    best_signs = signs.copy()
    best_disc = _max_ap_discrepancy_v2(signs, n)
    
    # Local search: steepest descent with random kicks
    max_iterations = 50000
    kick_every = 30  # every 30 iterations, do a random kick if stuck
    
    stuck_count = 0
    current_disc = best_disc
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try flipping positions in order of influence
        # Use a rotating subset for diversity
        if it % 200 == 0:
            # Full search over all positions sorted by influence
            candidate_indices = np.argsort(-influence)
        else:
            # Random subset of top positions
            top_k = min(32, n)
            top_indices = np.argsort(-influence)[:top_k * 3]
            candidate_indices = rng.choice(top_indices, size=min(top_k, len(top_indices)), replace=False)
        
        improved = False
        for p in candidate_indices:
            if time.time() > deadline:
                break
            # Try flipping position p
            signs[p] = -signs[p]
            new_disc = _max_ap_discrepancy_v2(signs, n)
            if new_disc < current_disc:
                current_disc = new_disc
                improved = True
                break
            else:
                signs[p] = -signs[p]  # revert
        
        if improved:
            stuck_count = 0
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            stuck_count += 1
            if stuck_count >= kick_every:
                # Random kick: flip 3-5 random positions
                k = rng.randint(3, 6)
                kick_pos = rng.choice(n, size=k, replace=False)
                signs[kick_pos] = -signs[kick_pos]
                new_disc = _max_ap_discrepancy_v2(signs, n)
                if new_disc < current_disc:
                    current_disc = new_disc
                    stuck_count = 0
                else:
                    # Revert
                    signs[kick_pos] = -signs[kick_pos]
                    stuck_count = 0
    
    # Final fine-tuning: try flipping each position once
    signs = best_signs.copy()
    current_disc = best_disc
    for p in range(n):
        if time.time() > deadline:
            break
        signs[p] = -signs[p]
        new_disc = _max_ap_discrepancy_v2(signs, n)
        if new_disc < current_disc:
            current_disc = new_disc
        else:
            signs[p] = -signs[p]
    
    if current_disc < best_disc:
        best_disc = current_disc
        best_signs = signs.copy()
    
    return list(best_signs)
# EVOLVE-BLOCK-END
