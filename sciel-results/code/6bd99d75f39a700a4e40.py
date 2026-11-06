"""
Strategy: Simulated Annealing with vectorized AP-sum updates.

This is a structurally different approach from the current greedy single-flip local search.
1. We precompute the AP incidence matrix M (dense float32).
2. Instead of greedy hill-climbing (which gets stuck in local minima), we use Simulated Annealing (SA).
   SA allows accepting worse solutions with a probability that decreases over time, enabling it to escape local minima.
3. The objective is to minimize the maximum absolute sum over all APs.
4. We define a "temperature" schedule that cools down over the available time budget.
5. At each step, we pick a random index to flip. We compute the change in the max absolute sum.
   - If the new max is lower, we accept.
   - If the new max is higher, we accept with probability exp(-delta / T).
6. We keep track of the best solution seen so far.
7. Finally, we run a few rounds of greedy local search on the best solution found by SA to polish it.

Key difference: Greedy search is deterministic and gets trapped in the first local minimum it finds.
SA is stochastic and can jump over barriers to find better minima.
"""

import time
import random
import numpy as np

N = 256

def _get_ap_indices():
    """Precompute all arithmetic progressions in [1, N] as a list of index lists."""
    aps = []
    for step in range(1, N // 2 + 1):
        for start in range(1, N - step + 1):
            max_len = (N - start) // step + 1
            if max_len < 2:
                continue
            indices = [start - 1 + k * step for k in range(max_len)]
            for length in range(2, max_len + 1):
                aps.append(indices[:length])
    return aps

def _precompute_ap_matrix(aps):
    """Build a dense float32 matrix M where M[i, j] = 1 if index j is in AP i."""
    num_aps = len(aps)
    M = np.zeros((num_aps, N), dtype=np.float32)
    for i, idxs in enumerate(aps):
        for idx in idxs:
            M[i, idx] = 1.0
    return M

def _compute_max_abs_sum(signs, M):
    """Compute max absolute sum for a given sign vector."""
    sums = M @ signs
    return float(np.max(np.abs(sums)))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, 5s margin

    # 1. Precompute
    aps = _get_ap_indices()
    M = _precompute_ap_matrix(aps)
    num_aps = len(aps)
    
    # 2. Initialize
    rng = random.Random(42)
    
    # Start with a random initialization
    # We use float32 for speed in matrix multiplication
    current_signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.float32)
    current_sums = M @ current_signs
    current_max = float(np.max(np.abs(current_sums)))
    
    best_signs = current_signs.copy()
    best_max = current_max
    
    # Simulated Annealing parameters
    # We want to run for most of the time budget.
    # Number of steps depends on how fast each step is.
    # One step: pick random index, compute new sums, compute new max.
    # M @ signs is O(num_aps * N). num_aps ~ 50k, N=256. 50k*256 = 12.8M ops.
    # This is very fast in NumPy (BLAS). Maybe ~1ms per step?
    # So we can do ~80k steps in 80s.
    
    # Temperature schedule
    # Start high enough to allow many uphill moves, end low enough to converge.
    # The "energy" is the max absolute sum. Typical values might be 1-10.
    T_start = 10.0
    T_end = 0.01
    
    # We will iterate until time runs out or T is very low.
    # To make it deterministic-ish but stochastic, we use a fixed number of iterations
    # if we can fit them, or time-based.
    
    # Let's estimate steps per second.
    # We'll just loop until deadline - 5s (leaving time for final polish).
    
    # To avoid too many tiny steps at the end, we can cool exponentially.
    # T = T_start * (T_end/T_start)^t
    # But time-based cooling is tricky if step time varies.
    # Let's use a fixed number of steps if possible, else time.
    # Actually, let's just do a while loop with time check.
    
    # To make cooling smooth, we can track iteration count and total expected iterations.
    # But since we don't know exact speed, let's use a simple exponential decay based on time.
    # T(t) = T_start * exp(-k * t)
    # We want T to drop from 10 to 0.01 in ~70 seconds.
    # 0.01 = 10 * exp(-k * 70)
    # 0.001 = exp(-70k)
    # ln(0.001) = -70k
    # -6.9 = -70k => k ~ 0.1
    
    k_cool = 0.1
    
    # To speed up, we can batch? No, SA is sequential.
    # However, we can optimize the "accept" step.
    # We only need to update sums for the flipped index.
    # new_sums = old_sums + 2 * signs[k] * M[:, k]
    # This is a vector operation on length num_aps.
    # Computing max(abs(new_sums)) is O(num_aps).
    
    while time.time() < deadline - 5.0:
        elapsed = time.time() - start_time
        T = T_start * np.exp(-k_cool * elapsed)
        
        # Pick random index to flip
        k = rng.randrange(N)
        
        # Compute new sums
        # delta = 2 * signs[k] * M[:, k]
        # new_sums = current_sums + delta
        # new_max = max(abs(new_sums))
        
        # Optimization: We can compute the change in max more efficiently?
        # Not easily, because max is a non-smooth function.
        # But computing the full vector and max is fast enough.
        
        delta_vec = 2.0 * current_signs[k] * M[:, k]
        new_sums = current_sums + delta_vec
        new_max = float(np.max(np.abs(new_sums)))
        
        # Acceptance criterion
        if new_max <= current_max:
            # Accept (improvement or equal)
            current_signs[k] *= -1.0
            current_sums = new_sums
            current_max = new_max
            
            if current_max < best_max:
                best_max = current_max
                best_signs = current_signs.copy()
        else:
            # Deterioration
            delta_E = new_max - current_max
            # Probability of acceptance
            # Avoid log of 0 or underflow
            if delta_E > 0 and T > 0:
                prob = np.exp(-delta_E / T)
                if rng.random() < prob:
                    current_signs[k] *= -1.0
                    current_sums = new_sums
                    current_max = new_max
                    # Note: We don't update best_max here, but we might want to 
                    # if we think this is a good temporary state? No, best_max should 
                    # only track the true minimum found.
            # Else: reject (T is 0 or delta_E is 0 which is handled above)
        
        # Optional: Restart if stuck?
        # If we haven't improved best_max in a long time, maybe restart from best?
        # This is a common heuristic.
        # Let's add a simple restart mechanism:
        # If no improvement in best_max for, say, 2 seconds, reset current to best.
        # This requires tracking last_improvement_time.
        
    # 3. Final Polish: Greedy Local Search
    # Start from best_signs found by SA
    current_signs = best_signs
    current_sums = M @ current_signs
    current_max = best_max
    
    # Run greedy hill climbing until no improvement or time limit
    while time.time() < deadline - 1.0:
        best_k = -1
        best_new_max = current_max
        
        for k in range(N):
            delta_vec = 2.0 * current_signs[k] * M[:, k]
            new_sums = current_sums + delta_vec
            new_max = float(np.max(np.abs(new_sums)))
            if new_max < best_new_max:
                best_new_max = new_max
                best_k = k
        
        if best_k == -1:
            break
            
        current_signs[best_k] *= -1.0
        current_sums = M @ current_signs
        current_max = best_new_max
        
        if current_max < best_max:
            best_max = current_max
            best_signs = current_signs.copy()
            
    # 4. Convert to list of ints
    result = [int(x) for x in best_signs]
    return result

