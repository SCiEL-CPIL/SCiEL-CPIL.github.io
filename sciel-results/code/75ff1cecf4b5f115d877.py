"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction
import heapq

DENOMS = list(range(2, 49))
N = len(DENOMS)

L = 1
for d in DENOMS:
    L = lcm(L, d)

WEIGHTS = [L // d for d in DENOMS]
# Sort indices by weight descending
WEIGHT_IDX = sorted([(w, i) for i, w in enumerate(WEIGHTS)], key=lambda x: -x[0])
SORTED_WEIGHTS = [w for w, i in WEIGHT_IDX]
SORTED_ORIG_IDX = [i for w, i in WEIGHT_IDX]

# Precompute suffix sums
suffix_sum = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    suffix_sum[i] = suffix_sum[i + 1] + SORTED_WEIGHTS[i]

def _greedy_init():
    """Greedy: take largest weights that fit under L."""
    s = 0
    idxs = []
    for i in range(N):
        if s + SORTED_WEIGHTS[i] < L:
            s += SORTED_WEIGHTS[i]
            idxs.append(SORTED_ORIG_IDX[i])
    return s, idxs

def _local_swap_improve(sum_w, indices):
    """Try to improve by swapping or adding elements."""
    in_set = set(indices)
    improved = True
    while improved and time.time() < _deadline:
        improved = False
        # Try adding one element
        for i in range(N):
            if SORTED_ORIG_IDX[i] not in in_set:
                new_s = sum_w + SORTED_WEIGHTS[i]
                if new_s < L and new_s > sum_w:
                    sum_w = new_s
                    indices.append(SORTED_ORIG_IDX[i])
                    in_set.add(SORTED_ORIG_IDX[i])
                    improved = True
                    break
        # Try swapping: remove one, add one or two
        for j in list(indices):
            sj = 0
            for k in indices:
                if k == j:
                    sj = SORTED_WEIGHTS[SORTED_ORIG_IDX.index(k)]
                    break
            # Find weight of j
            wj = WEIGHTS[j]
            for i in range(N):
                oi = SORTED_ORIG_IDX[i]
                if oi == j:
                    continue
                if oi in in_set:
                    continue
                wi = SORTED_WEIGHTS[i]
                new_s = sum_w - wj + wi
                if new_s < L and new_s > sum_w:
                    indices.remove(j)
                    in_set.remove(j)
                    indices.append(oi)
                    in_set.add(oi)
                    sum_w = new_s
                    improved = True
                    break
            if improved:
                break
    return sum_w, indices

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 55.0
    
    # Initial greedy
    best_sum_w, best_indices = _greedy_init()
    
    # Local swap improvement
    best_sum_w, best_indices = _local_swap_improve(best_sum_w, best_indices)
    
    # Branch and Bound with memoization
    # Use a priority queue: (negative_upper_bound, i, current_sum, indices)
    # Upper bound = current_sum + suffix_sum[i]
    
    visited = set()  # (i, current_sum) to avoid revisiting same state
    
    # Start: we can include or exclude item 0
    # Use iterative deepening or A* style
    
    def search(i, current_sum, current_indices):
        nonlocal best_sum_w, best_indices
        if time.time() > _deadline:
            return
        
        # Pruning: if we can't beat the best even with all remaining
        if current_sum + suffix_sum[i] <= best_sum_w:
            return
        
        # Memoization key
        state_key = (i, current_sum)
        if state_key in visited:
            return
        visited.add(state_key)
        
        # Base case
        if i == N:
            if current_sum < L and current_sum > best_sum_w:
                best_sum_w = current_sum
                best_indices = list(current_indices)
            return
        
        w_i = SORTED_WEIGHTS[i]
        orig_i = SORTED_ORIG_IDX[i]
        
        # Option 1: Include item i (try first since we want to maximize)
        new_sum = current_sum + w_i
        if new_sum < L:
            current_indices.append(orig_i)
            search(i + 1, new_sum, current_indices)
            current_indices.pop()
            
        # Option 2: Exclude item i
        if current_sum + suffix_sum[i + 1] > best_sum_w:
            search(i + 1, current_sum, current_indices)
    
    # Run Branch and Bound
    search(0, 0, [])
    
    # If best_indices is empty (shouldn't happen as 1/2 < 1), fallback
    if not best_indices:
        best_indices = [0]
        
    return best_indices
# EVOLVE-BLOCK-END
