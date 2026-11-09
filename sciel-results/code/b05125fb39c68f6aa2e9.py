"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)
# Precompute weights as integers for exact comparison
# We want sum(1/d_i) < 1.
# Multiply by L = lcm(2..48).
# Let W_i = L / d_i.
# We want to maximize sum(W_i) subject to sum(W_i) < L.
# Deficit in original space = 1 - sum(1/d_i) = (L - sum(W_i)) / L.
# Minimizing deficit is equivalent to maximizing sum(W_i) subject to sum(W_i) < L.

L = 1
for d in DENOMS:
    L = lcm(L, d)

WEIGHTS = [L // d for d in DENOMS]
# Sort indices by weight descending to help pruning
# But we need to return original indices.
# Let's keep track of (weight, original_index)
WEIGHT_IDX = sorted([(w, i) for i, w in enumerate(WEIGHTS)], key=lambda x: -x[0])
SORTED_WEIGHTS = [w for w, i in WEIGHT_IDX]
SORTED_ORIG_IDX = [i for w, i in WEIGHT_IDX]

# Precompute suffix sums of weights for pruning
# suffix_sum[i] = sum of SORTED_WEIGHTS[i:]
suffix_sum = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    suffix_sum[i] = suffix_sum[i + 1] + SORTED_WEIGHTS[i]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_sum_w = 0  # Max sum of weights found so far (< L)
    best_indices = []
    
    # If we can't find any, we return empty? No, must be non-empty.
    # But 1/2 < 1, so index 0 is always valid.
    
    # Branch and Bound
    # State: index i, current_sum
    # We decide to include or exclude item i.
    # Pruning:
    # 1. If current_sum + suffix_sum[i] <= best_sum_w: 
    #    Even if we take ALL remaining, we can't beat best_sum_w. Prune.
    #    (Note: we need strictly less than L. If current_sum + suffix_sum[i] >= L, 
    #     we still might find a valid subset, but we can't take all. 
    #     Actually, if current_sum + suffix_sum[i] <= best_sum_w, we can't improve.
    #     If current_sum >= L, invalid.
    #     If current_sum < L, it's a valid candidate.
    #     We want to maximize current_sum.
    
    def search(i, current_sum, current_indices):
        nonlocal best_sum_w, best_indices
        if time.time() > deadline:
            return
            
        # Pruning: if we can't beat the best even with all remaining
        if current_sum + suffix_sum[i] <= best_sum_w:
            return
            
        # Base case
        if i == N:
            if current_sum < L and current_sum > best_sum_w:
                best_sum_w = current_sum
                best_indices = list(current_indices)
            return
        
        # Option 1: Include item i
        new_sum = current_sum + SORTED_WEIGHTS[i]
        if new_sum < L:
            current_indices.append(SORTED_ORIG_IDX[i])
            search(i + 1, new_sum, current_indices)
            current_indices.pop()
            
        # Option 2: Exclude item i
        # Only worth exploring if we can still potentially beat best_sum_w
        # If current_sum + suffix_sum[i+1] > best_sum_w
        if current_sum + suffix_sum[i + 1] > best_sum_w:
            search(i + 1, current_sum, current_indices)

    # Initial greedy to get a good lower bound for best_sum_w
    # Greedy: take largest weights that fit
    greedy_sum = 0
    greedy_indices = []
    for i in range(N):
        if greedy_sum + SORTED_WEIGHTS[i] < L:
            greedy_sum += SORTED_WEIGHTS[i]
            greedy_indices.append(SORTED_ORIG_IDX[i])
            
    best_sum_w = greedy_sum
    best_indices = list(greedy_indices)
    
    # Run Branch and Bound
    search(0, 0, [])
    
    # If best_indices is empty (shouldn't happen as 1/2 < 1), fallback
    if not best_indices:
        best_indices = [0]
        
    return best_indices
# EVOLVE-BLOCK-END
