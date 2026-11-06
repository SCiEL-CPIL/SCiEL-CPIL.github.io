"""Optimized solver for large reciprocal weight under LCM separation."""
# EVOLVE-BLOCK-START
import math
from math import gcd
import sys

# Increase recursion limit for deep branch and bound
sys.setrecursionlimit(10000)

def construct():
    """
    Find a nonempty subset of [2, 256] such that lcm(a,b) > 256 for all distinct pairs,
    maximizing the sum of reciprocals.
    Uses Branch and Bound with Greedy Upper Bound Pruning.
    """
    MAX_N = 256
    N = MAX_N - 1  # Number of candidates: 2..256
    values = [i + 2 for i in range(N)]
    weights = [1.0 / v for v in values]
    
    # Precompute incompatibility
    # incompat[i] is a set of indices j > i such that lcm(values[i], values[j]) <= 256
    # We store full sets for ease of checking, but only fill j > i initially, then symmetrize or use masks.
    # For B&B, we process indices in order. We need to know if a candidate node `j` (j > i) 
    # is compatible with all selected nodes.
    
    # Let's use bitmasks for speed? N=255, fits in 4 integers or a Python int (arbitrary precision).
    # Python ints are fast for bit operations.
    # mask[i] has bit j set if i and j are INCOMPATIBLE.
    masks = [0] * N
    for i in range(N):
        vi = values[i]
        for j in range(i + 1, N):
            vj = values[j]
            g = gcd(vi, vj)
            if (vi // g) * vj <= 256:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
                
    # Sort indices by weight descending (value ascending) to find good solutions early
    # Index 0 is value 2 (weight 0.5), Index 254 is value 256 (weight ~0.004)
    # We want to try adding high weight nodes first.
    # So we iterate i from 0 to N-1. This is already sorted by weight descending.
    
    best_score = 0.0
    best_selection = []
    
    # Upper Bound Helper: Greedy Max Weight Independent Set on remaining compatible nodes
    def greedy_upper_bound(start_idx, forbidden_mask):
        """
        Estimate the max additional score from nodes [start_idx, N) 
        that are not in forbidden_mask and compatible with each other.
        forbidden_mask contains bits for nodes that conflict with current selection.
        """
        # We need to consider nodes j >= start_idx.
        # A node j is available if (forbidden_mask >> j) & 1 == 0.
        # Greedy: pick the available node with highest weight, add its weight, 
        # and add all its conflicts to forbidden_mask.
        
        # To speed up, we can just iterate from start_idx to N.
        # Since weights are decreasing (value increasing), the first available node 
        # is the one with the highest weight in the remaining set.
        
        total = 0.0
        current_forbidden = forbidden_mask
        
        # Optimization: If current_forbidden has bits set for all remaining, return 0.
        # We iterate j from start_idx to N-1
        for j in range(start_idx, N):
            if (current_forbidden >> j) & 1:
                continue
            # Node j is available
            total += weights[j]
            # Add conflicts of j to forbidden mask
            # Only care about bits >= start_idx? No, future steps only look at >= j+1 effectively 
            # but the mask is global. 
            current_forbidden |= masks[j]
            
        return total

    def branch_and_bound(idx, current_score, selected_mask, selected_list):
        nonlocal best_score, best_selection
        
        # Pruning: Upper Bound
        # The remaining nodes are idx to N-1.
        # The forbidden mask for these nodes is determined by the current selection.
        # Wait, `selected_mask` contains bits for selected nodes.
        # A node `k` is incompatible with selection if `masks[k] & selected_mask != 0`.
        # This check is O(1) per node.
        
        # Calculate bound
        # We need a mask of all nodes in [idx, N) that are incompatible with current selection.
        # Let's construct `forbidden_remaining`.
        # Actually, the greedy_upper_bound function takes a `forbidden_mask`.
        # The `forbidden_mask` passed to it should contain bits for nodes that are 
        # INCOMPATIBLE with the current selection OR already selected (though selected 
        # are < idx usually, or we handle that).
        
        # Let's refine the bound calculation.
        # Nodes < idx are already decided.
        # Nodes >= idx are candidates.
        # A candidate k >= idx is valid if it is compatible with all selected nodes.
        # selected nodes are in `selected_mask`.
        # So `forbidden_mask_for_bound` should have bit k set if `masks[k] & selected_mask != 0`.
        
        # Computing this mask for every node k in [idx, N) is O(N) per step.
        # Total O(N^2) per path. Depth N. Total O(N^3). 255^3 ~ 16M. Might be ok in Python if optimized?
        # Python is slow. 16M operations might take 1-2 seconds. 
        # We have 60s.
        
        # Better: Pass `selected_mask` to the bound function and let it check compatibility on the fly?
        # The greedy bound iterates j from idx to N. For each j, check if `masks[j] & selected_mask == 0`.
        # If yes, it's a candidate. Then pick it, add weight, and add `masks[j]` to a local `local_forbidden`.
        
        def calc_bound(start_idx, sel_mask):
            total = 0.0
            local_forbidden = 0
            for j in range(start_idx, N):
                # Check if j is compatible with current selection
                if sel_mask & masks[j]:
                    continue
                # Check if j is compatible with previously picked nodes in this greedy bound
                if local_forbidden & (1 << j):
                    continue
                
                # Pick j
                total += weights[j]
                local_forbidden |= masks[j]
                
            return total

        bound = calc_bound(idx, selected_mask)
        
        if current_score + bound <= best_score:
            return

        if idx == N:
            if current_score > best_score:
                best_score = current_score
                best_selection = list(selected_list)
            return

        # Option 1: Include idx
        # Check if idx is compatible with current selection
        if not (selected_mask & masks[idx]):
            new_score = current_score + weights[idx]
            new_mask = selected_mask | (1 << idx)
            new_list = selected_list + [idx]
            branch_and_bound(idx + 1, new_score, new_mask, new_list)
            
        # Option 2: Exclude idx
        branch_and_bound(idx + 1, current_score, selected_mask, selected_list)

    # Initial call
    branch_and_bound(0, 0.0, 0, [])
    
    # Convert indices to values
    result = [i + 2 for i in best_selection]
    
    # Ensure nonempty (problem says nonempty, but if best_selection is empty, something is wrong. 
    # 2 is always valid alone. So it won't be empty.)
    if not result:
        result = [2]
        
    return result

# EVOLVE-BLOCK-END
