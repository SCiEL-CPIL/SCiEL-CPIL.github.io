"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Numbers are in [2, 256]
    N = 255
    # Map index i to number i+2
    # We want to maximize sum(1/x).
    # Conflict: lcm(a, b) <= 256.
    
    # Precompute conflict bitmasks for fast compatibility checks.
    # conflict_mask[i] is a bitmask where bit j is 1 if number (i+2) conflicts with (j+2).
    # Note: We only care about j > i for building, but mask is symmetric.
    conflict_mask = [0] * N
    
    # Also precompute weights as Fractions for exact comparison if needed, 
    # or floats for speed. Given the small size, floats are likely sufficient 
    # for ordering, but let's be careful. The task says "Maximize exact reciprocal sum".
    # However, B&B bounds usually use floats. We will keep the best set and 
    # calculate the exact sum at the end.
    
    # Build conflict masks
    for i in range(N):
        a = i + 2
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 2
            g = gcd(a, b)
            # lcm(a, b) = a * b // g
            # Check if lcm <= 256
            if a * b // g <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Weights: 1/x
    weights = [1.0 / (i + 2) for i in range(N)]
    
    # Sort indices by weight descending (highest value first) to improve B&B pruning.
    # If we process high value nodes first, we find good solutions early,
    # which tightens the upper bound for pruning.
    order = list(range(N))
    order.sort(key=lambda x: -weights[x])
    
    # Re-index? No, keep original indices but iterate in 'order'.
    # The B&B state needs to know which nodes are "available" (compatible with selected).
    
    best_score = 0.0
    best_set = []
    
    # Global variables for B&B
    # To avoid recursion depth issues or overhead, we might use iterative or optimized recursion.
    # N=255 is small enough for recursion if pruned well, but Python recursion limit might be an issue.
    # Let's increase recursion limit or use an explicit stack if needed. 
    # However, B&B on independent set can be deep.
    # Let's try a recursive approach with a time check.
    
    import sys
    sys.setrecursionlimit(5000)
    
    def solve(idx, current_mask, current_sum, current_set):
        nonlocal best_score, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if idx == len(order):
            if current_sum > best_score:
                best_score = current_sum
                best_set = list(current_set)
            return
            
        # Upper Bound Calculation:
        # The maximum possible additional sum is the sum of weights of all remaining nodes 
        # in 'order' starting from idx that are compatible with current_mask.
        # A tighter bound: Sum of weights of all remaining nodes compatible with current_mask.
        # We can precompute suffix sums of weights for the 'order' to get a quick bound,
        # but we must filter by compatibility.
        
        # Optimization: 
        # If current_sum + sum(weights[order[idx]:]) <= best_score, prune.
        # This is a loose bound because it ignores conflicts among remaining nodes and 
        # compatibility with current_mask.
        
        # Better Bound:
        # Calculate sum of weights of nodes in order[idx:] that are NOT in conflict with current_mask.
        # This is still an upper bound (ignores conflicts among the remaining nodes themselves).
        
        bound = current_sum
        # Iterate through remaining candidates to compute bound
        # To speed this up, we can precompute prefix sums of weights in 'order'?
        # But the compatibility check depends on current_mask which changes.
        
        # Let's just iterate. N is 255.
        # In the worst case, this is O(N^2) per node, which might be slow.
        # Alternative: 
        # Precompute for each node i, the list of subsequent nodes in 'order' that are compatible?
        # No, compatibility is dynamic based on current_mask.
        
        # Let's try a simple bound first: Sum of all remaining weights.
        # If that's too loose, we'll compute the compatible sum.
        
        # To make it faster, let's precompute suffix sums of weights in the 'order' array.
        # suffix_sum[i] = sum(weights[order[k]] for k in i..N-1)
        # bound = current_sum + suffix_sum[idx]
        # If bound <= best_score: return
        
        # But we can do better:
        # bound = current_sum + sum(weights[order[k]] for k in idx..N-1 if not (conflict_mask[order[k]] & current_mask))
        
        # Let's implement the tighter bound.
        b = current_sum
        for k in range(idx, len(order)):
            node = order[k]
            # Check if node is compatible with current selection (represented by current_mask)
            # current_mask has bits set for all nodes that CONFLICT with the selected set?
            # Or does current_mask represent the selected set?
            # Let's define current_mask as the bitmask of SELECTED nodes.
            # Then a new node 'node' is compatible if (conflict_mask[node] & current_mask) == 0.
            
            if (conflict_mask[node] & current_mask) == 0:
                b += weights[node]
                
        if b <= best_score:
            return
            
        node = order[idx]
        
        # Option 1: Include node
        if (conflict_mask[node] & current_mask) == 0:
            # New mask includes node
            new_mask = current_mask | (1 << node)
            current_set.append(node)
            solve(idx + 1, new_mask, current_sum + weights[node], current_set)
            current_set.pop()
            
        # Option 2: Exclude node
        solve(idx + 1, current_mask, current_sum, current_set)

    # Initial call
    # current_mask = 0 (no nodes selected)
    # current_sum = 0
    # current_set = []
    
    # One optimization: We can start with a greedy solution to set a good initial best_score.
    # Greedy: Pick highest weight node compatible with current set.
    greedy_set = []
    greedy_mask = 0
    greedy_sum = 0.0
    for i in range(N):
        if (conflict_mask[i] & greedy_mask) == 0:
            greedy_set.append(i)
            greedy_mask |= (1 << i)
            greedy_sum += weights[i]
            
    best_score = greedy_sum
    best_set = list(greedy_set)
    
    # Now run B&B to see if we can beat the greedy solution.
    # Since the greedy solution is a valid lower bound, B&B will prune branches that can't beat it.
    
    solve(0, 0, 0.0, [])
    
    # Convert best_set indices back to numbers
    result_numbers = [idx + 2 for idx in best_set]
    
    # Final verification
    # Check validity
    for i in range(len(result_numbers)):
        for j in range(i+1, len(result_numbers)):
            a = result_numbers[i]
            b = result_numbers[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                # Fallback to greedy if B&B failed or produced invalid result
                # (Should not happen if logic is correct, but safety first)
                return [idx + 2 for idx in greedy_set]
                
    return result_numbers

# EVOLVE-BLOCK-END
