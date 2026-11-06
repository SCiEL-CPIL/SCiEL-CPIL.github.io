import time
from math import lcm, gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 55.0
    
    # Numbers 2..256
    N = 255
    nums = list(range(2, 257))
    
    # Precompute incompatibility: two numbers are incompatible if lcm(a,b) <= 256
    # incompat[i] is a bitmask of all j != i such that lcm(nums[i], nums[j]) <= 256
    incompat = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            if lcm(nums[i], nums[j]) <= 256:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    # We want to select a subset S such that for all distinct a,b in S, lcm(a,b) > 256
    # This is equivalent to: no two selected numbers are incompatible.
    # So we're finding a maximum-weight independent set in the incompatibility graph,
    # where weight of node i is 1/nums[i].
    
    # Sort nodes by weight descending (i.e., by nums[i] ascending) for better B&B pruning
    order = list(range(N))
    order.sort(key=lambda i: nums[i])  # smallest nums first = largest reciprocal first
    
    # Re-index so that we process in this order
    # Let's keep original indices but process in sorted order
    
    # Reciprocals as Fractions for exact computation
    recip = [Fraction(1, nums[i]) for i in range(N)]
    
    # Upper bound: sum of reciprocals of all remaining nodes that are compatible with current selection
    # For B&B, we need a good upper bound on the remaining contribution.
    
    best_sum = Fraction(0, 1)
    best_set = None
    
    # Branch and Bound
    # State: (idx, current_mask_of_selected, current_sum)
    # idx: next index in 'order' to consider
    # current_mask: bitmask of already-selected nodes (in original indices)
    # current_sum: Fraction sum of reciprocals of selected nodes
    
    # We'll use recursion with memoization-free B&B
    # Key insight: process nodes in order of decreasing weight (increasing nums)
    
    def bnb(idx, selected_mask, current_sum):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if idx == N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = selected_mask
            return
        
        node = order[idx]
        
        # Option 1: Include node
        # Check if node is compatible with all selected nodes
        if (incompat[node] & selected_mask) == 0:
            new_mask = selected_mask | (1 << node)
            new_sum = current_sum + recip[node]
            
            # Upper bound check: even if we take all remaining compatible nodes, can we beat best?
            # Simple upper bound: current_sum + sum of reciprocals of all nodes from idx+1 onwards
            # that are not incompatible with node and not incompatible with each other... 
            # For a tighter bound, just sum all remaining reciprocals (loose but fast)
            # Actually, let's use a looser but faster bound first
            
            bnb(idx + 1, new_mask, new_sum)
        
        # Option 2: Exclude node
        bnb(idx + 1, selected_mask, current_sum)
    
    # The above naive B&B will be too slow. Let's use a better approach.
    
    # Better approach: Use the fact that the incompatibility graph has structure.
    # Numbers with large values (close to 256) have fewer incompatibilities.
    # Let's try a different strategy: greedy with local search, or a smarter B&B.
    
    # Let's try: sort by nums ascending (largest recip first), and use B&B with a good upper bound.
    # Upper bound at step idx: current_sum + sum of reciprocals of all unprocessed nodes 
    # that are compatible with the current selection.
    
    # Precompute suffix sums of reciprocals in processing order for quick upper bounds
    suffix_recip_sum = [Fraction(0, 1)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_recip_sum[i] = suffix_recip_sum[i + 1] + recip[order[i]]
    
    best_sum = Fraction(0, 1)
    best_set = 0
    
    def bnb2(idx, selected_mask, current_sum):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_sum + sum of all remaining nodes' reciprocals
        # (loose, but fast)
        ub = current_sum + suffix_recip_sum[idx]
        if ub <= best_sum:
            return
        
        if idx == N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = selected_mask
            return
        
        node = order[idx]
        
        # Option 1: Include node if compatible
        if (incompat[node] & selected_mask) == 0:
            new_mask = selected_mask | (1 << node)
            new_sum = current_sum + recip[node]
            bnb2(idx + 1, new_mask, new_sum)
        
        # Option 2: Exclude node
        bnb2(idx + 1, selected_mask, current_sum)
    
    bnb2(0, 0, Fraction(0, 1))
    
    # Extract result
    if best_set == 0:
        # Fallback: return a single valid element
        return [2]
    
    result = []
    for i in range(N):
        if best_set & (1 << i):
            result.append(nums[i])
    
    return result
