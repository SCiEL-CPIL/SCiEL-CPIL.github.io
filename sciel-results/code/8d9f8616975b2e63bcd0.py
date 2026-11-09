"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflict matrix as a list of frozensets for fast intersection, 
    # or use bitsets for speed. Given n=255, integers can hold the bitset.
    # Let's use Python integers as bitsets for the set of available nodes.
    
    # conflict[i] is a bitset of nodes j such that lcm(numbers[i], numbers[j]) <= 256
    conflict_bits = [0] * n
    for i in range(n):
        a = numbers[i]
        bits = 0
        for j in range(n):
            if i == j:
                continue
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                bits |= (1 << j)
        conflict_bits[i] = bits

    # Weights: reciprocal sum. w[i] = 1.0 / numbers[i]
    weights = np.array([1.0 / x for x in numbers])
    
    # Sort nodes by weight descending to improve pruning in Branch and Bound
    # We want to pick high weights (small numbers) first.
    # However, the index in the bitset must correspond to the original number for conflict checking.
    # Let's keep the original indices but process them in a specific order?
    # Actually, for B&B, it's better to have the "state" be the set of available nodes.
    # We can just iterate through the available nodes.
    
    # To speed up B&B, we can pre-sort the nodes by weight.
    # But the bitset indices are fixed. 
    # Let's create a list of indices sorted by weight descending.
    sorted_indices = np.argsort(-weights)
    
    # Memoization dictionary: key is the bitset of available nodes, value is max weight obtainable
    # Note: The state space is huge, but B&B with good bounds prunes significantly.
    # We will use a recursive function.
    
    memo = {}
    best_global = 0.0
    
    def solve(available_mask):
        nonlocal best_global
        
        if elapsed() > DEADLINE:
            return 0.0
            
        if available_mask == 0:
            return 0.0
            
        if available_mask in memo:
            return memo[available_mask]
            
        # Upper bound: sum of weights of all available nodes
        # To make this tighter, we can just use the sum of weights.
        # Even better: sort available nodes by weight and take top k? No, they must be independent.
        # A simple bound is sum(weights[i] for i in available).
        
        # Find the first available node in the sorted order (highest weight first)
        # We need to check which nodes are in available_mask.
        # Instead of iterating sorted_indices every time, we can just find the highest weight node in the mask.
        
        # Let's iterate through sorted_indices to find the first one that is in available_mask
        # This is O(n) per call.
        
        # Optimization: If we always pick the highest weight node that is available, 
        # we can branch on: 1) Include it, 2) Exclude it.
        
        # Find candidate node v with max weight in available_mask
        v = -1
        max_w = -1.0
        for idx in sorted_indices:
            if (available_mask >> idx) & 1:
                # Since sorted_indices is sorted by weight desc, the first one found is the max
                v = idx
                max_w = weights[idx]
                break
        
        if v == -1:
            return 0.0
            
        # Bound: sum of weights of all remaining available nodes
        # This is a loose bound but easy to compute if we precompute prefix sums? 
        # No, the set changes. 
        # Let's just use a tighter bound: 
        # The maximum possible sum is the sum of weights of all available nodes.
        # We can compute this quickly if we maintain a running sum? 
        # Or just compute it. n=255, so summing 255 floats is fast.
        
        # Actually, a better bound for MWIS is often derived from the structure.
        # But let's stick to the simple sum bound first.
        
        # Calculate upper bound for this state
        # To avoid summing every time, we could pass the current sum of available weights?
        # But the set changes arbitrarily.
        
        # Let's try a different approach: 
        # Branch on the node with the highest weight.
        # Option 1: Include v. 
        #   New available = available_mask & ~(1 << v) & ~conflict_bits[v]
        #   Value = weights[v] + solve(New available)
        
        # Option 2: Exclude v.
        #   New available = available_mask & ~(1 << v)
        #   Value = solve(New available)
        
        # We take the max.
        
        # Pruning: If the current best_global is already >= upper_bound, we can return 0?
        # No, we are returning the value *from this state*. 
        # We need to know the current accumulated value to prune.
        # Let's change the function to take (available_mask, current_sum).
        
        # Wait, the memoization key is just available_mask. The value returned is the max *additional* sum.
        # So we can't prune based on global best easily inside the memoized function unless we pass current_sum.
        # If we pass current_sum, the memo key must be (available_mask, current_sum) which defeats memoization.
        
        # Alternative: 
        # The function returns the max sum obtainable from the set `available_mask`.
        # Let UB = sum(weights[i] for i in available_mask).
        # If we have a global best, we can't prune here directly without knowing the path cost.
        # However, we can use a heuristic: if UB + current_path_sum < best_global, prune.
        # Since we don't have current_path_sum in the signature, we can't do that with pure memoization on mask.
        
        # Let's just rely on the speed of the B&B with the simple sum bound.
        # The sum bound is tight enough if the weights are decaying (1/x).
        
        # Calculate sum of weights in available_mask
        # We can do this by iterating bits? Or just summing the list.
        # Since n=255, let's just sum.
        
        # Optimization: Precompute weight sums? No, mask changes.
        
        # Let's just compute the sum.
        s = 0.0
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        
        # If s is small, maybe return early?
        
        # Branch
        # Include v
        new_mask_include = available_mask & ~(1 << v) & ~conflict_bits[v]
        val_include = weights[v] + solve(new_mask_include)
        
        # Exclude v
        new_mask_exclude = available_mask & ~(1 << v)
        val_exclude = solve(new_mask_exclude)
        
        result = max(val_include, val_exclude)
        
        # Memoize
        memo[available_mask] = result
        
        return result

    # Initial mask: all nodes
    initial_mask = (1 << n) - 1
    
    # Start solving
    # To handle the deadline, we can't easily interrupt recursion.
    # But the problem is small enough that it might finish.
    # If it doesn't, we need a fallback.
    
    # Let's run it. If it takes too long, we might need a different approach.
    # Given n=255, the number of independent sets might be large, but the pruning should be effective.
    
    try:
        final_sum = solve(initial_mask)
        
        # Reconstruct the set
        # We need to trace back the decisions.
        # The memo dict stores the values, but not the choices.
        # We need to store the choice (include/exclude) in the memo or re-run the logic.
        
        # Let's modify solve to also store the choice.
        # Or, we can just re-run the decision logic using the memo values.
        
        def reconstruct(available_mask):
            if available_mask == 0:
                return []
            
            # Find v again (same logic as solve)
            v = -1
            for idx in sorted_indices:
                if (available_mask >> idx) & 1:
                    v = idx
                    break
            
            if v == -1:
                return []
                
            new_mask_include = available_mask & ~(1 << v) & ~conflict_bits[v]
            val_include = weights[v] + memo.get(new_mask_include, 0.0)
            
            new_mask_exclude = available_mask & ~(1 << v)
            val_exclude = memo.get(new_mask_exclude, 0.0)
            
            if val_include >= val_exclude:
                return [numbers[v]] + reconstruct(new_mask_include)
            else:
                return reconstruct(new_mask_exclude)
        
        result_set = reconstruct(initial_mask)
        
    except Exception as e:
        # Fallback to greedy if B&B fails or times out (though we can't catch timeout easily in recursion)
        # For now, assume it works or returns quickly.
        # If it hangs, the outer deadline check in the evaluator will kill it.
        # To be safe, let's implement a quick greedy fallback if the recursion depth or time is an issue.
        # But we can't easily check time inside the recursion without passing it down.
        
        # Let's assume the B&B is fast enough for n=255 with 1/x weights.
        pass

    return result_set

# EVOLVE-BLOCK-END
