import math
from collections import defaultdict

def construct():
    # We want to find a subset S of {211, ..., M} such that 
    # XOR of kernels of elements in S is 210.
    # Minimize M.
    
    # Target kernel is 210.
    # 210 = 2 * 3 * 5 * 7.
    
    # We iterate M from 211 upwards.
    # states: dict mapping XOR sum -> (prev_xor, number_used)
    # We start with state 0: ()
    
    # To save memory, we can just keep a set of reachable states 
    # and a dict to trace back.
    # However, the number of reachable states can grow.
    # But for small M, it's fine.
    
    # Let's limit the search to M=10210.
    
    # kernel computation
    # We can precompute kernels up to 10210.
    limit = 10210
    kernels = [1] * (limit + 1)
    for i in range(2, limit + 1):
        if kernels[i] == 1: # prime
            for j in range(i, limit + 1, i):
                temp = j
                exp = 0
                while temp % i == 0:
                    temp //= i
                    exp += 1
                if exp % 2 == 1:
                    kernels[j] *= i
    
    # BFS/DP
    # state: xor_sum
    # We want to reach 210.
    
    # reachable[xor_sum] = (prev_xor_sum, number)
    reachable = {0: None}
    
    # We add numbers one by one.
    # When we add a number x with kernel k, 
    # new_states = {s ^ k for s in reachable}
    # If 210 is in new_states, we found a solution with max x.
    
    # To reconstruct, we need to store the path.
    # Since we add numbers in increasing order, the first time we reach 210
    # will be with the smallest possible max value.
    
    # Optimization: The number of states might explode.
    # But the kernels are products of primes.
    # Let's try it.
    
    for x in range(211, limit + 1):
        k = kernels[x]
        # We need to update reachable states.
        # We iterate over a copy of current states to avoid using x multiple times 
        # in the same step? 
        # Actually, each number can be used at most once.
        # So we should generate new states from the *previous* set of states.
        
        # However, if we just do `for s in list(reachable.keys())`, 
        # we might add states that use x multiple times if we update in place.
        # So we must collect new states first.
        
        new_states = {}
        for s, parent in reachable.items():
            ns = s ^ k
            if ns not in reachable: # Only add if not already reachable with a smaller max?
                # Actually, if ns is already in reachable, it was reached with a smaller max value (since we are adding x in increasing order).
                # So we don't need to update it.
                # But wait, we want to minimize the MAX value.
                # If we reach a state with max 100, and then reach the same state with max 200, 
                # the max 100 one is better for any future extension?
                # No, the max value is determined by the largest number in the set.
                # If we have a set S1 with max 100 reaching state s, and S2 with max 200 reaching state s.
                # If we add a number 300 to both, the resulting max is 300 in both cases.
                # So the history of how we reached s doesn't matter for the final max, 
                # ONLY that s is reachable.
                # BUT, we need the set of numbers to return.
                # If we return the set with max 100, and then add 300, the max is 300.
                # If we return the set with max 200, and then add 300, the max is 300.
                # So we can just keep ONE parent for each state?
                # Yes, because we only care about reachability.
                # The "cost" is the max value, which is determined by the step we are at.
                # Since we process x in increasing order, any state reached at step x has max <= x.
                # If a state was reached at step y < x, it has max <= y < x.
                # So the earlier reaching is "better" or at least as good.
                
                # So we only add ns if it's not already in reachable.
                new_states[ns] = (s, x)
        
        # Add new states to reachable
        for ns, parent in new_states.items():
            reachable[ns] = parent
            
        if 210 in reachable:
            # Reconstruct the set
            S = []
            cur = 210
            while cur != 0:
                prev, x_used = reachable[cur]
                S.append(x_used)
                cur = prev
            return S
    
    # If not found (should not happen), fallback
    return [840]
