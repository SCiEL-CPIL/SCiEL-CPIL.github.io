"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n = 512
    max_size = 48
    
    def get_subset_bitset(elements):
        """Returns an integer bitset where bit i is 1 if i is a subset sum of elements."""
        bitset = 1
        for e in elements:
            bitset |= (bitset << e)
        return bitset

    def is_subset_sum_present(bitset, target):
        """Checks if target sum is achievable."""
        if target < 0:
            return False
        return (bitset >> target) & 1

    def can_add(x, S):
        """
        Checks if x can be added to S without violating the condition.
        Condition: No element in S U {x} is the mean of a subset of the others (size >= 2).
        """
        # 1. Check if x is the mean of a subset of S (size >= 2)
        # x = sum(T)/k => sum(T) = k*x
        # T is a subset of S.
        bitset_S = get_subset_bitset(S)
        for k in range(2, len(S) + 1):
            target = k * x
            if target > sum(S):
                break
            if is_subset_sum_present(bitset_S, target):
                return False
        
        # 2. Check if any a in S is the mean of a subset of (S U {x}) \ {a}
        # The subset T must be from (S U {x}) \ {a}.
        # T can be from S \ {a} (already valid) or T = T' U {x} where T' subset S \ {a}.
        # If T includes x:
        # mean(T) = (sum(T') + x) / (|T'| + 1) = a
        # sum(T') + x = a * (|T'| + 1)
        # sum(T') = a * |T'| + a - x
        # Let k' = |T'|. We need k' >= 1 (since |T| >= 2).
        # We need to check if there exists k' >= 1 such that sum(T') = a * k' + a - x
        # is a subset sum of S \ {a} of size k'.
        
        # Note: We don't strictly need to track size k' in the bitset if we just check existence,
        # BUT the target sum depends on k'.
        # However, if a sum `s` is achievable by a subset of `S \ {a}`, it is achievable by some size.
        # The condition is: Is there a subset `T'` of `S \ {a}` with sum `a * |T'| + a - x`?
        # This is a self-referential condition on the size.
        
        # Let's iterate through possible k' values.
        # k' can range from 1 to len(S) - 1.
        
        for a in S:
            others = [y for y in S if y != a]
            # We can optimize by precomputing bitset for others, but S is small.
            # Actually, we can just check the condition.
            # Is there a subset T' of others with sum = a * k' + a - x for some k' >= 1?
            
            # Let's get the bitset of others.
            bitset_others = get_subset_bitset(others)
            
            # We need to check if for any k' in [1, len(others)], 
            # the target sum target_kp = a * k' + a - x is a subset sum of others.
            # AND the subset achieving it must have size k'.
            # The bitset only tells us if the sum is achievable, not by what size.
            # However, if a sum is achievable, it is achievable by SOME size.
            # Does the size matter?
            # The equation was derived from `sum(T') = a * k' + a - x`.
            # If we find a subset `T'` with sum `S_target`, its size is `|T'|`.
            # The condition requires `S_target = a * |T'| + a - x`.
            # So we can't just check if `S_target` is in the bitset. We must ensure the size matches.
            
            # To handle size, we need a 2D DP or a list of bitsets per size.
            # Let's use a list of bitsets `bp[k]` for `S \ {a}`.
            
            bp = [0] * (len(others) + 1)
            bp[0] = 1
            for y in others:
                for k in range(len(others), 0, -1):
                    bp[k] |= (bp[k-1] << y)
            
            for k_prime in range(1, len(others) + 1):
                target_sum = a * k_prime + a - x
                if target_sum < 0:
                    continue
                if target_sum > sum(others):
                    break # Assuming a > 0, x > 0, this might not break early if a is large? 
                          # Actually target_sum increases with k_prime.
                if is_subset_sum_present(bp[k_prime], target_sum):
                    return False
        
        return True

    # Greedy construction
    # We will try multiple random orders and keep the best set.
    best_set = []
    
    candidates = list(range(1, n + 1))
    
    # Try a few random shuffles
    for _ in range(10):
        random.shuffle(candidates)
        current_set = []
        for x in candidates:
            if len(current_set) >= max_size:
                break
            if can_add(x, current_set):
                current_set.append(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) == max_size:
                break
                
    # If we didn't reach max size, try a greedy from 1 to 512
    # Or from 512 to 1
    # The random approach is usually better for packing.
    
    return best_set

# EVOLVE-BLOCK-END
