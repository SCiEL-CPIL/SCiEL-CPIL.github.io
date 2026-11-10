import random

def construct():
    max_n = 512
    max_size = 48
    
    def is_valid_set(C):
        # C is a list of distinct integers
        n = len(C)
        if n == 0:
            return True
            
        # For each x in C, check if x is the mean of any subset of C \ {x} with size >= 2
        # This is equivalent to: 0 is a subset sum of {y - x for y in C, y != x} with size >= 2
        # Since y != x, y - x != 0.
        # So any subset summing to 0 must have size >= 2 (size 1 would imply y-x=0).
        
        # Optimization: If we find any violation, return False.
        
        # We can use a global offset to handle negative sums in the bitset.
        # Max sum magnitude: 48 * 512 = 24576.
        # Offset = 25000.
        offset = 25000
        
        for x in C:
            others = [y - x for y in C if y != x]
            M = len(others)
            
            # dp[k] is a bitset of sums reachable with exactly k elements
            # We only care if sum 0 is reachable with k >= 2
            # We can use a single bitset if we track min/max size?
            # No, just use array of bitsets.
            
            # To save memory/time, we can just check if 0 is reachable at all, 
            # and if the only way is empty set?
            # But empty set is size 0.
            # If we find a non-empty subset summing to 0, it must be size >= 2.
            
            # Let's use a bitset `reachable` where bit `s` is set if sum `s` is reachable.
            # But we need to distinguish empty set.
            # We can initialize `reachable` to 0.
            # But then we can't add elements.
            
            # Standard trick:
            # dp[0] = 1 << offset
            # for v in others:
            #   for k from M down to 1:
            #     dp[k] |= dp[k-1] shifted by v
            # This is O(M^2) bitset ops.
            
            # Faster:
            # We just need to know if there is ANY subset of size >= 2 summing to 0.
            # This is equivalent to: Is there a subset S of others, |S| >= 2, such that sum(S) = 0?
            
            # Let's just use the array of bitsets. It's robust.
            dp = [0] * (M + 1)
            dp[0] = 1 << offset
            
            for v in others:
                # Update dp backwards
                # We only need to go up to M
                for k in range(M - 1, -1, -1):
                    if dp[k]:
                        if v >= 0:
                            dp[k+1] |= (dp[k] << v)
                        else:
                            dp[k+1] |= (dp[k] >> (-v))
            
            target_bit = 1 << offset
            # Check if target_bit is set in any dp[k] for k >= 2
            valid = True
            for k in range(2, M + 1):
                if dp[k] & target_bit:
                    valid = False
                    break
            
            if not valid:
                return False
        
        return True

    # Try to find a valid set of size max_size
    # Use random sampling
    for _ in range(1000): # Try many times
        C = random.sample(range(1, max_n + 1), max_size)
        if is_valid_set(C):
            return C
    
    # If 48 fails, try smaller
    for size in range(max_size - 1, 0, -1):
        for _ in range(1000):
            C = random.sample(range(1, max_n + 1), size)
            if is_valid_set(C):
                return C
    
    # Fallback: return empty or single element
    return [1]
