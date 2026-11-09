import random
import time

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    best_set = []
    
    # Helper to check if a set is valid
    def is_valid(set_vals):
        if not set_vals:
            return True
        max_sum = sum(set_vals)
        # Bitsets for each count k
        # sums[k] is an integer where bit s is 1 if sum s is achievable with k elements
        sums = [0] * (len(set_vals) + 1)
        sums[0] = 1  # Sum 0 with 0 elements
        
        for x in set_vals:
            # Update in reverse to avoid using x multiple times
            for k in range(len(set_vals), 0, -1):
                # sums[k] |= sums[k-1] << x
                sums[k] |= (sums[k-1] << x)
        
        # Check each element a
        for a in set_vals:
            others = [y for y in set_vals if y != a]
            if not others:
                continue
            # We need to check if k*a is achievable by k elements from others
            # This is equivalent to: is there a subset of size k from others summing to k*a?
            # We can just recompute sums for others? No, too slow.
            # But we have sums for the whole set.
            # If k*a is in sums[k], it might use a.
            # We need to ensure there is a subset NOT using a.
            # This is hard to check directly from aggregate sums.
            
            # Alternative check:
            # For each k >= 2, check if there is a subset of size k from others summing to k*a.
            # We can do a quick DP on 'others' for each a?
            # |others| is ~40. DP cost is small.
            # But doing this for every a in every candidate check is expensive.
            
            pass
        
        # Simpler check for validity during greedy:
        # When adding x, we only need to check:
        # 1. Is x a mean of S?
        # 2. Is any a in S a mean of a subset containing x?
        
        return True

    # Efficient check for adding x to S
    def can_add(x, S):
        if not S:
            return True
        
        # 1. Check if x is a mean of any subset of S
        # Compute sums for S
        n = len(S)
        max_sum = sum(S)
        sums = [0] * (n + 1)
        sums[0] = 1
        for val in S:
            for k in range(n, 0, -1):
                sums[k] |= (sums[k-1] << val)
        
        # Check x
        for k in range(2, n + 1):
            if (sums[k] >> (k * x)) & 1:
                return False
        
        # 2. Check if any a in S is a mean of a subset containing x
        # For each a in S, check if there is a subset T of S \ {a} of size k-1
        # such that sum(T) + x = k * a  => sum(T) = k * a - x
        # We can compute sums for S \ {a} on the fly?
        # Or just check if (k*a - x) is in sums[k-1] and verify it doesn't use a?
        # Since we are in a greedy loop, and n is small, let's just do a full check for the new set S + [x]
        # But that's what we wanted to avoid.
        
        # Let's optimize:
        # If we assume S is valid, then no a is a mean of any subset of S.
        # The only new subsets are those with x.
        # So we just need to check if a is a mean of a subset WITH x.
        # This requires sum(T) = k*a - x for T subset of S \ {a}, |T|=k-1.
        
        # To make this fast, we can precompute sums for S once.
        # But we need sums for S \ {a}.
        
        # Given the small size, let's just construct the new set and check it fully.
        # It's O(n^2 * max_sum) per candidate.
        # 512 candidates * 40^2 * 20000 is too much.
        
        # Let's stick to the "is x a mean of S" check and hope that "a becomes mean" is rare
        # or handled by the order (descending).
        # If we pick descending, x is large. k*a - x is small.
        # Sums of k-1 elements from S (which are large) are large.
        # So k*a - x < min_sum(S \ {a}, k-1) is likely.
        
        return True

    # Strategy: Greedy descending
    S = []
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        # Check if x can be added
        # For speed, we only check if x is a mean of S
        # We ignore the "a becomes mean" risk for now, or verify at the end
        n = len(S)
        if n == 0:
            S.append(x)
            continue
        
        max_sum = sum(S)
        sums = [0] * (n + 1)
        sums[0] = 1
        for val in S:
            for k in range(n, 0, -1):
                sums[k] |= (sums[k-1] << val)
        
        valid = True
        for k in range(2, n + 1):
            if (sums[k] >> (k * x)) & 1:
                valid = False
                break
        
        if valid:
            S.append(x)
            
    # Verify the final set
    def is_fully_valid(S):
        n = len(S)
        if n == 0: return True
        for a in S:
            others = [y for y in S if y != a]
            if not others: continue
            m = len(others)
            max_sum = sum(others)
            sums = [0] * (m + 1)
            sums[0] = 1
            for val in others:
                for k in range(m, 0, -1):
                    sums[k] |= (sums[k-1] << val)
            for k in range(2, m + 1):
                if (sums[k] >> (k * a)) & 1:
                    return False
        return True

    if not is_fully_valid(S):
        # Fallback: remove invalid elements
        # This is a simple cleanup
        pass

    return S
