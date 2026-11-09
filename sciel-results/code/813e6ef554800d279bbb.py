"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    n = 30
    # Calculate LCM of 1..30
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
        
    # Weights for integer arithmetic: S = sum(c_i / i) = (1/LCM) * sum(c_i * LCM/i)
    # c_i in {-1, 0, 1}
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    # Split into two halves for meet-in-the-middle
    # First half: indices 0..14 (denominators 1..15)
    # Second half: indices 15..29 (denominators 16..30)
    half = 15
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate all possible sums for the first half and second half
    # To save memory and time, we can use a set of sums? 
    # No, we need to find the min |a+b|. 
    # If we just store the set of sums, we can't easily find the pair.
    # But we can store the set of sums for one half, and iterate the other.
    # Let S1 be the set of sums for w1.
    # Let S2 be the set of sums for w2.
    # We want min |a+b| where a in S1, b in S2, a+b != 0.
    # This is equivalent to: for each b in S2, find a in S1 such that a is close to -b.
    # If we sort S1, we can binary search for -b.
    
    # Generating S1: 3^15 = 14,348,907 elements.
    # Storing 14M integers in a list and sorting is feasible in Python (memory ~100MB, time ~few seconds).
    # Let's generate S1 as a list, sort it.
    # Then for each b in S2 (also 14M), we do a binary search in S1.
    # 14M * log(14M) ~ 14M * 24 = 336M ops. This might be too slow in Python (60s limit).
    # 336M simple ops in Python might take 30-60s. It's risky.
    
    # Alternative: Two-pointer technique on sorted arrays?
    # If we have sorted S1 and sorted S2.
    # We want a+b close to 0.
    # Let i = 0 (start of S1), j = len(S2)-1 (end of S2).
    # S1[i] + S2[j]
    # If sum > 0, decrease j.
    # If sum < 0, increase i.
    # If sum == 0, we found a zero sum. We need the smallest NON-zero.
    # This is tricky.
    
    # Better approach:
    # Since the range of sums is limited (max sum is sum(weights)), we can use a boolean array or set.
    # Max weight is LCM/1 = LCM.
    # LCM of 1..30 is 2329089562800. This is too large for an array.
    # But the number of distinct sums is at most 3^15 = 14M.
    # So using a set is efficient.
    
    # Let's generate S1 as a set.
    # Generating 14M elements in a set.
    # Memory: ~100-200MB.
    # Time: ~2-5 seconds.
    
    # Then, generate S2 sums one by one (or in batches) and check against S1.
    # For each b in S2, we want to find a in S1 such that |a+b| is minimized and non-zero.
    # This requires finding the nearest neighbor in S1 to -b.
    # A set does not support efficient nearest neighbor search.
    # We need S1 sorted for binary search.
    
    # So:
    # 1. Generate S1 list.
    # 2. Sort S1 list.
    # 3. Generate S2 list.
    # 4. Sort S2 list? Or just iterate S2 and binary search in S1?
    # If we iterate S2 (14M elements) and binary search in S1 (14M elements), it's 14M * 24 ops.
    # In Python, a binary search is a loop of ~24 steps. 14M * 24 = 336M steps.
    # This is likely to exceed 60s.
    
    # Can we optimize?
    # We can use the two-pointer approach if we handle the "non-zero" constraint carefully.
    # Or, we can just find the global minimum possible non-zero sum.
    # The minimum possible non-zero integer sum is 1.
    # If we can find a pair (a,b) such that a+b = 1 or a+b = -1, we are done.
    # We can check if 1 is in S1+S2 or -1 is in S1+S2.
    # (S1+S2) is the set of all sums.
    # We can compute the set of all sums? No, that's 3^30, too big.
    
    # Let's try the two-pointer approach on sorted S1 and S2.
    # It is O(N) where N is the size of the lists.
    # N = 14M.
    # O(N) is 14M steps. This is very fast in Python (<< 1s).
    
    # Algorithm:
    # Sort S1, S2.
    # i = 0, j = len(S2)-1.
    # best = inf.
    # while i < len(S1) and j >= 0:
    #    s = S1[i] + S2[j]
    #    if s == 0:
    #       # We found a zero sum. We need to find the smallest non-zero.
    #       # The next smallest non-zero must be either S1[i]+S2[j-1] or S1[i+1]+S2[j] (if they exist and are non-zero).
    #       # Actually, if s==0, we can just move both pointers?
    #       # If we move i++, S1[i+1] >= S1[i]. S1[i+1]+S2[j] >= 0.
    #       # If we move j--, S2[j-1] <= S2[j]. S1[i]+S2[j-1] <= 0.
    #       # So one will be positive, one negative.
    #       # We should check both.
    #       # But to keep it O(N), we can just update best with abs(S1[i]+S2[j-1]) and abs(S1[i+1]+S2[j]) if they are non-zero?
    #       # No, that's not strictly O(N) if we do extra checks, but it's O(1) per step.
    #       # However, if we just move pointers, we might miss the minimum.
    #       # Example: S1=[-1, 1], S2=[-1, 1].
    #       # i=0, j=1: S1[0]+S2[1] = -1+1 = 0.
    #       # Neighbors: S1[0]+S2[0] = -2, S1[1]+S2[1] = 2.
    #       # Min non-zero is 1? Wait.
    #       # S1=[-1, 1], S2=[-1, 1].
    #       # Possible sums: -2, 0, 0, 2.
    #       # Min non-zero is 2.
    #       # My example was bad.
    #       # Let S1=[-1, 0, 1], S2=[-1, 0, 1].
    #       # Sums: -2, -1, 0, -1, 0, 1, 0, 1, 2.
    #       # Min non-zero is 1.
    #       # i=0, j=2: -1+1=0.
    #       # Neighbors: S1[0]+S2[1] = -1+0 = -1. Abs=1.
    #       # S1[1]+S2[2] = 0+1 = 1. Abs=1.
    #       # So we find 1.
    #       # Then we can stop if we find 1, since it's the minimum possible.
    
    #    elif s > 0:
    #       if s < best: best = s
    #       j -= 1
    #    else:
    #       if -s < best: best = -s
    #       i += 1
    
    # This O(N) approach is very promising.
    
    # Step 1: Generate S1
    def gen_sums(ws):
        sums = [0]
        for w in ws:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums

    t0 = time.time()
    sums1 = gen_sums(w1)
    sums1.sort()
    t1 = time.time()
    
    if time.time() > deadline - 10:
        # Fallback if generation is too slow
        return [0]*28 + [1, -1]
        
    sums2 = gen_sums(w2)
    sums2.sort()
    
    t2 = time.time()
    # print(f"Gen1: {t1-t0}, Gen2: {t2-t1}")
    
    # Step 2: Two-pointer scan
    i = 0
    j = len(sums2) - 1
    n1 = len(sums1)
    n2 = len(sums2)
    
    best_abs = float('inf')
    best_a = None
    best_b = None
    
    while i < n1 and j >= 0:
        val = sums1[i] + sums2[j]
        if val == 0:
            # Check neighbors
            # Left neighbor in S2
            if j > 0:
                v = sums1[i] + sums2[j-1]
                if v != 0:
                    av = abs(v)
                    if av < best_abs:
                        best_abs = av
                        best_a = sums1[i]
                        best_b = sums2[j-1]
                        if best_abs == 1:
                            break
            # Right neighbor in S1
            if i < n1 - 1:
                v = sums1[i+1] + sums2[j]
                if v != 0:
                    av = abs(v)
                    if av < best_abs:
                        best_abs = av
                        best_a = sums1[i+1]
                        best_b = sums2[j]
                        if best_abs == 1:
                            break
            # Move pointers to avoid infinite loop and continue searching
            # We must move at least one pointer.
            # If we move both, we might skip a better pair?
            # If val=0, then S1[i] = -S2[j].
            # S1 is increasing, S2 is increasing.
            # S1[i+1] >= S1[i] => S1[i+1] + S2[j] >= 0.
            # S2[j-1] <= S2[j] => S1[i] + S2[j-1] <= 0.
            # So we have one non-negative and one non-positive candidate.
            # We checked them.
            # Can we move both?
            # If we move i++, we consider S1[i+1] + S2[j].
            # If we move j--, we consider S1[i] + S2[j-1].
            # We already checked these specific values.
            # But moving both skips S1[i+1] + S2[j-1].
            # Is S1[i+1] + S2[j-1] a candidate for the minimum?
            # S1[i+1] >= S1[i] and S2[j-1] <= S2[j].
            # So S1[i+1] + S2[j-1] can be anything.
            # However, the two-pointer algorithm is typically correct for "closest pair to 0" if we handle 0 carefully.
            # To be safe, let's just move one pointer?
            # If we move i++, we keep j. Next val = S1[i+1] + S2[j] >= 0.
            # If we move j--, we keep i. Next val = S1[i] + S2[j-1] <= 0.
            # We should probably move the one that brings us closer to 0?
            # But we don't know which side has the smaller absolute value.
            # However, we already updated best_abs with the immediate neighbors.
            # If we move both, we might miss a smaller value further away?
            # No, because the values are sorted.
            # If S1[i]+S2[j]=0, then S1[i+1]+S2[j] is the smallest positive sum involving S2[j].
            # And S1[i]+S2[j-1] is the largest negative sum involving S1[i].
            # Any other sum involving S1[i] or S2[j] will be further from 0.
            # So we can safely move both pointers?
            # Wait, what if S1[i+1]+S2[j-1] is closer to 0 than S1[i+1]+S2[j] or S1[i]+S2[j-1]?
            # Example: S1=[-10, -9], S2=[9, 10].
            # i=0, j=1: -10+10=0.
            # Neighbors: S1[0]+S2[0] = -1. Abs=1.
            # S1[1]+S2[1] = 1. Abs=1.
            # Move both: i=1, j=0.
            # S1[1]+S2[0] = -9+9=0.
            # Neighbors: S1[1]+S2[-1] (invalid), S1[2]+S2[0] (invalid).
            # So we found 1.
            
            # Example 2: S1=[-5, 5], S2=[-5, 5].
            # i=0, j=1: -5+5=0.
            # Neighbors: S1[0]+S2[0] = -10. S1[1]+S2[1] = 10.
            # best_abs=10.
            # Move both: i=1, j=0.
            # S1[1]+S2[0] = 5-5=0.
            # Neighbors: S1[1]+S2[-1] (invalid), S1[2]+S2[0] (invalid).
            # best_abs remains 10.
            # But the true min non-zero is 0? No, 0 is excluded.
            # Sums: -10, 0, 0, 10. Min non-zero is 10. Correct.
            
            # Example 3: S1=[-1, 1], S2=[-2, 2].
            # i=0, j=1: -1+2=1. best=1. j=0.
            # i=0, j=0: -1-2=-3. best=1. i=1.
            # i=1, j=0: 1-2=-1. best=1. i=2. Stop.
            # Min non-zero is 1. Correct.
            
            # It seems moving both is safe if we check neighbors.
            i += 1
            j -= 1
        elif val > 0:
            if val < best_abs:
                best_abs = val
                best_a = sums1[i]
                best_b = sums2[j]
                if best_abs == 1:
                    break
            j -= 1
        else: # val < 0
            av = -val
            if av < best_abs:
                best_abs = av
                best_a = sums1[i]
                best_b = sums2[j]
                if best_abs == 1:
                    break
            i += 1
            
    # Now we have best_a and best_b.
    # We need to find coefficients.
    # We can use the DFS approach again.
    
    def find_coeffs(weights, target):
        n = len(weights)
        coeffs = [0] * n
        
        # Suffix sums for pruning
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + weights[i]
            
        def dfs(idx, rem):
            if idx == n:
                return rem == 0
            if abs(rem) > suffix_sums[idx]:
                return False
            for c in [-1, 0, 1]:
                coeffs[idx] = c
                if dfs(idx + 1, rem - c * weights[idx]):
                    return True
            return False
            
        if dfs(0, target):
            return coeffs
        return None
        
    if best_a is not None and best_b is not None:
        c1 = find_coeffs(w1, best_a)
        c2 = find_coeffs(w2, best_b)
        if c1 and c2:
            return c1 + c2
            
    # Fallback
    return [0] * (n - 2) + [1, -1]

# EVOLVE-BLOCK-END
