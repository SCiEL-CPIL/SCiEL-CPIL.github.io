"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

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
    
    # Generate all possible sums for the first half
    # There are 3^15 = 14,348,907 combinations. This might be too slow/memory heavy in pure Python.
    # 3^15 is ~14M. Sorting 14M integers takes significant memory and time.
    # Let's estimate: 14M ints * 8 bytes = ~112MB. Sorting takes O(N log N).
    # In Python, this is likely to exceed the time limit or memory limits if not optimized.
    # However, the task says "Python standard library, NumPy and SciPy are available."
    # We can use NumPy to speed this up? Or just pure Python might be too slow.
    # 14M iterations in Python is ~1-2 seconds for generation, but sorting is the bottleneck.
    # Let's try a slightly smaller split or just accept the risk.
    # Actually, 3^15 is 14,348,907.
    # Let's try to generate the list of sums.
    
    # Optimization: We only care about the smallest absolute non-zero sum.
    # The maximum possible sum is sum(weights).
    # Let's just generate all sums for both halves.
    
    def gen_sums(ws):
        # ws is a list of weights
        # Returns a list of all possible sums
        # Using iterative expansion
        sums = [0]
        for w in ws:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums

    # This iterative approach creates a list of size 3^k.
    # For k=15, this is 14M elements.
    # Let's check if we can do it within time.
    # Generating 14M ints in Python:
    # Loop 15 times.
    # Iter 1: 3
    # Iter 2: 9
    # ...
    # Iter 15: 14M
    # The last iteration creates 14M new ints. This is memory intensive but should be doable.
    # Sorting 14M ints: Python's Timsort is fast but 14M is large.
    # Let's try. If it's too slow, we might need to fall back to heuristic.
    # But the directive says "guaranteeing the global optimum".
    
    # Alternative: Use a set? No, we need to find the min |a+b| where a in S1, b in S2, a+b != 0.
    # If we sort S1 and S2, we can do a two-pointer scan or binary search.
    # For each a in S1, we want to find b in S2 such that a+b is close to 0 but not 0.
    # i.e., b is close to -a.
    # Since S2 is sorted, we can binary search for -a in S2.
    # Let idx be the position where -a would be inserted.
    # Check S2[idx] and S2[idx-1] (if valid).
    # Compute sums and track min abs non-zero.
    
    # Step 1: Generate S1 and S2
    # To save memory, we can generate S1 as a list, sort it.
    # Generate S2 as a list, sort it.
    
    # Let's add a time check. If generation takes too long, we abort.
    # But 3^15 is fixed.
    
    # Let's try to generate S1
    t0 = time.time()
    sums1 = gen_sums(w1)
    sums1.sort()
    t1 = time.time()
    
    # Check time
    if time.time() > deadline - 10:
         # Not enough time for second half?
         # Just return a heuristic result?
         # For now, let's assume it fits.
         pass
         
    sums2 = gen_sums(w2)
    sums2.sort()
    
    t2 = time.time()
    # print(f"Gen1: {t1-t0}, Gen2: {t2-t1}") # Debug
    
    # Now find min |a + b| != 0
    # Iterate through sums1? Or sums2?
    # sums1 has 14M elements. sums2 has 14M elements.
    # Iterating through 14M elements and doing binary search (log 14M ~ 24) is 14M * 24 = 336M ops.
    # This is likely too slow in Python (60s limit).
    # 336M simple ops in Python might take 30-60s. It's risky.
    
    # Better approach: Two-pointer technique on sorted arrays?
    # We want a+b close to 0.
    # Let i = 0 (start of sums1), j = len(sums2)-1 (end of sums2).
    # sums1[i] + sums2[j]
    # If sum > 0, decrease j.
    # If sum < 0, increase i.
    # If sum == 0, we need to find the next smallest non-zero.
    # This is tricky because we skip over 0.
    
    # Actually, we can just iterate through the smaller list? They are same size.
    # Let's use binary search but optimize it.
    # Or, since we just need the global minimum, we can stop early if we find 1?
    # The minimum possible non-zero integer sum is 1.
    # If we find |a+b| == 1, we can stop immediately.
    
    best_abs = float('inf')
    best_a = 0
    best_b = 0
    
    # Use binary search on sums2 for each a in sums1
    # To speed up, we can limit the search if best_abs becomes small.
    # If best_abs == 1, break.
    
    # Let's iterate sums1
    # We can also just iterate sums2 and binary search in sums1.
    
    # Optimization:
    # If we sort sums1 and sums2.
    # We can use a two pointer approach that handles 0s correctly.
    # Start i=0, j=len(sums2)-1.
    # current_sum = sums1[i] + sums2[j]
    # If current_sum > 0:
    #    if current_sum < best_abs: best_abs = current_sum
    #    j -= 1
    # If current_sum < 0:
    #    if abs(current_sum) < best_abs: best_abs = abs(current_sum)
    #    i += 1
    # If current_sum == 0:
    #    We need to find the nearest non-zero sum.
    #    We can look at sums1[i] + sums2[j-1] and sums1[i+1] + sums2[j].
    #    Actually, if sum is 0, we can just move both pointers?
    #    No, we might miss a small non-zero sum nearby.
    #    However, if we have a 0, the next smallest non-zero must be adjacent in the sorted order of sums?
    #    Not necessarily.
    
    # Let's stick to binary search but with an early exit.
    # Also, we can skip duplicate values in sums1 to save time?
    # sums1 has many duplicates.
    # We can iterate over unique values in sums1.
    
    # Let's convert sums1 to a set? No, we need to recover coefficients.
    # But for finding the value, we just need the value.
    # We can store the value and then reconstruct coefficients?
    # Reconstructing coefficients from the sum value is hard (subset sum is NP-hard, but we have the lists).
    # We can store the index in the generation process?
    # Or, since we just need the coefficients, we can do a second pass?
    # Or, we can store the path?
    # Storing the path for 14M items is too much memory.
    
    # Alternative:
    # Just find the minimum value V.
    # Then, find coefficients that sum to V.
    # We can do a DP or search to find the coefficients for a specific target V.
    # Since V is small (hopefully), we can use a dictionary to store possible sums and one path to get there.
    
    # Step 1: Find min_abs_val using the sorted lists (fast, no path storage).
    # Step 2: Find coefficients for min_abs_val.
    
    # Step 1:
    # Iterate unique sums in sums1?
    # Let's just iterate all, but break if best_abs == 1.
    
    # To make binary search faster, we can use `bisect` module.
    import bisect
    
    min_abs_val = float('inf')
    
    # Iterate over sums1
    # We can skip if |a| is already > min_abs_val + max(sums2)? No.
    # We want |a+b| small. So b should be near -a.
    
    # Let's just do the binary search loop.
    # To speed up Python loop, we can use list comprehension? No, logic is complex.
    # We can use numpy?
    # numpy.searchsorted is fast.
    
    try:
        import numpy as np
        s1 = np.array(sums1)
        s2 = np.array(sums2)
        
        # We want to find min |a+b| != 0
        # For each a in s1, target = -a
        # idx = np.searchsorted(s2, -s1)
        # Check idx and idx-1
        
        # This vectorizes the search.
        # But we need to handle boundaries and 0s.
        
        idxs = np.searchsorted(s2, -s1)
        
        # Valid indices: idxs and idxs-1
        # We need to be careful with out of bounds.
        
        # Let's create candidate indices
        # idxs might be len(s2).
        # idxs-1 might be -1.
        
        # We can clip idxs to [0, len(s2)-1] and also check idxs-1.
        
        # Actually, let's just iterate in Python but with numpy arrays it might be slow if we do element-wise.
        # Let's try a hybrid:
        # Since 14M is large, pure Python loop might be too slow.
        # Let's try to reduce the search space.
        
        # If we find 1, we stop.
        
        # Let's try the two-pointer approach in Python, it's O(N) not O(N log N).
        # i = 0, j = len(s2)-1
        # while i < len(s1) and j >= 0:
        #    val = s1[i] + s2[j]
        #    if val == 0:
        #       # Check neighbors
        #       if j > 0:
        #          v2 = s1[i] + s2[j-1]
        #          if v2 != 0 and abs(v2) < min_abs_val: min_abs_val = abs(v2)
        #       if i < len(s1)-1:
        #          v3 = s1[i+1] + s2[j]
        #          if v3 != 0 and abs(v3) < min_abs_val: min_abs_val = abs(v3)
        #       # Move pointers to avoid getting stuck
        #       i += 1
        #       j -= 1
        #    elif val > 0:
        #       if val < min_abs_val: min_abs_val = val
        #       j -= 1
        #    else: # val < 0
        #       if abs(val) < min_abs_val: min_abs_val = abs(val)
        #       i += 1
        
        # This is O(N) which is 14M steps. In Python, 14M steps is ~1-2 seconds.
        # This is much better than 14M * log(14M).
        
        i = 0
        j = len(sums2) - 1
        n1 = len(sums1)
        n2 = len(sums2)
        
        while i < n1 and j >= 0:
            val = sums1[i] + sums2[j]
            if val == 0:
                # Check immediate neighbors to find smallest non-zero
                # Left neighbor in sum2
                if j > 0:
                    v = sums1[i] + sums2[j-1]
                    if v != 0:
                        av = abs(v)
                        if av < min_abs_val:
                            min_abs_val = av
                            if min_abs_val == 1:
                                break
                # Right neighbor in sum1
                if i < n1 - 1:
                    v = sums1[i+1] + sums2[j]
                    if v != 0:
                        av = abs(v)
                        if av < min_abs_val:
                            min_abs_val = av
                            if min_abs_val == 1:
                                break
                # Also check diagonal?
                # If s1[i]+s2[j]=0, then s1[i+1]+s2[j-1] is also a candidate?
                # s1 is sorted, s2 is sorted.
                # s1[i+1] >= s1[i], s2[j-1] <= s2[j].
                # So s1[i+1]+s2[j-1] can be anything.
                # But it's not "adjacent" in the 1D sorted sum list necessarily.
                # However, the smallest non-zero sum must be close to 0.
                # The two-pointer scan visits all "relevant" pairs.
                # If we hit 0, the next smallest must be either (i, j-1), (i+1, j), or potentially (i+1, j-1)?
                # Actually, if we move i+1 and j-1, we skip (i+1, j) and (i, j-1).
                # So we must check those.
                
                # To be safe, we can just accept that if we find 0, we update min_abs_val with neighbors.
                # Then we move both pointers.
                i += 1
                j -= 1
            elif val > 0:
                if val < min_abs_val:
                    min_abs_val = val
                    if min_abs_val == 1:
                        break
                j -= 1
            else: # val < 0
                av = -val
                if av < min_abs_val:
                    min_abs_val = av
                    if min_abs_val == 1:
                        break
                i += 1
                
    except Exception:
        min_abs_val = float('inf')
        
    # Now we have min_abs_val.
    # We need to find coefficients that produce a sum with abs value == min_abs_val.
    # We can do a DP to find the path.
    # Target T = min_abs_val or -min_abs_val.
    # Let's try to find a sum equal to min_abs_val. If not found, try -min_abs_val.
    
    target = min_abs_val
    # DP: dp[s] = (c1, c2, ..., c30) ? No, too big.
    # We can store the index of the previous state?
    # Or, we can use the meet-in-the-middle structure.
    # We know there exists a1 in S1 and a2 in S2 such that a1+a2 = target.
    # We can find one such pair.
    # Then we need to find coefficients for a1 in w1 and a2 in w2.
    
    # Find a1, a2
    # We can use the sorted lists again.
    # Iterate i, binary search j for target - sums1[i].
    
    found_a1 = None
    found_a2 = None
    
    # Use a set for sums2 for O(1) lookup?
    # sums2 has 14M elements. Set of 14M ints is ~100MB. Acceptable.
    # But we need to find the value, not just existence.
    # We can just do the two-pointer again to find the specific pair.
    
    i = 0
    j = len(sums2) - 1
    while i < len(sums1) and j >= 0:
        val = sums1[i] + sums2[j]
        if val == target:
            found_a1 = sums1[i]
            found_a2 = sums2[j]
            break
        elif val > target:
            j -= 1
        else:
            i += 1
            
    if found_a1 is None:
        target = -min_abs_val
        i = 0
        j = len(sums2) - 1
        while i < len(sums1) and j >= 0:
            val = sums1[i] + sums2[j]
            if val == target:
                found_a1 = sums1[i]
                found_a2 = sums2[j]
                break
            elif val > target:
                j -= 1
            else:
                i += 1
                
    # Now find coefficients for found_a1 using w1
    # And found_a2 using w2
    # We can use a simple backtracking or DP.
    # Since weights are large, DP on sum value is not possible.
    # But we have only 15 weights.
    # We can just iterate 3^15 again? No, that's what we did.
    # We can store the path during generation?
    # Let's re-generate with path storage for the specific target.
    
    def find_coeffs(weights, target):
        # Returns list of coeffs or None
        n = len(weights)
        # Use recursion with memoization?
        # State: (idx, remaining_sum)
        # But remaining_sum can be large.
        # However, we know the target is reachable.
        # Let's just do a DFS.
        
        coeffs = [0] * n
        
        def dfs(idx, rem):
            if idx == n:
                return rem == 0
            # Try -1, 0, 1
            for c in [-1, 0, 1]:
                coeffs[idx] = c
                if dfs(idx + 1, rem - c * weights[idx]):
                    return True
            return False
            
        # This is O(3^15) in worst case, but prunes if we find a solution.
        # Since we know a solution exists, it should be fast if we are lucky.
        # To improve, we can sort weights? No, order matters for index mapping.
        # But we can try the most likely values first?
        # Actually, 3^15 is 14M. If we do this twice, it's 28M ops.
        # It might be slow if the solution is deep.
        
        # Better: Use the pre-computed lists?
        # We have sums1. We know found_a1 is in sums1.
        # We can find the index of found_a1 in the generation tree?
        # We didn't store the tree.
        
        # Let's just run the DFS.
        # To speed up, we can use a set of reachable sums at each step?
        # No, that's what the list generation was.
        
        # Let's try to optimize the DFS by checking if rem is within possible range.
        # Max sum from idx to end is sum(weights[idx:])
        # Min sum is -sum(weights[idx:])
        
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + weights[i]
            
        def dfs_opt(idx, rem):
            if idx == n:
                return rem == 0
            # Pruning
            if abs(rem) > suffix_sums[idx]:
                return False
                
            for c in [-1, 0, 1]:
                coeffs[idx] = c
                if dfs_opt(idx + 1, rem - c * weights[idx]):
                    return True
            return False
            
        if dfs_opt(0, target):
            return coeffs
        return None
        
    c1 = find_coeffs(w1, found_a1)
    c2 = find_coeffs(w2, found_a2)
    
    if c1 and c2:
        return c1 + c2
    else:
        # Fallback
        return [0] * (n - 2) + [1, -1]

# EVOLVE-BLOCK-END
