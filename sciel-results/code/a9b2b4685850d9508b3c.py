"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 24
    elements = list(range(1, N + 1))
    
    # LCM of 1..24
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    # Precompute 1/x as L/x
    recip = [0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = L // i
        
    # Find all minimal conflict sets
    # A conflict for x is a subset T of {1..N}\{x} such that sum(1/y for y in T) == 1/x
    # We want minimal T.
    # C[x] will be a list of bitmasks representing T
    
    C = {i: [] for i in range(1, N + 1)}
    
    # To find minimal subsets T summing to 1/x, we can iterate over all subsets?
    # 2^23 is too big.
    # But note that 1/x is fixed.
    # We can use a recursive search to find subsets of {1..N}\{x} summing to recip[x].
    # Since we want minimal, we can stop exploring a branch if the current sum >= target? No, we need exact sum.
    # We can use a meet-in-the-middle or just simple DFS with pruning.
    # Given N=24, and the values are distinct integers L/y.
    # The target is L/x.
    # The values L/y are roughly L/N to L.
    # Sum of a few large values can reach the target.
    
    # Let's find all subsets T (as bitmasks) that sum to recip[x].
    # We only need minimal ones.
    # If T1 is a subset of T2 and both sum to recip[x], then T2 is not minimal.
    # So we just need to find all subsets summing to recip[x], and then filter out non-minimal ones.
    
    # Optimization: Sort elements by value descending?
    # Actually, let's just do a DFS for each x.
    # The search space is 2^(N-1). For N=24, that's 8M.
    # Doing this for 24 x's is 200M operations. Might be too slow in Python.
    
    # Alternative: Precompute all subset sums of the full set? No.
    # Let's try to optimize the search for T.
    # Note that L/y decreases as y increases.
    # We can prune if the sum of remaining elements cannot reach the target.
    
    def find_conflicts_for_x(x):
        target = recip[x]
        # Available elements are all except x
        # We want subsets T of available elements summing to target.
        # Let's use a set to store found masks.
        found_masks = set()
        
        # Sort available elements by value (recip[y]) descending to prune faster?
        # Or ascending?
        # Let's use a recursive function.
        # idx: index in sorted list of available elements
        # curr_sum: current sum
        # curr_mask: current bitmask
        
        # To make it faster, we can precompute suffix sums of reciprocals.
        # But the set of available elements depends on x.
        
        # Let's just iterate. N is small.
        # We'll use a list of (value, mask_bit)
        # We can use a dictionary or set for memoization? No, sum is the key.
        
        # Let's use a simple DFS with pruning.
        # We'll sort the available elements by value descending.
        avail = [y for y in range(1, N + 1) if y != x]
        # Sort by recip[y] descending
        avail.sort(key=lambda y: recip[y], reverse=True)
        
        suffix_sum = [0] * (len(avail) + 1)
        for i in range(len(avail) - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + recip[avail[i]]
            
        def dfs(idx, curr_sum, curr_mask):
            if time.time() > deadline:
                return
            if curr_sum == target:
                found_masks.add(curr_mask)
                return
            if idx == len(avail):
                return
            if curr_sum > target:
                return
            if curr_sum + suffix_sum[idx] < target:
                return
                
            # Include avail[idx]
            y = avail[idx]
            dfs(idx + 1, curr_sum + recip[y], curr_mask | (1 << (y - 1)))
            
            # Exclude avail[idx]
            dfs(idx + 1, curr_sum, curr_mask)
            
        dfs(0, 0, 0)
        
        # Filter minimal masks
        # A mask m1 is non-minimal if there exists m2 in found_masks such that m2 is a proper subset of m1
        # and sum(m2) == target (which is true by construction).
        # Wait, if m2 is a subset of m1 and both sum to target, then the elements in m1 \ m2 must sum to 0, which is impossible for positive reciprocals.
        # So all found masks are minimal?
        # Yes, because all reciprocals are positive. If T1 is a subset of T2, sum(T1) < sum(T2).
        # So if sum(T1) = sum(T2) = target, then T1 = T2.
        # Thus, all subsets found are minimal.
        
        return list(found_masks)

    # Precompute C[x]
    for x in range(1, N + 1):
        C[x] = find_conflicts_for_x(x)
        
    # Precompute Required[j][i]
    # R[j][i] is a list of masks M such that M U {i} is a conflict for j.
    # i.e., M is a subset of {1..N}\{j,i} such that sum(M) + 1/i = 1/j.
    # sum(M) = 1/j - 1/i = (i-j)/(ij).
    # We can just check if (1 << (i-1)) is in a conflict mask T for j, then T \ {i} is the required mask.
    
    R = {j: {i: [] for i in range(1, N + 1) if i != j} for j in range(1, N + 1)}
    
    for j in range(1, N + 1):
        for i in range(1, N + 1):
            if i == j:
                continue
            bit_i = 1 << (i - 1)
            for mask in C[j]:
                if mask & bit_i:
                    # i is in the conflict set T
                    # The rest is T \ {i}
                    rest_mask = mask ^ bit_i
                    R[j][i].append(rest_mask)
                    
    # Now perform Branch and Bound
    # We want to maximize the size of the selected set.
    # We iterate elements from 1 to N.
    # State: index, current_mask, current_size
    
    best_size = 0
    best_mask = 0
    
    # Order elements: maybe larger elements are safer?
    # Let's try 1 to N.
    
    def backtrack(idx, curr_mask, curr_size):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if curr_size > best_size:
            best_size = curr_size
            best_mask = curr_mask
            
        # Pruning: even if we take all remaining, can we beat best?
        remaining = N - idx
        if curr_size + remaining <= best_size:
            return
            
        if idx == N:
            return
            
        # Option 1: Exclude element idx
        backtrack(idx + 1, curr_mask, curr_size)
        
        # Option 2: Include element idx
        x = idx
        bit_x = 1 << (x - 1)
        new_mask = curr_mask | bit_x
        
        # Check validity of adding x
        # 1. Check if curr_mask contains any conflict for x
        valid = True
        for mask in C[x]:
            if (curr_mask & mask) == mask:
                valid = False
                break
        
        if valid:
            # 2. Check if adding x creates a conflict for any existing element j in curr_mask
            # For each j in curr_mask, check if curr_mask (without x) contains any required mask for (j, x)
            # Note: The required mask is a subset of curr_mask \ {x}
            # curr_mask does not contain x yet, so we just check against curr_mask
            existing_elements = [j for j in range(1, N + 1) if (curr_mask & (1 << (j - 1)))]
            for j in existing_elements:
                for req_mask in R[j][x]:
                    if (curr_mask & req_mask) == req_mask:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            backtrack(idx + 1, new_mask, curr_size + 1)
            
    backtrack(0, 0, 0)
    
    # Convert best_mask to list
    result = [i for i in range(1, N + 1) if (best_mask & (1 << (i - 1)))]
    
    return sorted(result)

# EVOLVE-BLOCK-END
