import time
import math
import numpy as np

def construct():
    start_time = time.time()
    # Set a deadline 10 seconds before the 150s limit (evaluator kills at 150s)
    # Using 140s to be safe.
    deadline = start_time + 140.0

    # Strategy:
    # 1. Generate powerful numbers up to a limit L.
    # 2. Use NumPy broadcasting to compute all sums P_i + P_j.
    # 3. Count occurrences to find the N with the maximum number of representations.
    # 4. Extract the pairs for that N.
    
    # Choose L such that K (number of powerful numbers) is manageable for O(K^2) memory/time.
    # K ~ sqrt(L) + L^(1/3).
    # For L = 10^8, K ~ 10^4 + 464 ~ 10.5k. K^2 ~ 1.1 * 10^8.
    # Memory for int64 array of 1.1 * 10^8 is ~ 880 MB. This is risky but often works on modern systems.
    # Let's try L = 5 * 10^7 to be safer on memory, giving K ~ 7k, K^2 ~ 5*10^7 (400 MB).
    # Or L = 10^8 if we block the computation.
    # The inspiration 1 used L=10^8 and scored 5.93. Let's try to replicate that success.
    # To avoid memory errors, we can compute the sums in blocks if necessary, but np.unique is the bottleneck.
    # Let's try L = 10^8 first. If it fails, we'd need a fallback, but we can't easily fallback in a single run.
    # Let's use L = 8 * 10^7 to be slightly safer.
    
    limit = 80000000  # 8 * 10^7
    
    # Generate powerful numbers
    pows = set([1])
    max_i = math.isqrt(limit)
    max_j_global = int(round(limit ** (1/3))) + 2
    cubes = [j ** 3 for j in range(1, max_j_global + 1)]
    
    # Efficient generation
    for i in range(1, max_i + 1):
        sq = i * i
        if sq > limit:
            break
        max_j = int(round((limit // sq) ** (1/3)))
        if max_j > max_j_global:
            max_j = max_j_global
        
        for j in range(1, max_j + 1):
            val = sq * cubes[j-1]
            if val <= limit:
                pows.add(val)
            else:
                break
                
        if time.time() > deadline - 30:
            break

    P_list = sorted(pows)
    K = len(P_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Convert to NumPy array
    P = np.array(P_list, dtype=np.int64)
    
    # We want to find N that maximizes the number of pairs (a, b) with a <= b and a + b = N.
    # This is equivalent to finding the mode of the sums P_i + P_j, adjusted for the a<=b constraint.
    # If we compute the full KxK matrix of sums, the count of N in the full matrix is C_full.
    # The number of pairs with a <= b is (C_full + D) // 2, where D is 1 if N/2 is in P, else 0.
    # Since D is at most 1, maximizing C_full is a very good proxy for maximizing the pair count.
    # Specifically, if two sums N1 and N2 have C_full(N1) > C_full(N2) + 1, then N1 definitely has more pairs.
    # If they are close, we might need to check D. But usually the top C_full is significantly larger.
    
    # Compute sums in blocks to save memory?
    # np.unique on a 1D array is faster than on a 2D array? No, it flattens it.
    # Let's try to compute the full matrix.
    
    # Check if we have enough time
    if time.time() > deadline - 20:
        # Fallback to a smaller limit or simple heuristic
        # For now, just return what we have if we can't do the heavy lift
        # Let's try a smaller limit just in case
        pass 

    # Compute all pairwise sums
    # P[:, None] + P[None, :] creates a K x K matrix
    # This can be memory intensive.
    # K ~ 10000 -> 100M elements -> 800MB.
    # Let's try it.
    
    try:
        sums = P[:, None] + P[None, :]
        # Flatten and find unique counts
        # np.unique is O(N log N) where N is the number of elements (K^2)
        unique_sums, counts = np.unique(sums, return_counts=True)
        
        # Find the sum with the maximum count
        max_c = np.max(counts)
        idx = np.argmax(counts)
        best_sum = unique_sums[idx]
        
        # Verify and extract pairs
        pows_set = set(P_list)
        pairs = []
        for a in P_list:
            if a > best_sum:
                break
            b = best_sum - a
            if b < a:
                continue
            if b in pows_set:
                pairs.append([a, b])
        
        if not pairs:
            return {'N': 2, 'pairs': [[1, 1]]}
            
        return {'N': int(best_sum), 'pairs': pairs}
        
    except MemoryError:
        # Fallback: Use a smaller limit or block processing
        # Let's reduce limit and retry
        limit = 20000000
        pows = set([1])
        max_i = math.isqrt(limit)
        max_j_global = int(round(limit ** (1/3))) + 2
        cubes = [j ** 3 for j in range(1, max_j_global + 1)]
        
        for i in range(1, max_i + 1):
            sq = i * i
            if sq > limit:
                break
            max_j = int(round((limit // sq) ** (1/3)))
            if max_j > max_j_global:
                max_j = max_j_global
            for j in range(1, max_j + 1):
                val = sq * cubes[j-1]
                if val <= limit:
                    pows.add(val)
                else:
                    break
            if time.time() > deadline - 10:
                break

        P_list = sorted(pows)
        K = len(P_list)
        P = np.array(P_list, dtype=np.int64)
        
        sums = P[:, None] + P[None, :]
        unique_sums, counts = np.unique(sums, return_counts=True)
        max_c = np.max(counts)
        idx = np.argmax(counts)
        best_sum = unique_sums[idx]
        
        pows_set = set(P_list)
        pairs = []
        for a in P_list:
            if a > best_sum:
                break
            b = best_sum - a
            if b < a:
                continue
            if b in pows_set:
                pairs.append([a, b])
                
        if not pairs:
            return {'N': 2, 'pairs': [[1, 1]]}
            
        return {'N': int(best_sum), 'pairs': pairs}

