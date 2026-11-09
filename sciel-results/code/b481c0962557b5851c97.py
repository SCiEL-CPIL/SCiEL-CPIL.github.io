import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
    
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = L // i
        
    # Split into two halves
    half1 = list(range(1, 16))
    half2 = list(range(16, 31))
    
    # Generate all sums for first half
    def gen_sums_list(indices):
        current_sums = [0]
        for i in indices:
            a_i = A[i]
            # Using list comprehension for speed
            add = [s + a_i for s in current_sums]
            sub = [s - a_i for s in current_sums]
            current_sums = add + sub + current_sums
        return current_sums

    sums1 = gen_sums_list(half1)
    sums1.sort()
    
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    # Generate sums for second half iteratively
    # We use a recursive generator to avoid storing all sums2
    def gen_sums_gen(indices, idx=0, current_sum=0):
        if idx == len(indices):
            yield current_sum
            return
        a_i = A[indices[idx]]
        yield from gen_sums_gen(indices, idx + 1, current_sum + a_i)
        yield from gen_sums_gen(indices, idx + 1, current_sum)
        yield from gen_sums_gen(indices, idx + 1, current_sum - a_i)
        
    # Recursion depth 15 is fine.
    # But Python recursion overhead might be significant for 14M yields.
    # Alternative: iterative generation using a stack or just a list of partial sums.
    # Let's stick to the list method for half2 as well, but process and discard?
    # No, we can just generate the list for half2 too if memory allows.
    # 14M ints * 28 bytes = 400MB.
    # sums1 is 400MB. Total 800MB.
    # If memory is tight, we can use the generator.
    # Let's try the generator approach but optimized.
    
    # Actually, the list method for half2 is faster than generator.
    # Let's check memory.
    # If we assume 2GB limit, 800MB is safe.
    # If 1GB limit, it's risky.
    
    # Let's use the generator to be safe on memory, even if slightly slower.
    # Or, we can generate half2 sums in chunks?
    
    # Let's try the generator.
    for s2 in gen_sums_gen(half2):
        if time.time() > deadline - 5:
            break
        target = -s2
        pos = bisect.bisect_left(sums1, target)
        
        candidates = []
        if pos < len(sums1):
            candidates.append(sums1[pos])
        if pos > 0:
            candidates.append(sums1[pos-1])
            
        for s1 in candidates:
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs == 1:
                        return reconstruct(best_s1, best_s2, half1, half2, A)
                        
    if best_s1 is None:
        return [0]*28 + [1, -1]
        
    return reconstruct(best_s1, best_s2, half1, half2, A)

def reconstruct(s1, s2, half1, half2, A):
    def find_coeffs(target, indices):
        n = len(indices)
        coeffs = [0] * n
        
        # Sort indices by A[i] descending to prune faster
        # half1 and half2 are already 1..15 and 16..30, so A[i] is descending
        # So no need to sort again.
        
        def dfs(idx, current_sum):
            if idx == n:
                return current_sum == target
            
            a_i = A[indices[idx]]
            
            # Heuristic: try values that get closer to target
            # target - current_sum is the remaining sum needed
            rem = target - current_sum
            
            # Try 1, 0, -1
            # If rem > 0, 1 is good. If rem < 0, -1 is good.
            # Let's just try all 3, order might not matter much for depth 15.
            
            coeffs[idx] = 1
            if dfs(idx + 1, current_sum + a_i):
                return True
            
            coeffs[idx] = 0
            if dfs(idx + 1, current_sum):
                return True
            
            coeffs[idx] = -1
            if dfs(idx + 1, current_sum - a_i):
                return True
            
            return False
            
        if dfs(0, 0):
            return coeffs
        return None

    c1 = find_coeffs(s1, half1)
    c2 = find_coeffs(s2, half2)
    
    if c1 is None or c2 is None:
        return [0]*28 + [1, -1]
        
    result = [0] * 30
    for i, idx in enumerate(half1):
        result[idx-1] = c1[i]
    for i, idx in enumerate(half2):
        result[idx-1] = c2[i]
        
    return result
