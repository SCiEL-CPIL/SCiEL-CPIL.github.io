import time
from math import lcm
import bisect
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L // d
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2  # 23
    half1 = list(range(mid))
    half2 = list(range(mid, n))
    
    w1 = [weights[i] for i in half1]
    w2 = [weights[i] for i in half2]
    
    def gen_sums(weights_list):
        """
        Generate all subset sums < L with their masks.
        Returns list of (sum, mask).
        """
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            if time.time() > deadline:
                break
            bit = 1 << i
            current_len = len(sums)
            new_sums = []
            for s, m in sums:
                ns = s + w
                if ns < L:
                    new_sums.append((ns, m | bit))
            sums.extend(new_sums)
        return sums

    print("Generating sums for half 1...")
    sums1 = gen_sums(w1)
    print(f"Half 1 size: {len(sums1)}")
    
    if time.time() > deadline:
        # Fallback if generation takes too long
        return [0] 

    print("Generating sums for half 2...")
    sums2 = gen_sums(w2)
    print(f"Half 2 size: {len(sums2)}")
    
    if time.time() > deadline:
        # Try to use what we have? Or fallback.
        # Let's try to sort and search with partial data if possible, 
        # but usually if gen fails, we are in trouble.
        # However, gen is usually fast for N=24.
        pass

    # Sort sums2 by sum
    sums2.sort(key=lambda x: x[0])
    sums2_vals = [s[0] for s in sums2]
    
    best_sum = -1
    best_m1 = 0
    best_m2 = 0
    
    # Iterate through sums1
    for s1, m1 in sums1:
        if time.time() > deadline:
            break
            
        # We need s2 such that s1 + s2 < L
        # s2 <= L - 1 - s1
        target = L - 1 - s1
        if target < 0:
            continue
            
        # Find the largest s2 <= target
        idx = bisect.bisect_right(sums2_vals, target)
        if idx > 0:
            s2, m2 = sums2[idx - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2
                
    # Reconstruct indices
    if best_sum == -1:
        # Fallback: pick a single small denominator
        # Max weight is L/2. L/2 < L. So [0] is valid.
        best_indices = [0]
    else:
        best_indices = []
        for i in range(mid):
            if best_m1 & (1 << i):
                best_indices.append(i)
        for i in range(mid, n):
            if best_m2 & (1 << (i - mid)):
                best_indices.append(i)
                
    # Validation
    if not best_indices:
        best_indices = [0]
        
    # Verify
    s = sum(weights[i] for i in best_indices)
    if s >= L:
        # This shouldn't happen, but if it does, remove one item
        # This is a safeguard
        best_indices.pop()
        
    print(f"Best sum: {best_sum}/{L}")
    print(f"Indices: {best_indices}")
    
    return best_indices
