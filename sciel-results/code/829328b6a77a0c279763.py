import time
import math
from itertools import product
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, aim for 80s
    deadline = start_time + 80.0
    
    n = 30
    # Calculate LCM of 1..30
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w[i] = L / (i+1) for i in 0..29
    weights = [L // (i + 1) for i in range(n)]
    
    # Split into two halves
    half = n // 2 # 15
    
    # Generate all possible sums and coefficients for first half
    # coeffs are in {-1, 0, 1}
    # We store (sum, coeffs_tuple)
    
    def generate_sums(weights_part):
        sums = []
        # 3^15 is ~14M. This loop might take a few seconds in pure Python.
        # To speed up, we can use recursion or iterative product.
        # product(range(-1, 2), repeat=len(weights_part))
        
        # Optimization: Pre-compute partial sums? 
        # Actually, 14M iterations in Python is ~2-5 seconds.
        # We need to be careful with memory and speed.
        
        # Let's try a recursive generation to avoid creating huge intermediate lists if possible,
        # but we need the list for sorting/binary search anyway.
        
        # Alternative: Use numpy? 
        # np.array(list(product([-1,0,1], repeat=15))) -> 14M rows, 15 cols. 
        # 14M * 15 * 4 bytes = ~800MB. Might be too much memory or slow to generate.
        # Pure python list of tuples is also heavy.
        
        # Let's stick to pure python but optimize the loop.
        # Actually, we can generate sums more efficiently.
        
        # We will generate the list of (sum, coeffs)
        # coeffs is a tuple of 15 integers.
        
        # To speed up, we can just store the sum and the coeffs as a small tuple.
        
        items = []
        # Using product is standard
        for coeffs in product([-1, 0, 1], repeat=len(weights_part)):
            s = 0
            # Unroll loop? No, 15 is small.
            for c, w in zip(coeffs, weights_part):
                if c != 0:
                    s += c * w
            items.append((s, coeffs))
            
        return items

    # Check time before generating
    if time.time() > deadline - 10:
        # Fallback to a simple heuristic if we are already late (unlikely)
        return [0]*28 + [1, -1]

    # Generate first half
    sums1 = generate_sums(weights[:half])
    
    # Generate second half
    sums2 = generate_sums(weights[half:])
    
    # Sort second half by sum
    # We need to sort by the sum value
    sums2.sort(key=lambda x: x[0])
    
    # Extract just the sums for binary search
    sums2_vals = [x[0] for x in sums2]
    
    best_abs = None
    best_coeffs = None
    
    # Iterate through first half
    for s1, coeffs1 in sums1:
        if time.time() > deadline - 5:
            break
            
        # We want s1 + s2 to be close to 0 but not 0.
        # So we look for -s1 in sums2_vals.
        
        # Find insertion point for -s1
        target = -s1
        idx = bisect_left(sums2_vals, target)
        
        # Candidates to check: idx, idx-1, idx+1, idx-2, idx+2
        # We check a small window around the target to find the smallest non-zero sum.
        
        # The sorted list is sums2_vals.
        # We need to check indices in sums2 corresponding to these positions.
        
        candidates = []
        for k in range(idx - 2, idx + 3):
            if 0 <= k < len(sums2_vals):
                s2_val = sums2_vals[k]
                total = s1 + s2_val
                if total == 0:
                    continue
                abs_val = abs(total)
                if best_abs is None or abs_val < best_abs:
                    best_abs = abs_val
                    # Retrieve coeffs2
                    coeffs2 = sums2[k][1]
                    best_coeffs = coeffs1 + coeffs2
                    
                    # If we found 1, it's the absolute minimum possible (since integer sum)
                    # Wait, the sum is integer N. The minimum non-zero absolute value is 1.
                    # If we find 1, we can stop early? 
                    # Yes, 1 is the smallest possible non-zero integer.
                    # However, we must be careful: is 1 achievable?
                    # If we find 1, we can break out of the outer loop too.
                    if best_abs == 1:
                        # Found the theoretical minimum
                        # We can return immediately? 
                        # We need to construct the final list.
                        # But we are in the middle of the loop.
                        # Let's set a flag or just break.
                        pass
        
        if best_abs == 1:
            # We found the minimum possible non-zero integer sum.
            # We can stop searching.
            break
            
    if best_coeffs is None:
        # Fallback
        return [0]*28 + [1, -1]
        
    return list(best_coeffs)

# EVOLVE-BLOCK-END
