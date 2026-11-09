import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Finds 30 integer coefficients from {-1, 0, 1} such that the signed sum
    sum(c[i-1]/i) is non-zero and its absolute value is minimized.
    
    Strategy:
    1. Scale the problem by the LCM of 1..30 to convert to integer sums.
    2. Use Meet-in-the-Middle to split the 30 coefficients into two halves of 15.
    3. Enumerate all 3^15 = 14,348,907 possible sums for the first half, storing (sum, coeffs).
    4. Sort the first half's sums.
    5. Enumerate the second half's sums. For each sum S2, binary search in the sorted first half
       for the value closest to -S2 (to minimize |S1 + S2|) and closest to 0 (if S2 is 0).
       Actually, we just need to find S1 such that |S1 + S2| is minimized and non-zero.
       Since the list is sorted, for a target T = -S2, the best candidates are the largest
       value <= T and the smallest value >= T. We check these and also check 0 if T is not 0?
       No, simply checking the neighbors of -S2 in the sorted list is sufficient to find the
       minimum |S1 - (-S2)|.
       However, we must ensure the total sum is not 0.
       If the closest neighbor results in sum 0, we must check the next closest neighbor.
    6. Track the global minimum non-zero absolute sum and the corresponding coefficients.
    7. Return the coefficients.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s limit
    
    n = 30
    # Calculate LCM of 1..30
    # 1..30 LCM is 2329089562800
    # Let's compute it dynamically to be safe, though it's a constant.
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Precompute scaled weights: w[i] = L // (i+1) for i in 0..29
    # Sum is sum(c[i] * L/i+1). Let S = sum(c[i] * w[i]).
    # We want to minimize |S| such that S != 0.
    
    weights = [L // (i + 1) for i in range(n)]
    
    half = n // 2
    # Split weights into two halves
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Enumerate first half
    # 3^15 = 14348907. This is manageable in memory and time in Python if optimized.
    # We will generate a list of tuples (sum_val, coeff_tuple)
    
    def generate_sums(weights_half):
        sums = []
        k = len(weights_half)
        # Recursive generation might hit recursion limit or be slow.
        # Iterative base-3 counting is better.
        # Total combinations: 3**k
        # We can iterate from 0 to 3**k - 1 and interpret as base 3 digits.
        # Digits: 0 -> coeff -1, 1 -> coeff 0, 2 -> coeff 1?
        # Or 0 -> -1, 1 -> 0, 2 -> 1.
        # Let's use 0, 1, 2 mapping to -1, 0, 1.
        
        # Optimization: Precompute powers of 3? No, just loop.
        # 14 million iterations in Python might take a few seconds.
        # Let's try to be efficient.
        
        # Alternative: Recursive with tail recursion optimization? Python doesn't do that well.
        # Let's use a simple iterative approach with a stack or just nested loops if k is small?
        # k=15 is too large for nested loops.
        
        # We can use a list and extend it.
        # Start with [(0, ())]
        # For each weight, extend the list.
        # This creates 3 * 3^(i-1) items. Total memory usage is sum of sizes, which is fine.
        # But we only need the final list.
        
        current_sums = [(0, ())]
        for i, w in enumerate(weights_half):
            new_sums = []
            # For each existing sum, add -w, 0, +w
            # To keep track of coefficients, we append the choice.
            # Storing tuples of length 15 for 14M items is memory heavy (14M * 15 * 8 bytes ~ 1.7GB?).
            # This might be too much memory.
            
            # Better approach: Store only the sum value and an index into a precomputed table?
            # Or just store the sum value, and reconstruct coefficients later?
            # We need to return the coefficients.
            # We can store the sum value and a 15-bit integer representing the coefficients.
            # 0 -> -1, 1 -> 0, 2 -> 1.
            # We can pack 15 coefficients into a single integer?
            # 3^15 fits in 64 bits? 3^15 = 14,348,907. Yes, fits in 24 bits easily.
            # Let's store (sum_val, coeff_code).
            
            # coeff_code: base 3 representation of the choices.
            # If we have choices c0, c1, ..., c14.
            # code = c0 + c1*3 + c2*9 + ...
            
            # Let's try this.
            new_sums = []
            # We can't just extend in place easily without doubling memory temporarily.
            # But 14M items * (8 bytes int + 8 bytes int) = 224 MB. This is acceptable.
            
            for s, code in current_sums:
                # Choice 0: coeff -1
                new_sums.append((s - w, code))
                # Choice 1: coeff 0
                new_sums.append((s, code + 1 * (3 ** i)))
                # Choice 2: coeff 1
                new_sums.append((s + w, code + 2 * (3 ** i)))
            current_sums = new_sums
            
            if time.time() > deadline - 5:
                # Fallback if it takes too long
                return None, None
                
        return current_sums, None

    # Generate first half sums
    sums1, _ = generate_sums(w1)
    
    if sums1 is None:
        # Fallback to simple heuristic if MITM fails or times out
        return _fallback_heuristic(weights, deadline)
        
    # Sort by sum value
    # We need to find, for any target T, the element in sums1 closest to T.
    # We just need the sum values for binary search, but we also need the code to reconstruct.
    # So we sort the list of (sum_val, code).
    sums1.sort(key=lambda x: x[0])
    
    # Extract sum values for binary search
    sum_vals1 = [s for s, c in sums1]
    
    best_abs_sum = float('inf')
    best_code1 = 0
    best_code2 = 0
    
    # Enumerate second half
    # We don't need to store all sums2, we can iterate and check on the fly.
    # But generate_sums returns a list. Let's modify generate_sums to yield?
    # Or just use the list. 14M items is fine.
    sums2, _ = generate_sums(w2)
    
    if sums2 is None:
        return _fallback_heuristic(weights, deadline)
        
    import bisect
    
    for s2, code2 in sums2:
        if time.time() > deadline - 2:
            break
            
        # We want s1 such that |s1 + s2| is minimized and non-zero.
        # Target s1 = -s2.
        # Find insertion point of -s2 in sum_vals1.
        target = -s2
        idx = bisect.bisect_left(sum_vals1, target)
        
        # Candidates: idx-1, idx
        # Also, if the candidate sum is 0, we might need to check idx-2 or idx+1.
        
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(sum_vals1):
            candidates.append(idx)
            
        # Check these candidates
        for c_idx in candidates:
            s1, c1_code = sums1[c_idx]
            total_sum = s1 + s2
            if total_sum == 0:
                continue # Invalid
            abs_sum = abs(total_sum)
            if abs_sum < best_abs_sum:
                best_abs_sum = abs_sum
                best_code1 = c1_code
                best_code2 = code2
                
        # Optimization: If best_abs_sum becomes 1 (scaled), we can't do better (except 0 which is invalid).
        # But wait, the LCM is huge. The smallest non-zero integer sum is 1.
        # If we find 1, we can stop?
        # Yes, because sums are integers. The smallest non-zero absolute value is 1.
        if best_abs_sum == 1:
            break
            
    if best_abs_sum == float('inf'):
        # Should not happen, but fallback
        return _fallback_heuristic(weights, deadline)
        
    # Reconstruct coefficients from codes
    def decode_code(code, length):
        coeffs = []
        for i in range(length):
            digit = code % 3
            code //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs
    
    coeffs1 = decode_code(best_code1, half)
    coeffs2 = decode_code(best_code2, n - half)
    
    final_coeffs = coeffs1 + coeffs2
    
    # Verify
    # s = sum(c[i] / (i+1) for i in range(n))
    # It should be non-zero.
    
    return final_coeffs

def _fallback_heuristic(weights, deadline):
    import random
    from fractions import Fraction
    
    n = 30
    reciprocals = [Fraction(1, i) for i in range(1, n + 1)]
    
    best_score = None
    best_coeffs = None
    
    def get_score(coeffs):
        s = Fraction(0, 1)
        for c, r in zip(coeffs, reciprocals):
            if c != 0:
                s += c * r
        if s == 0:
            return None
        return abs(s)
        
    # Simple random restarts
    while time.time() < deadline - 5:
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        if all(c == 0 for c in coeffs):
            coeffs[0] = 1
        score = get_score(coeffs)
        if score is not None:
            if best_score is None or score < best_score:
                best_score = score
                best_coeffs = coeffs[:]
                
    if best_coeffs:
        return best_coeffs
    return [0] * 28 + [1, -1]
# EVOLVE-BLOCK-END
