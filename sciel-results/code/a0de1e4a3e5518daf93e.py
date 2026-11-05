import time
from math import gcd
import bisect

def construct():
    start_time = time.time()
    deadline = 85  # 85 seconds, 5s safety margin
    
    n = 30
    
    # Compute L = lcm(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n+1):
        L = lcm(L, i)
    
    # a[i] = L // i for i in 1..30
    # 0-indexed: a[i] corresponds to 1/(i+1)
    a = [L // (i+1) for i in range(n)]
    
    mid = 15
    left_A = a[:mid]
    right_A = a[mid:]
    
    # Precompute powers of 3 for base-3 encoding
    pow3_left = [1]
    for i in range(mid - 1):
        pow3_left.append(pow3_left[-1] * 3)
        
    pow3_right = [1]
    for i in range(n - mid - 1):
        pow3_right.append(pow3_right[-1] * 3)

    def generate_sums(A_part, pow3):
        # Returns list of (sum, code)
        # Start with 0
        current = [(0, 0)]
        for i, val in enumerate(A_part):
            p = pow3[i]
            new_current = []
            for s, c in current:
                # c = -1
                new_current.append((s - val, c + (-1) * p))
                # c = 0
                new_current.append((s, c))
                # c = 1
                new_current.append((s + val, c + (1) * p))
            current = new_current
            # Time check
            if time.time() - start_time > deadline:
                return None
        return current

    # Generate left sums
    left_data = generate_sums(left_A, pow3_left)
    if left_data is None:
        return [0] * 28 + [1, -1]
    
    # Build dict: sum -> code
    # If multiple codes produce same sum, keep one
    left_dict = {}
    for s, c in left_data:
        if s not in left_dict:
            left_dict[s] = c
            
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]
        
    # Sort keys for binary search
    sorted_left_sums = sorted(left_dict.keys())
    
    # Generate right sums
    right_data = generate_sums(right_A, pow3_right)
    if right_data is None:
        return [0] * 28 + [1, -1]
        
    best_abs = float('inf')
    best_code_L = None
    best_code_R = None
    
    # Iterate right sums and find best left match
    for s_r, c_r in right_data:
        if time.time() - start_time > deadline:
            break
            
        target = -s_r
        # Find insertion point in sorted_left_sums
        idx = bisect.bisect_left(sorted_left_sums, target)
        
        # Check idx, idx-1, idx+1
        candidates = [idx - 1, idx, idx + 1]
        
        for ci in candidates:
            if 0 <= ci < len(sorted_left_sums):
                s_l = sorted_left_sums[ci]
                total_sum = s_l + s_r
                
                if total_sum != 0:
                    abs_total = abs(total_sum)
                    if abs_total < best_abs:
                        best_abs = abs_total
                        best_code_L = left_dict[s_l]
                        best_code_R = c_r
                        
                        # If we found 1, it's the minimum possible non-zero integer sum
                        # (since sums are integers)
                        if best_abs == 1:
                            # We can't do better than 1
                            # But we should still check if we can break early?
                            # No, we need to finish to ensure we didn't miss something? 
                            # Actually 1 is the minimum possible non-zero absolute integer value.
                            # So we can stop searching immediately.
                            # But we need to decode.
                            pass 
                            # We can break out of loops if we want to save time, 
                            # but we must ensure we have the codes.
                            # Since 1 is min, we can break.
                            # However, the loop structure makes it hard to break cleanly from both.
                            # We'll just let it run or add a flag.
                            # For now, just update.
        
    if best_code_L is None:
        # Fallback
        return [0] * 28 + [1, -1]
        
    # Decode coefficients
    def decode_code(code, pow3_list):
        coeffs = []
        for i, p in enumerate(pow3_list):
            # Extract digit
            # code is sum of c_i * 3^i
            # c_i = (code // 3^i) % 3
            # But our encoding was: new_code = old_code + val * 3^i
            # where val is -1, 0, 1.
            # So the digit is (code // 3^i) % 3.
            # However, negative numbers in Python modulo behave as expected for extraction?
            # e.g. -1 % 3 = 2. But we stored -1 * 3^i.
            # So we need to handle negative codes carefully.
            # Actually, `code` can be negative.
            # Let's just reconstruct by dividing.
            # A safer way:
            # We know the range.
            # Let's just use the property:
            # digit = (code // p) % 3
            # If code is negative, this might be tricky.
            # Example: code = -1 (which is -1 * 3^0).
            # -1 // 1 = -1. -1 % 3 = 2. This is wrong, we want -1.
            # So we should use:
            # digit = (code // p) % 3
            # if digit > 1: digit -= 3
            
            d = (code // p) % 3
            if d > 1:
                d -= 3
            coeffs.append(d)
        return coeffs

    left_coeffs = decode_code(best_code_L, pow3_left)
    right_coeffs = decode_code(best_code_R, pow3_right)
    
    result = left_coeffs + right_coeffs
    
    # Verify result
    # sum(c[i] / (i+1))
    val = 0
    for i in range(n):
        val += result[i] / (i + 1)
        
    # Check if non-zero
    if val == 0:
        # This shouldn't happen if we checked total_sum != 0
        # But floating point might be an issue?
        # We used integer sums, so total_sum != 0 implies val != 0.
        pass

    return result
