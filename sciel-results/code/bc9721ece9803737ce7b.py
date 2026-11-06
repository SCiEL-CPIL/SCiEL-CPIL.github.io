"""Optimized constructor for distinct three-term unit-fraction reps of 4/n."""
import time
import math

DENOM_BOUND = 10**12

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin under 90s kill
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        x_min = n // 4 + 1
        found = None
        
        # Strategy: Try x values, for each x iterate y in bounded range
        # For x close to n/4, the y-range is large; use divisor method there
        # For x further out, y-range is small; direct iteration is fine
        
        for x_attempt in range(x_min, x_min + 300):
            if time.time() > deadline:
                break
            A = 4 * x_attempt - n
            if A <= 0:
                continue
            B = n * x_attempt
            
            # y must satisfy: B/A < y <= 2B/A and y > x_attempt (for x < y < z)
            y_lo = B // A + 1
            y_hi = (2 * B) // A
            if y_hi < y_lo:
                continue
            if y_lo <= x_attempt:
                y_lo = x_attempt + 1
            if y_lo > y_hi:
                continue
            
            range_len = y_hi - y_lo + 1
            
            if range_len <= 20000:
                # Direct iteration over y
                y_limit = min(y_hi + 1, y_lo + 20000)
                A_local = A
                B_local = B
                x_local = x_attempt
                for y in range(y_lo, y_limit):
                    if (y - y_lo) % 2000 == 0 and time.time() > deadline:
                        break
                    denom = A_local * y - B_local
                    if denom <= 0:
                        continue
                    if (B_local * y) % denom == 0:
                        z = (B_local * y) // denom
                        if z > y and z != x_local and z <= DENOM_BOUND:
                            if x_local != y and y != z and x_local != z:
                                found = [x_local, y, z]
                                break
            else:
                # Large range: use divisor-based approach
                # (A*y - B)(A*z - B) = B^2
                # Let u = A*y - B, v = A*z - B, u*v = B^2
                # u ≡ -B (mod A), v ≡ -B (mod A)
                # y = (u+B)/A, z = (v+B)/A
                # Need u > B (so y > x_attempt when u+B > A*x_attempt => u > A*x_attempt - B = 4*x_attempt^2 - n*x_attempt - n*x_attempt... 
                # Actually y > x_attempt means (u+B)/A > x_attempt => u > A*x_attempt - B = (4x-n)*x - n*x = 4x^2 - nx - nx = 4x^2 - 2nx
                # And u*v = B^2, so v = B^2/u
                # We need to find divisors u of B^2 such that u ≡ -B (mod A) and u > A*x_attempt - B
                
                # Instead, iterate over divisors of B^2 that are ≡ -B (mod A)
                # This is expensive for large B. Alternative: iterate u in a smarter way.
                # 
                # Actually, for the divisor approach to be fast, we'd need to factor B^2.
                # For n up to 256 and x up to x_min+300, B = n*x can be up to ~256*164 = 41984
                # B^2 ~ 1.76 * 10^9. Factoring is feasible but complex.
                # 
                # Simpler: just do direct iteration but with a higher cap since we have time.
                # The range is at most ~n^2/16 for x near n/4. For n=256, that's ~4096.
                # Wait, let me recompute: if x = n/4 + 1, A = 4, B = n*(n/4+1) ≈ n^2/4
                # y_lo ≈ B/4 = n^2/16, y_hi ≈ 2B/4 = n^2/8
                # range ≈ n^2/16. For n=256, that's 4096. So 20000 cap is generous.
                # The "large range" case shouldn't actually trigger for n<=256.
                # Let's just use direct iteration with a reasonable cap.
                y_limit = min(y_hi + 1, y_lo + 50000)
                A_local = A
                B_local = B
                x_local = x_attempt
                for y in range(y_lo, y_limit):
                    if (y - y_lo) % 5000 == 0 and time.time() > deadline:
                        break
                    denom = A_local * y - B_local
                    if denom <= 0:
                        continue
                    if (B_local * y) % denom == 0:
                        z = (B_local * y) // denom
                        if z > y and z != x_local and z <= DENOM_BOUND:
                            if x_local != y and y != z and x_local != z:
                                found = [x_local, y, z]
                                break
            
            if found:
                break
        
        answers[i] = found
    
    return answers
