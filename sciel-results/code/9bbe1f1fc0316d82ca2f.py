"""Optimized constructor for distinct three-term unit-fraction reps of 4/n."""
import time
from math import gcd

DENOM_BOUND = 10**12

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s kill
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        x_min = n // 4 + 1
        found = None
        
        # Try a larger range of x values to find solutions
        # The key insight: for 1/x + 1/y + 1/z = 4/n, we can fix x and solve for y,z
        # 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # So (y - (nx)/(4x-n))(z - (nx)/(4x-n)) = (nx/(4x-n))^2
        # Let A = 4x - n, B = nx
        # Then (Ay - B)(Az - B) = B^2
        # So Az - B = B^2 / (Ay - B)
        # z = (B + B^2/(Ay - B)) / A
        # For z to be integer, Ay - B must divide B^2
        
        # We iterate x from x_min upward, and for each x, find divisors d of B^2
        # such that d = Ay - B for some integer y > x, and z = (B + B^2/d)/A is integer > y
        
        # More efficient: iterate y directly in the valid range
        # y must satisfy: 1/y + 1/z = A/B with z >= y
        # So 1/y >= A/B/2, i.e., y <= 2B/A
        # And 1/y < A/B, i.e., y > B/A
        # So B/A < y <= 2B/A
        
        # For x close to n/4, A is small, so the range is large
        # For x >> n/4, A is large, so the range is small
        
        # Strategy: try x values in order of increasing A (i.e., increasing x)
        # For each x, the y range is [B//A + 1, 2B//A]
        # The length of this range is approximately B/A = nx/(4x-n)
        # For x = n/4 + k, A = 4k, B = n(n/4+k) = n^2/4 + nk
        # Range length ≈ (n^2/4 + nk)/(4k) = n^2/(16k) + n/4
        # For k=1: range ≈ n^2/16 + n/4. For n=256: 4096 + 64 = 4160
        # For k=2: range ≈ n^2/32 + n/4 = 2048 + 64 = 2112
        # So total work for first few x values is manageable
        
        # We'll try up to 500 x values, but break early if found
        for x_attempt in range(x_min, x_min + 500):
            if time.time() > deadline:
                break
            A = 4 * x_attempt - n
            if A <= 0:
                continue
            B = n * x_attempt
            
            # y range: B//A + 1 to 2B//A
            y_lo = B // A + 1
            y_hi = (2 * B) // A
            
            if y_hi < y_lo:
                continue
            
            # Ensure y > x_attempt for ordering
            if y_lo <= x_attempt:
                y_lo = x_attempt + 1
            if y_lo > y_hi:
                continue
            
            # Cap the range to avoid pathological cases
            # For x = x_min, A = 4*(n//4+1) - n. If n=4m, A=4. If n=4m+1, A=1.
            # Wait, x_min = n//4 + 1.
            # If n = 4m, x_min = m+1, A = 4(m+1) - 4m = 4
            # If n = 4m+1, x_min = m+1, A = 4(m+1) - (4m+1) = 3
            # If n = 4m+2, x_min = m+1, A = 4(m+1) - (4m+2) = 2
            # If n = 4m+3, x_min = m+1, A = 4(m+1) - (4m+3) = 1
            
            # So A can be as small as 1. If A=1, B = n*x. Range is B+1 to 2B. Length ~ B = n*x.
            # For n=255, x_min=64, A=1, B=255*64=16320. Range length ~ 16320.
            # That's a lot. Let's cap it.
            
            y_limit = min(y_hi + 1, y_lo + 20000)
            
            A_local = A
            B_local = B
            x_local = x_attempt
            bound = DENOM_BOUND
            
            for y in range(y_lo, y_limit):
                if (y - y_lo) % 2000 == 0 and time.time() > deadline:
                    break
                denom = A_local * y - B_local
                if denom <= 0:
                    continue
                if (B_local * y) % denom == 0:
                    z = (B_local * y) // denom
                    if z > y and z != x_local and z <= bound:
                        if x_local != y and y != z and x_local != z:
                            found = [x_local, y, z]
                            break
            if found:
                break
        
        answers[i] = found
    
    return answers
