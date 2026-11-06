import time
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    MAX_DENOM = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        
        # Determine A based on n % 4 to make x an integer and A small
        # 4x - n = A => x = (n+A)/4
        # n % 4 == 0 => A=4 => x = n/4 + 1
        # n % 4 == 1 => A=3 => x = (n+3)/4
        # n % 4 == 2 => A=2 => x = (n+2)/4
        # n % 4 == 3 => A=1 => x = (n+1)/4
        
        rem = n % 4
        if rem == 0:
            A = 4
        elif rem == 1:
            A = 3
        elif rem == 2:
            A = 2
        else: # rem == 3
            A = 1
            
        x = (n + A) // 4
        
        # Verify x is integer and 4x - n == A
        if (n + A) % 4 != 0:
            # Fallback, though logic above ensures this
            continue
            
        B = n * x
        B_sq = B * B
        
        # We need u such that:
        # 1. u divides B^2
        # 2. u = -B (mod A)
        # 3. y = B + u > x
        # 4. z = B + B^2/u > y
        # 5. x, y, z are distinct
        
        # Try small u values
        # Since A is small (1-4), the modulo constraint is loose.
        # We just need u to be a divisor of B^2.
        # 1 is always a divisor.
        # Check u=1
        # If A=1: 1 = -B (mod 1) => 0=0. Always ok.
        # If A=2: 1 = -B (mod 2). If B is odd, -B is 1 mod 2. If B is even, -B is 0 mod 2.
        # If A=3: 1 = -B (mod 3).
        # If A=4: 1 = -B (mod 4).
        
        # Let's iterate u from 1 to 1000
        # Optimization: we can just check divisors of B^2? No, B can be huge.
        # But we only need ONE solution.
        # For A=1, u=1 always works.
        #   y = B+1, z = B+B^2.
        #   x = (n+1)/4.
        #   Are they distinct?
        #   n=3, x=1. B=3. y=4, z=12. Distinct.
        #   n=7, x=2. B=14. y=15, z=210. Distinct.
        #   For large n, x ~ n/4, y ~ n^2/4. Distinct.
        
        # For A=2, we need u odd if B is odd, u even if B is even?
        # u = -B (mod 2).
        # If B is even, u must be even. u=2 is a divisor of B^2 if B is even.
        # If B is odd, u must be odd. u=1 is a divisor.
        
        # For A=3, u = -B (mod 3).
        # If B % 3 == 0, u must be 0 mod 3. u=3 is a divisor if 3|B.
        # If B % 3 == 1, u must be 2 mod 3. u=2? 2|B^2?
        # If B % 3 == 2, u must be 1 mod 3. u=1?
        
        # For A=4, u = -B (mod 4).
        
        # General strategy: Try u in range(1, 20).
        # Check if B^2 % u == 0 and (u + B) % A == 0.
        
        for u in range(1, 20):
            if B_sq % u != 0:
                continue
            if (u + B) % A != 0:
                continue
            
            y = B + u
            z = B + B_sq // u
            
            # Check bounds
            if y > MAX_DENOM or z > MAX_DENOM:
                continue
                
            # Check distinctness
            if x == y or x == z or y == z:
                continue
                
            # Check positivity (should be true)
            if x < 1 or y < 1 or z < 1:
                continue
                
            found = [x, y, z]
            break
        
        # If not found with small u, try next x (A+4, A+8, etc.)
        if not found:
            # Fallback: iterate x
            for k in range(1, 10):
                A_curr = A + 4 * k
                x_curr = (n + A_curr) // 4
                if (n + A_curr) % 4 != 0:
                    continue
                B_curr = n * x_curr
                B_sq_curr = B_curr * B_curr
                
                for u in range(1, 20):
                    if B_sq_curr % u != 0:
                        continue
                    if (u + B_curr) % A_curr != 0:
                        continue
                    
                    y_curr = B_curr + u
                    z_curr = B_curr + B_sq_curr // u
                    
                    if y_curr > MAX_DENOM or z_curr > MAX_DENOM:
                        continue
                        
                    if x_curr == y_curr or x_curr == z_curr or y_curr == z_curr:
                        continue
                        
                    found = [x_curr, y_curr, z_curr]
                    break
                if found:
                    break
        
        results[idx] = found
        
    return results
