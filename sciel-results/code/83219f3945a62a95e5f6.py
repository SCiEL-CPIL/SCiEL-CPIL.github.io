import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x >= 4/n => x <= 3n/4 (assuming x is smallest)
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
            
            B_sq = B * B
            # We need u | B^2 and u = A*y - B => u ≡ -B (mod A)
            # Also y = (u+B)/A must be integer.
            # And z = (B^2/u + B)/A must be integer.
            # And x, y, z distinct.
            
            # Iterate divisors d of B^2 up to sqrt(B^2)=B
            # To speed up, we can just iterate d from 1 to B
            # But B can be up to 256 * 192 = 49152. 
            # Iterating 50k divisors for each x (50 x's) is 2.5M ops per n.
            # 254 n's => 600M ops. Might be too slow in Python.
            
            # Better: Factorize B and generate divisors?
            # Or just check d in a smarter way?
            # Actually, B is small enough that simple iteration might be okay if optimized.
            # But 600M iterations in Python is likely > 60s.
            
            # Let's optimize the divisor search.
            # We only need to find ONE solution.
            # Most n will have a solution for small x.
            # So we might not iterate all x's.
            
            # Let's try to be efficient.
            # For a given x, we need a divisor u of B^2.
            # u = k*A - B for some k? No, u ≡ -B mod A.
            # u = j*A - B.
            # Since u | B^2, we can iterate j?
            # u > 0 => j*A > B => j > B/A.
            # Also u <= B^2 => j*A - B <= B^2 => j <= (B^2+B)/A.
            
            # Let's iterate j from floor(B/A)+1 to (B^2+B)//A.
            # Range of j:
            # A = 4x-n.
            # If x is close to n/4, A is small.
            # If x = n/4 + 1, A = 4(n/4+1)-n = 4.
            # B = n(n/4+1) ~ n^2/4.
            # B/A ~ n^2/16.
            # B^2/A ~ n^4/64.
            # This range is huge for large n.
            
            # So iterating j is bad.
            # Iterating divisors of B^2 is better if we can generate them fast.
            # B <= 50000.
            # Number of divisors of B^2 is at most ~ 1000-2000.
            # Generating divisors by factoring B is fast.
            
            # Factor B
            temp = B
            factors = {}
            f = 2
            while f * f <= temp:
                while temp % f == 0:
                    factors[f] = factors.get(f, 0) + 1
                    temp //= f
                f += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # Generate divisors of B^2
            # Exponents in B^2 are 2 * exponents in B
            divs = [1]
            for p, e in factors.items():
                new_divs = []
                pe = 2 * e
                pow_p = 1
                for i in range(pe + 1):
                    for d in divs:
                        new_divs.append(d * pow_p)
                    pow_p *= p
                divs = new_divs
            
            # Check each divisor u
            # We can just check if u % A == (-B) % A
            # And then verify y, z
            found_sol = None
            for u in divs:
                # u must be positive, which it is
                # Check congruence
                if u % A == (-B) % A:
                    # Calculate y
                    num_y = u + B
                    if num_y % A != 0:
                        continue
                    y = num_y // A
                    
                    v = B_sq // u
                    num_z = v + B
                    if num_z % A != 0:
                        continue
                    z = num_z // A
                    
                    # Check distinctness
                    if x == y or x == z or y == z:
                        continue
                    
                    # Check bounds
                    if y > 10**12 or z > 10**12:
                        continue
                    
                    found_sol = [x, y, z]
                    break
            
            if found_sol:
                found = found_sol
                break
        
        answers.append(found)
        
    return answers
