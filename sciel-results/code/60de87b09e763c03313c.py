import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must be > n/4
        min_x = n // 4 + 1
        # We limit x to a reasonable range to ensure speed
        # Based on previous analysis, small x usually works
        max_x = min(10000, n * 100) 
        
        for x in range(min_x, max_x + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # Check if y, z can be >= 1
            # y = (u+A)/B >= 1 => u+A >= B => u >= B-A
            # u <= A^2 (max divisor)
            # So we need A^2 >= B - A => A^2 + A >= B
            if A * A + A < B:
                continue
                
            # Iterate divisors u of A^2
            # u must be <= A^2
            # We iterate u from 1 to A (since if u > A, A^2/u < A, so we cover pairs)
            # Actually we just need to find ANY valid u.
            # We can iterate u from 1 to A^2, but that's slow.
            # Iterating 1 to A and checking A^2 % u == 0 is O(A).
            
            found_pair = False
            for u in range(1, A + 1):
                if (A * A) % u != 0:
                    continue
                    
                # Check if y is integer
                if (u + A) % B != 0:
                    continue
                    
                y = (u + A) // B
                if y < 1:
                    continue
                    
                v = (A * A) // u
                if v == u:
                    # y = z, not distinct
                    continue
                    
                # Check if z is integer
                if (v + A) % B != 0:
                    continue
                    
                z = (v + A) // B
                if z < 1:
                    continue
                
                # Check distinctness and bounds
                if y == x or z == x or y == z:
                    continue
                    
                if y > 1000000000000 or z > 1000000000000:
                    continue
                    
                found = [x, y, z]
                found_pair = True
                break
                
            if found_pair:
                break
                
        answers.append(found)
        
    return answers
