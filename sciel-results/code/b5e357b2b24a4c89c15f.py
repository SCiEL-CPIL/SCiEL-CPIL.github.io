import math
import time

def construct():
    deadline = time.time() + 80
    answers = [None] * 254  # for n=3..256
    limit_val = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # x must satisfy: 4x - n > 0 => x > n/4
        # and 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        if x_max < x_min:
            x_max = x_min
            
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            B2 = B * B
            
            # We look for U such that U = kA - B, U | B^2, U <= B
            # k_min: U > 0 => kA > B => k > B/A
            # k_max: U <= B => kA - B <= B => kA <= 2B => k <= 2B/A
            
            k_min = B // A + 1
            k_max = (2 * B) // A
            
            for k in range(k_min, k_max + 1):
                U = k * A - B
                if U <= 0:
                    continue
                # Check if U divides B^2
                if B2 % U == 0:
                    V = B2 // U
                    # Check if V = m A - B for some integer m
                    # i.e., (V + B) % A == 0
                    if (V + B) % A == 0:
                        y = (U + B) // A
                        z = (V + B) // A
                        
                        # Check distinctness and bounds
                        if y > 0 and z > 0 and y != z and y != x and z != x:
                            if y <= limit_val and z <= limit_val:
                                found = [x, y, z]
                                break
            if found is not None:
                break
                
        answers[idx] = found
        
    return answers
