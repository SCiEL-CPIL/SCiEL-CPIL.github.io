"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # We iterate from largest to smallest.
    # Larger numbers are less likely to divide the product of two other numbers
    # because they have fewer multiples in the range [2, 256].
    # Also, if a large number is the divisor, it's hard to satisfy a|bc.
    
    S = []
    
    for x in range(256, 1, -1):
        if time.time() > deadline:
            break
        
        can_add = True
        n = len(S)
        
        if n < 2:
            S.append(x)
            continue
            
        for i in range(n):
            y = S[i]
            for j in range(n):
                if i == j:
                    continue
                z = S[j]
                
                # Check if any of x, y, z divides the product of the other two
                # 1. x | y*z
                if (y * z) % x == 0:
                    can_add = False
                    break
                # 2. y | x*z
                if (x * z) % y == 0:
                    can_add = False
                    break
                # 3. z | x*y
                if (x * y) % z == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S.append(x)
            
    return S
# EVOLVE-BLOCK-END
