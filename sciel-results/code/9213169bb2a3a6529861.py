import time
import math

def construct():
    """
    Find distinct three-term unit-fraction representations of 4/n for n=3..256.
    Returns a list of 254 rows, each None or [x, y, z] with 1/x + 1/y + 1/z = 4/n.
    """
    deadline = time.time() + 85  # Safety margin before 90s kill
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    
    for idx in range(254):
        n = idx + 3
        
        # If we're running out of time, skip remaining
        if time.time() > deadline - 2:
            break
        
        found = None
        
        # For 1/x + 1/y + 1/z = 4/n with x <= y <= z:
        # 1/x < 4/n => x > n/4
        # 3/x > 4/n => x < 3n/4
        # So x ranges from floor(n/4)+1 to floor((3n-1)/4)
        
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        # For each x, we need 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let A = 4x - n, B = nx
        # 1/y + 1/z = A/B => (y+z)/(yz) = A/B => B(y+z) = A*yz
        # A*yz - B*y - B*z = 0
        # (Ay - B)(Az - B) = B^2
        # So we need to find factor pairs of B^2 where both factors are divisible by A
        # and yield y > x, z > y, and y != z (distinct)
        
        # Alternative: iterate y from x+1 to some bound
        # 1/y < A/B => y > B/A = nx/(4x-n)
        # 1/z = A/B - 1/y = (Ay - B)/(By)
        # z = By / (Ay - B)
        # Need Ay - B > 0 => y > B/A
        # Also z > y => By/(Ay-B) > y => B > Ay - B => 2B > Ay => y < 2B/A
        # So y ranges from floor(B/A)+1 to floor(2B/A)
        
        # To minimize search, try larger x first (smaller y range)
        for x in range(x_max, x_min - 1, -1):
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # y must satisfy: B/A < y < 2B/A
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            # Ensure y > x
            if y_min <= x:
                y_min = x + 1
            
            if y_min > y_max:
                continue
            
            # Search y in range
            for y in range(y_min, y_max + 1):
                # Check if z = B*y / (A*y - B) is an integer > y
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom != 0:
                    continue
                z = (B * y) // denom
                if z <= y:
                    continue
                # Check distinctness: x != y, y != z, x != z
                # We already have x < y < z so they're distinct
                if x != y and y != z and x != z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
