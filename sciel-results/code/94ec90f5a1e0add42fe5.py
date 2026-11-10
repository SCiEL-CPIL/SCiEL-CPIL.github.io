import time
import math

def get_divisors(n):
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return divs

def construct():
    answers = []
    deadline = time.time() + 85.0
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        x_start = n // 4 + 1
        # x must be < 3n/4
        x_end = min(3 * n // 4, x_start + 5000)
        
        for x in range(x_start, x_end + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # Method 1: Divisor search if B is small
            # z = B + B^2 / (A*D)
            # y = (D+B)/A
            # A*D must divide B^2
            # D must divide B^2
            # Let's check if B is small enough to factorize
            if B < 100000:
                B_sq = B * B
                # Get divisors of B_sq
                # To be efficient, get divisors of B, then square them? No.
                # Just get divisors of B_sq.
                # If B is small, B_sq is small.
                divs = get_divisors(B_sq)
                for D in divs:
                    # Check if A*D divides B_sq
                    if (B_sq % (A * D)) != 0:
                        continue
                    
                    # Check if y is integer
                    if (D + B) % A != 0:
                        continue
                    
                    y = (D + B) // A
                    z = B + B_sq // (A * D)
                    
                    if y <= x or z <= y:
                        continue
                    if z > 10**12:
                        continue
                    
                    # Verify distinctness and equation
                    if x != y and y != z and x != z:
                        # Cross multiplication check
                        # 1/x + 1/y + 1/z = 4/n
                        # n(yz + xz + xy) = 4xyz
                        if n * (y * z + x * z + x * y) == 4 * x * y * z:
                            found = [x, y, z]
                            break
                if found:
                    break
            
            # Method 2: Range search
            if not found:
                y_min = B // A + 1
                y_max = (2 * B - 1) // A
                
                if y_min <= x:
                    y_min = x + 1
                
                if y_max < y_min:
                    continue
                
                for y in range(y_min, y_max + 1):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom == 0:
                        z = (B * y) // denom
                        if z > y and z <= 10**12:
                            if x != y and y != z and x != z:
                                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                                    found = [x, y, z]
                                    break
                if found:
                    break
                    
        answers.append(found)
        
    return answers
