import time
import math

# EVOLVE-BLOCK-START

def construct():
    answers = [None] * 254
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # For each n from 3 to 256, find distinct x, y, z with 1/x + 1/y + 1/z = 4/n
    # We search over x (smallest denominator), then for each x, search over y.
    # Given x and y, we solve for z: 1/z = 4/n - 1/x - 1/y = (4xy - ny - nx) / (nxy)
    # So z = nxy / (4xy - ny - nx) = nxy / (y(4x - n) - nx)
    # Let A = 4x - n, B = nx. Then z = nxy / (Ay - B) = (n*x*y) / (A*y - B)
    # We need A*y - B > 0, i.e., y > B/A = nx/(4x-n)
    # Also need z to be integer and z > y and z != x and z != y.
    
    for n in range(3, 257):
        if time.time() > deadline:
            break
        idx = n - 3
        found = None
        
        # x must satisfy: 1/x < 4/n => x > n/4
        # Also, for a solution to exist with x as smallest, we need 1/x <= 4/n / 3 is not quite right
        # Actually: 1/x + 1/y + 1/z = 4/n, with x <= y <= z
        # So 3/x >= 4/n => x <= 3n/4
        # And 1/x < 4/n => x > n/4
        # So x is in (n/4, 3n/4]
        
        x_start = n // 4 + 1
        x_end = (3 * n) // 4 + 1  # inclusive upper bound
        
        # Limit search range for x to keep within time budget
        # For small n, x range is small. For larger n, we may need to be careful.
        
        for x in range(x_start, min(x_end, x_start + 200) + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y must satisfy: y > B/A and y >= x+1 (since x <= y, and distinct so y > x)
            y_min = max(x + 1, B // A + 1)
            
            # Also, since y <= z, we have 1/y + 1/z <= 1/y + 1/y = 2/y
            # And 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx) = A/B
            # So 2/y >= A/B => y <= 2B/A
            y_max = (2 * B) // A
            
            if y_max < y_min:
                continue
            
            # Limit y search range
            # For each x, the number of y values is about 2B/A - B/A = B/A = nx/(4x-n)
            # This can be large for x close to n/4. Let's cap it.
            y_search_end = min(y_max, y_min + 500)
            
            for y in range(y_min, y_search_end + 1):
                if time.time() > deadline:
                    break
                
                denom = A * y - B
                if denom <= 0:
                    continue
                
                num = n * x * y
                if num % denom != 0:
                    continue
                
                z = num // denom
                if z <= y:
                    continue
                if z == x:
                    continue
                # z > y > x, so all distinct and positive
                if z <= 1000000000000:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers

# EVOLVE-BLOCK-END
