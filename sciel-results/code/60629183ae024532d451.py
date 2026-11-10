"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Use a safety margin to ensure we return before the 90s hard limit
    deadline = start_time + 85.0
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # 1/x + 1/y + 1/z = 4/n
        # Assume x <= y <= z for uniqueness of search space
        # Since 1/x >= 1/y >= 1/z, 3/x >= 4/n => x <= 3n/4
        # Also 1/x < 4/n => x > n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            # 4/n - 1/x = 1/y + 1/z
            # (4x - n) / nx = 1/y + 1/z
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # 1/y + 1/z = A/B
            # y must satisfy: B/A < y <= 2B/A
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            # Enforce ordering y >= x
            if y_min < x:
                y_min = x
            
            if y_min > y_max:
                continue
                
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                    
                # z = B*y / (A*y - B)
                denom_z = A * y - B
                if denom_z <= 0:
                    continue
                
                num_z = B * y
                if num_z % denom_z != 0:
                    continue
                
                z = num_z // denom_z
                
                # Enforce ordering z >= y
                if z < y:
                    continue
                
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                
                # Check upper bound
                if z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
