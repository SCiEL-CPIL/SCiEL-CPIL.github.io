import time
import math

def construct():
    # Deadline guard with safety margin
    start_time = time.time()
    deadline = start_time + 85
    
    answers = []
    MAX_Z = 10**12
    n_total = 254  # n from 3 to 256 inclusive
    
    # Time budget per n: leave some slack for overhead
    # If we've spent too much time, reduce budget for remaining n's
    for idx, n in enumerate(range(3, 257)):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        remaining_n = n_total - idx
        
        if remaining <= 0:
            answers.append(None)
            continue
        
        # Allocate time for this n, with diminishing budget for later n's
        # Give more time to early n's (they're easier), less to later ones
        # But ensure we don't starve any single n
        time_per_n = min(remaining / remaining_n, 0.5)
        n_deadline = time.time() + time_per_n
        
        found = None
        
        # x must satisfy: x > n/4 and x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in a smart order: smaller x first (gives smaller y ranges)
        for x in range(x_min, x_max + 1):
            if time.time() > n_deadline:
                break
                
            denom_x = 4 * x - n
            if denom_x <= 0:
                continue
            
            # B = n * x
            B = n * x
            A = denom_x
            
            # y > B/A and y < (n+B)/A for z > y
            y_min = B // A + 1
            y_max = (n + B) // A
            
            # Ensure y > x for distinctness (assuming x < y < z)
            if y_min <= x:
                y_min = x + 1
                
            if y_max < y_min:
                continue
            
            # Iterate y
            for y in range(y_min, y_max + 1):
                # z = B*y / (A*y - B)
                D = A * y - B
                if D <= 0:
                    continue
                
                if (B * y) % D != 0:
                    continue
                    
                z = (B * y) // D
                
                # Check distinctness and bounds
                if z <= y:
                    continue
                if z > MAX_Z:
                    continue
                
                # x < y < z ensures distinctness
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
