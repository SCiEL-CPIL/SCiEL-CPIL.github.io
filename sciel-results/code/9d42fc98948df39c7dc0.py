"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    answers = [None] * 254  # Index 0 corresponds to n=3, index 253 to n=256
    
    for idx in range(254):
        n = idx + 3
        
        # Check time
        if time.time() > deadline:
            break
            
        found = None
        
        # x < y < z
        # 1/x + 1/y + 1/z = 4/n
        # 1/x < 4/n => x > n/4
        # 3/x > 4/n => x < 3n/4
        
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
                
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            # Let A = 4x - n, B = nx
            # 1/y + 1/z = A/B
            # y > B/A = nx / (4x - n)
            # z > y => 1/z < 1/y => 1/z = A/B - 1/y < 1/y
            # A/B < 2/y => y < 2B/A = 2nx / (4x - n)
            
            A = 4 * x - n
            B = n * x
            
            # y must be integer > B/A
            y_min = B // A + 1
            # y must be integer < 2B/A
            y_max = (2 * B) // A
            
            # Ensure y > x
            if y_min <= x:
                y_min = x + 1
                
            for y in range(y_min, y_max + 1):
                # 1/z = A/B - 1/y = (Ay - B) / (By)
                # z = By / (Ay - B)
                num = B * y
                den = A * y - B
                
                if den <= 0:
                    continue
                    
                if num % den == 0:
                    z = num // den
                    if z > y:
                        # Check distinctness: x < y < z already ensures distinctness
                        # Check bound
                        if z <= 1000000000000:
                            found = [x, y, z]
                            break
                            
            if found:
                break
                
        answers[idx] = found
        
    return answers
# EVOLVE-BLOCK-END
