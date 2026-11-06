import time

def construct():
    answers = []
    LIMIT = 10**12
    start_time = time.time()
    
    for n in range(3, 257):
        # Check time limit
        if time.time() - start_time > 80:
            answers.append(None)
            continue
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            # Let A = 4x - n, B = nx
            A = 4 * x - n
            B = n * x
            
            # If A <= 0, no solution (1/x >= 4/n)
            if A <= 0:
                continue
                
            # y must satisfy 1/y < 4/n - 1/x => y > B/A
            # and 3/y > 4/n - 1/x => y < 3B/A
            # Also y > x for distinctness (assuming x < y < z)
            start_y = max(x + 1, B // A + 1)
            end_y = (3 * B) // A
            
            # If the range is empty, skip
            if start_y > end_y:
                continue
                
            for y in range(start_y, end_y + 1):
                # U = A*y - B
                # We need U | B^2
                U = A * y - B
                if U <= 0:
                    continue
                    
                if B * B % U != 0:
                    continue
                    
                V = (B * B) // U
                # z = (V + B) // A
                # Check if (V + B) is divisible by A
                if (V + B) % A != 0:
                    continue
                    
                z = (V + B) // A
                
                # Check distinctness and ordering
                # We assumed x < y. We need y < z.
                if z > y:
                    # Check bounds
                    if z <= LIMIT:
                        # Check distinctness again (x!=y is ensured by start_y, y!=z by z>y, x!=z by x<y<z)
                        found = [x, y, z]
                        break
                        
            if found:
                break
                
        answers.append(found)
        
    return answers
