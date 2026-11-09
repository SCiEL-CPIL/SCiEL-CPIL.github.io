import time

def construct():
    answers = []
    deadline = time.time() + 55.0
    LIMIT_Z = 10**12
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must be > n/4 and < 3n/4
        # x_min = n//4 + 1
        # x_max = (3*n - 1) // 4
        
        # Iterate x from largest to smallest to potentially find solutions faster?
        # Actually, for large x, A is large, range of y is small.
        # For small x, A is small, range of y is large.
        # It might be better to iterate x from smallest to largest?
        # No, usually solutions with larger denominators (larger x) are easier to find?
        # Let's stick to x from x_min to x_max or vice versa.
        # The number of y iterations is ~ B/A = nx / (4x-n).
        # This function is decreasing in x.
        # So small x has large search space, large x has small search space.
        # If a solution exists, it's likely to be found quickly in the small search space (large x).
        # So iterate x from x_max down to x_min.
        
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        for x in range(x_max, x_min - 1, -1):
            A = 4 * x - n
            B = n * x
            
            # y range
            # y > B/A
            y_min = B // A + 1
            # y < 2B/A
            # 2B/A might not be integer.
            # y_max is floor((2B - 1) / A)
            y_max = (2 * B - 1) // A
            
            if y_max < y_min:
                continue
                
            # We also need y > x
            if y_min <= x:
                y_min = x + 1
            if y_max < y_min:
                continue
                
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                # denom is positive because y > B/A
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z <= LIMIT_Z:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
