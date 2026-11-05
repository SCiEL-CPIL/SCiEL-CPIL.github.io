import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    answers = []
    for n in range(3, 257):
        found = None
        # Search for x starting from the minimum possible value
        # x must satisfy 4/n - 1/x > 0 => 1/x < 4/n => x > n/4
        start_x = n // 4 + 1
        
        # Try a range of x values. 
        # Empirically, small offsets from n/4 work.
        # If we don't find a solution in the first 50, we might need more, 
        # but let's start with a reasonable bound.
        for x in range(start_x, start_x + 200):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y must be > x
            # z > y implies 1/z < 1/y => By/(Ay-B) > y => B > Ay - B => 2B > Ay => y < 2B/A
            y_min = x + 1
            y_max = (2 * B) // A
            
            if y_max < y_min:
                continue
                
            # Iterate y
            # To speed up, we can step by 1. 
            # The range is usually small enough.
            for y in range(y_min, y_max + 1):
                D = A * y - B
                if D <= 0:
                    continue
                if (B * y) % D == 0:
                    z = (B * y) // D
                    if z > y:
                        found = [x, y, z]
                        break
            if found:
                break
                
        answers.append(found)
        
    return answers
