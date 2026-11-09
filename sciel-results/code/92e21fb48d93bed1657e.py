import time

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must be > n/4
        start_x = n // 4 + 1
        # x must be < 3n/4 because 3/x > 4/n -> x < 3n/4
        end_x = 3 * n // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
                
            # Calculate bounds for y
            # y > x
            # y > nx / (4x - n)  => den > 0
            # y < 2nx / (4x - n) => z > y
            
            denom_factor = 4 * x - n
            if denom_factor <= 0:
                continue
                
            # Lower bound for y: y > nx / denom_factor
            # y_min = floor(nx / denom_factor) + 1
            y_min = (n * x) // denom_factor + 1
            if y_min <= x:
                y_min = x + 1
                
            # Upper bound for y: y < 2nx / denom_factor
            # y_max = floor(2nx / denom_factor)
            y_max = (2 * n * x) // denom_factor
            
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                    
                # Calculate numerator and denominator for z
                # 1/z = 4/n - 1/x - 1/y = (4xy - nx - ny) / nxy
                # z = nxy / (4xy - nx - ny)
                den = 4 * x * y - n * x - n * y
                if den <= 0:
                    # Should not happen if y > nx/denom_factor, but just in case
                    continue
                
                num = n * x * y
                
                if num % den == 0:
                    z = num // den
                    if z > y:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
