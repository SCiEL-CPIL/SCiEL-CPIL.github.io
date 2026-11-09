import time

def construct():
    answers = []
    deadline = time.time() + 55
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must be > n/4 and < 3n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            # y > B/A => y >= floor(B/A) + 1
            # y < 2B/A => y <= floor((2B-1)/A)
            y_min = B // A + 1
            y_max = (2 * B - 1) // A
            
            if y_min > y_max:
                continue
                
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
