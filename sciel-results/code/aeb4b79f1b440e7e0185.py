import math

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            A = 4 * x - n
            B = n * x
            
            y_start = B // A + 1
            y_end = (2 * B) // A
            
            if y_start > y_end:
                continue
                
            for y in range(y_start, y_end + 1):
                denom = A * y - B
                # denom is always positive in this range
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    # z > y is guaranteed by y <= y_end
                    if z <= 1000000000000:
                        found = [x, y, z]
                        break
            if found:
                break
        answers.append(found)
    return answers
