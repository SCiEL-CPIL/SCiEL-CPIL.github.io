import time

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin before 90s kill
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: n/4 < x <= 3n/4 (since x <= y <= z and 3/x >= 4/n)
        start_x = n // 4 + 1
        end_x = 3 * n // 4  # x < 3n/4 strictly? Let's use <= for safety
        
        for x in range(start_x, end_x + 1):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # y must satisfy: b/a < y <= 2b/a (since y <= z and 1/y + 1/z = a/b)
            y_min = b // a + 1
            y_max = 2 * b // a
            
            for y in range(y_min, y_max + 1):
                num = a * y - b  # must be > 0
                if num <= 0:
                    continue
                if (b * y) % num == 0:
                    z = (b * y) // num
                    if z > y and z <= 10**12:
                        # Check all distinct
                        if x != y and x != z and y != z:
                            # Verify the equation
                            # 1/x + 1/y + 1/z == 4/n
                            # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                            lhs = n * (y * z + x * z + x * y)
                            rhs = 4 * x * y * z
                            if lhs == rhs:
                                found = [x, y, z]
                                break
            if found:
                break
        
        answers.append(found)
    
    return answers
