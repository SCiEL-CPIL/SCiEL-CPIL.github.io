import math

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        
        # Approach 1: Algebraic Identity for n divisible by 4
        if n % 4 == 0:
            k = n // 4
            x = 2 * k
            y = 3 * k
            z = 6 * k
            # Check distinctness
            if x != y and y != z and x != z:
                if x <= 1000000000000 and y <= 1000000000000 and z <= 1000000000000:
                    # Verify
                    if 1/x + 1/y + 1/z == 4/n: # Floating point check might be risky, use cross mult
                        if x*y*z*4 == n*(y*z + x*z + x*y):
                            found = [x, y, z]
        
        # Approach 2: 2-Loop Search for others
        if not found:
            start_x = n // 4 + 1
            # Extend search range
            max_x = n + 200
            for x in range(start_x, max_x):
                # 1/y < 4/n - 1/x
                # y > nx / (4x - n)
                denom_y = 4 * x - n
                if denom_y <= 0:
                    continue
                min_y = (n * x) // denom_y + 1
                if min_y <= x:
                    min_y = x + 1
                # Limit y search
                max_y = min_y + 200
                for y in range(min_y, max_y):
                    num = 4 * x * y - n * y - n * x
                    if num <= 0:
                        continue
                    den = n * x * y
                    if den % num == 0:
                        z = den // num
                        if z > y:
                            if z <= 1000000000000:
                                # Cross multiplication check
                                if 4 * x * y * z == n * (y * z + x * z + x * y):
                                    found = [x, y, z]
                                    break
                if found:
                    break
                    
        answers.append(found)
    return answers
