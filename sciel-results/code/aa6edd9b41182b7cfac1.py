import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    answers = [None] * 254
    
    # Phase 1: Direct x,y iteration for all n
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y must satisfy: 1/y + 1/z = A/B
            # y > B/A (so A*y - B > 0)
            # y <= 2*B/A (since y <= z implies 2/y >= A/B)
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A
            
            if y_max < y_min:
                continue
            
            # Limit y range to avoid too-long loops
            y_limit = min(y_max, y_min + 2000)
            
            for y in range(y_min, y_limit + 1):
                D = A * y - B
                if D <= 0:
                    continue
                N = B * y
                if N % D == 0:
                    z = N // D
                    if z > y and x != y and x != z:
                        if x <= 10**12 and y <= 10**12 and z <= 10**12:
                            # Verify
                            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                                found = [x, y, z]
                                break
            if found:
                break
        answers[idx] = found
    
    # Phase 2: For unsolved n, try divisor-based search
    for n in range(3, 257):
        idx = n - 3
        if answers[idx] is not None:
            continue
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = min((3 * n) // 4, x_min + 100)
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            # Factorize B
            b_factors = {}
            temp_b = B
            d = 2
            while d * d <= temp_b:
                while temp_b % d == 0:
                    b_factors[d] = b_factors.get(d, 0) + 1
                    temp_b //= d
                d += 1
            if temp_b > 1:
                b_factors[temp_b] = b_factors.get(temp_b, 0) + 1
            
            # Generate divisors of B^2
            divs = [1]
            for p, e in b_factors.items():
                new_divs = []
                exp_max = 2 * e
                pk = 1
                for k in range(exp_max + 1):
                    for dv in divs:
                        new_divs.append(dv * pk)
                    pk *= p
                divs = new_divs
            
            for u in divs:
                if (u + B) % A != 0:
                    continue
                y = (u + B) // A
                if y <= 0:
                    continue
                v = B2 // u
                if (v + B) % A != 0:
                    continue
                z = (v + B) // A
                if z <= 0:
                    continue
                if x == y or x == z or y == z:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            if found:
                break
        answers[idx] = found
    
    return answers
