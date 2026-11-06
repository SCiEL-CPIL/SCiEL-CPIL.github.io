import time

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        
        found = None
        
        # Strategy 1: Try x in a moderate range around n/4 to n
        # Strategy 2: Try specific parameterized forms
        
        # First, try the direct divisor approach with wider x range
        x_start = n // 4 + 1
        x_end = max(4 * n + 100, 1000)
        
        for x in range(x_start, min(x_end, 5000)):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            B_sq = B * B
            
            # Factor B to get divisors of B^2
            temp = B
            factors = {}
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    temp //= d
                d += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # Generate divisors of B^2, but only keep those <= B
            # to keep y in a reasonable range
            divs = [1]
            for p, e in factors.items():
                new_divs = []
                for dv in divs:
                    val = dv
                    for _ in range(2 * e + 1):
                        new_divs.append(val)
                        val *= p
                divs = new_divs
            
            # Sort to try smaller u first (gives smaller y)
            divs.sort()
            
            for u in divs:
                if u > B:
                    break
                if (u + B) % A != 0:
                    continue
                v = B_sq // u
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                if x == y or x == z or y == z:
                    continue
                
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        # If not found, try larger x values
        if not found and time.time() < deadline:
            for x in range(max(x_end, 5000), 50000):
                if time.time() > deadline:
                    break
                
                A = 4 * x - n
                if A <= 0:
                    continue
                
                B = n * x
                B_sq = B * B
                
                temp = B
                factors = {}
                d = 2
                while d * d <= temp:
                    while temp % d == 0:
                        factors[d] = factors.get(d, 0) + 1
                        temp //= d
                    d += 1
                if temp > 1:
                    factors[temp] = factors.get(temp, 0) + 1
                
                divs = [1]
                for p, e in factors.items():
                    new_divs = []
                    for dv in divs:
                        val = dv
                        for _ in range(2 * e + 1):
                            new_divs.append(val)
                            val *= p
                    divs = new_divs
                
                divs.sort()
                
                for u in divs:
                    if u > B:
                        break
                    if (u + B) % A != 0:
                        continue
                    v = B_sq // u
                    if (v + B) % A != 0:
                        continue
                    
                    y = (u + B) // A
                    z = (v + B) // A
                    
                    if y <= 0 or z <= 0:
                        continue
                    if y > BOUND or z > BOUND:
                        continue
                    if x == y or x == z or y == z:
                        continue
                    
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
                
                if found:
                    break
        
        answers[idx] = found
    
    return answers
