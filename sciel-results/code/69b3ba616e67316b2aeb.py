def construct():
    import math
    
    n = 512
    max_divisors = 0
    best_m = 1
    best_divisors = []
    
    for m in range(1, n + 1):
        # Count divisors of m
        count = 0
        divs = []
        for i in range(1, int(math.isqrt(m)) + 1):
            if m % i == 0:
                divs.append(i)
                if i != m // i:
                    divs.append(m // i)
                count += 1 if i != m // i else 0
                # Actually simpler to just list and count
        # Let's just generate list and count
        divs = []
        for i in range(1, int(math.isqrt(m)) + 1):
            if m % i == 0:
                divs.append(i)
                if i != m // i:
                    divs.append(m // i)
        
        if len(divs) > max_divisors:
            max_divisors = len(divs)
            best_m = m
            best_divisors = divs
            
    return best_divisors
