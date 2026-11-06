def construct():
    import math
    
    N = 512
    best_size = 0
    best_divs = [1]
    
    for m in range(1, N + 1):
        # Count divisors
        count = 0
        divs = []
        for i in range(1, int(math.isqrt(m)) + 1):
            if m % i == 0:
                divs.append(i)
                if i != m // i:
                    divs.append(m // i)
                count += 1
                if i != m // i:
                    count += 1
        if count > best_size:
            best_size = count
            best_divs = divs
            
    return sorted(best_divs)
