def construct():
    import math
    N = 512
    best_set = []
    max_divs = 0
    
    for k in range(1, N + 1):
        count = 0
        # Count divisors
        i = 1
        while i * i <= k:
            if k % i == 0:
                count += 1
                if i != k // i:
                    count += 1
            i += 1
        
        if count > max_divs:
            max_divs = count
            # Reconstruct divisors for this k
            divs = []
            i = 1
            while i * i <= k:
                if k % i == 0:
                    divs.append(i)
                    if i != k // i:
                        divs.append(k // i)
                i += 1
            best_set = divs
            
    return best_set
