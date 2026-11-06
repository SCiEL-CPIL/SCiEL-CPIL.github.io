def construct():
    N = 512
    
    def count_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count

    best_L = 1
    max_divs = 0
    
    for L in range(1, N + 1):
        d = count_divisors(L)
        if d > max_divs:
            max_divs = d
            best_L = L
            
    # Generate divisors of best_L
    divisors = []
    i = 1
    while i * i <= best_L:
        if best_L % i == 0:
            divisors.append(i)
            if i != best_L // i:
                divisors.append(best_L // i)
        i += 1
        
    return sorted(divisors)
