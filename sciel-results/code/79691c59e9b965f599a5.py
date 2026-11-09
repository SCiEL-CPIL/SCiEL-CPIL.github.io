def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = sorted([i for i in range(1, n + 1) if n % i == 0], reverse=True)
    m = len(divisors)
    div_set = set(divisors)
    
    # Check k=3: need 3 distinct divisors summing to target
    for i in range(m):
        d1 = divisors[i]
        if d1 >= target:
            continue
        rem = target - d1
        if rem < 2:
            break
        for j in range(i + 1, m):
            d2 = divisors[j]
            if d2 >= rem:
                continue
            d3 = rem - d2
            if d3 <= d2:
                break
            if d3 in div_set:
                return [d1, d2, d3]
    
    # Check k=4: need 4 distinct divisors summing to target
    for i in range(m):
        d1 = divisors[i]
        if d1 >= target:
            continue
        rem = target - d1
        if rem < 6:
            break
        for j in range(i + 1, m):
            d2 = divisors[j]
            if d2 >= rem:
                continue
            rem2 = rem - d2
            if rem2 < 3:
                break
            for k in range(j + 1, m):
                d3 = divisors[k]
                if d3 >= rem2:
                    continue
                d4 = rem2 - d3
                if d4 <= d3:
                    break
                if d4 in div_set:
                    return [d1, d2, d3, d4]
    
    # k=5: return known valid solution
    # 2520 + 1680 + 560 + 210 + 9 = 4979
    return [2520, 1680, 560, 210, 9]
