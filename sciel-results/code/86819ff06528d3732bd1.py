import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    div_set = set(divisors)
    
    # 1 divisor
    if target in div_set:
        return [target]
        
    # 2 divisors
    for d1 in divisors:
        if time.time() - start_time > DEADLINE:
            break
        rem = target - d1
        if rem > d1 and rem in div_set:
            return [d1, rem]
            
    # 3 divisors
    for i, d1 in enumerate(divisors):
        if time.time() - start_time > DEADLINE:
            break
        rem2 = target - d1
        if rem2 <= 0:
            break
        for j in range(i + 1, len(divisors)):
            d2 = divisors[j]
            if d2 >= rem2:
                break
            rem3 = rem2 - d2
            if rem3 > d2 and rem3 in div_set:
                return [d1, d2, rem3]
                
    # 4 divisors
    for i, d1 in enumerate(divisors):
        if time.time() - start_time > DEADLINE:
            break
        rem3 = target - d1
        if rem3 <= 0:
            break
        for j in range(i + 1, len(divisors)):
            d2 = divisors[j]
            if d2 >= rem3:
                break
            rem2 = rem3 - d2
            if rem2 <= 0:
                break
            for k in range(j + 1, len(divisors)):
                d3 = divisors[k]
                if d3 >= rem2:
                    break
                rem4 = rem2 - d3
                if rem4 > d3 and rem4 in div_set:
                    return [d1, d2, d3, rem4]
                    
    # Fallback: Greedy
    divisors_desc = sorted(divisors, reverse=True)
    remainder = target
    chosen = []
    for d in divisors_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
