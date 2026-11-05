import time

def construct():
    n = 5040
    target = 4979
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order
    divisors.sort(reverse=True)
    n_divs = len(divisors)
    div_set = set(divisors)
    
    # Check for k=1
    if target in divisors:
        return [target]
        
    # Check for k=2
    for i in range(n_divs):
        for j in range(i + 1, n_divs):
            if divisors[i] + divisors[j] == target:
                return [divisors[i], divisors[j]]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Check for k=3 using set lookup
    for i in range(n_divs):
        if divisors[i] >= target:
            continue
        for j in range(i + 1, n_divs):
            if divisors[i] + divisors[j] >= target:
                break
            rem = target - divisors[i] - divisors[j]
            if rem > 0 and rem in div_set and rem != divisors[i] and rem != divisors[j]:
                return [divisors[i], divisors[j], rem]
        if time.time() > deadline:
            break
    
    # Check for k=4 using meet-in-the-middle style approach
    # Fix i, j. Then find k, l such that divisors[k] + divisors[l] = rem2
    for i in range(n_divs):
        if divisors[i] >= target:
            continue
        for j in range(i + 1, n_divs):
            rem2 = target - divisors[i] - divisors[j]
            if rem2 <= 0:
                break
            # Need to find two distinct divisors (not i or j) that sum to rem2
            # Iterate k from j+1, find if rem2 - divisors[k] exists in divisors
            # and is not i, j, or k
            for k in range(j + 1, n_divs):
                d_k = divisors[k]
                if d_k >= rem2:
                    break
                d_l = rem2 - d_k
                if d_l > 0 and d_l in div_set:
                    if d_l != divisors[i] and d_l != divisors[j] and d_l != d_k:
                        return [divisors[i], divisors[j], d_k, d_l]
        if time.time() > deadline:
            break
    
    # Fallback: known k=5 solution
    sol = [2520, 1680, 720, 56, 3]
    if sum(sol) == target and all(x in div_set for x in sol) and len(set(sol)) == len(sol):
        return sol
    
    # Final fallback: greedy approach
    remaining = target
    result = []
    for d in divisors:
        if d <= remaining:
            result.append(d)
            remaining -= d
            if remaining == 0:
                break
    
    if sum(result) == target and all(x in div_set for x in result) and len(set(result)) == len(result):
        return result
    
    return sol
