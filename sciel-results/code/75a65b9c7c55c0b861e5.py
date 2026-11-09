def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = sorted([i for i in range(1, n + 1) if n % i == 0], reverse=True)
    m = len(divisors)
    
    # Check k=3: need 3 distinct divisors summing to target
    # d1 + d2 + d3 = 4979
    # d1 must be large: max d2+d3 = divisors[1]+divisors[2] = 1680+840 = 2520
    # So d1 >= 4979 - 2520 = 2459. Only d1=2520 qualifies.
    # Then d2+d3 = 2459. Max d2 = 1680, so d3 = 2459-d2.
    # Check all pairs from divisors[1:]
    found_k3 = None
    for i in range(m):
        d1 = divisors[i]
        if d1 >= target:
            continue
        if i + 2 >= m:
            break
        max_rem = divisors[i+1] + divisors[i+2]
        if d1 + max_rem < target:
            break
        rem = target - d1
        # Find 2 distinct divisors from divisors[i+1:] summing to rem
        for j in range(i+1, m):
            d2 = divisors[j]
            if d2 >= rem:
                continue
            if j + 1 >= m:
                break
            if d2 + divisors[j+1] < rem:
                break
            d3_needed = rem - d2
            # Check if d3_needed is in divisors[j+1:]
            # Since sorted desc, d3_needed must be < d2 and present
            # Use binary search or set for the suffix
            # For efficiency with small m, just check
            for k in range(j+1, m):
                if divisors[k] == d3_needed:
                    found_k3 = [d1, d2, d3_needed]
                    return found_k3
                if divisors[k] < d3_needed:
                    break
    
    # Check k=4: need 4 distinct divisors summing to target
    # d1 must be 2520 (as analyzed: any d1 <= 1680 gives max sum <= 3500 < 4979)
    # So d1 = 2520, remaining = 2459 from 3 divisors in divisors[1:]
    found_k4 = None
    if divisors[0] < target:
        d1 = divisors[0]
        rem = target - d1
        # Find 3 distinct divisors from divisors[1:] summing to rem
        rest = divisors[1:]
        m2 = len(rest)
        for j in range(m2):
            d2 = rest[j]
            if d2 >= rem:
                continue
            if j + 2 >= m2:
                break
            max_rem2 = rest[j+1] + rest[j+2]
            if d2 + max_rem2 < rem:
                break
            rem2 = rem - d2
            # Find 2 distinct divisors from rest[j+1:] summing to rem2
            for k in range(j+1, m2):
                d3 = rest[k]
                if d3 >= rem2:
                    continue
                if k + 1 >= m2:
                    break
                if d3 + rest[k+1] < rem2:
                    break
                d4_needed = rem2 - d3
                # Check if d4_needed is in rest[k+1:]
                for l in range(k+1, m2):
                    if rest[l] == d4_needed:
                        found_k4 = [d1, d2, d3, d4_needed]
                        return found_k4
                    if rest[l] < d4_needed:
                        break
    
    # k=5: return known valid solution
    # 2520 + 1680 + 560 + 210 + 9 = 4979
    return [2520, 1680, 560, 210, 9]
