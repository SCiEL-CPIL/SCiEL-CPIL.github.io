"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    divisor_set = set(divisors)
    
    # k=1
    if target in divisor_set:
        return [target]
    
    # k=2
    for i in range(num_divs):
        a = divisors[i]
        if a >= target:
            continue
        b = target - a
        if b in divisor_set and b != a:
            return [a, b]
    
    # k=3
    for i in range(num_divs):
        a = divisors[i]
        if a >= target:
            continue
        rem2 = target - a
        for j in range(i + 1, num_divs):
            b = divisors[j]
            if b >= rem2:
                continue
            c = rem2 - b
            if c in divisor_set and c != a and c != b:
                return [a, b, c]
        if time.time() > deadline:
            break
    
    # k=4
    for i in range(num_divs):
        a = divisors[i]
        if a >= target:
            continue
        rem3 = target - a
        for j in range(i + 1, num_divs):
            b = divisors[j]
            if b >= rem3:
                continue
            rem2 = rem3 - b
            for k_idx in range(j + 1, num_divs):
                c = divisors[k_idx]
                if c >= rem2:
                    continue
                d = rem2 - c
                if d in divisor_set and d != a and d != b and d != c:
                    return [a, b, c, d]
        if time.time() > deadline:
            break
    
    # k=5
    for i in range(num_divs):
        a = divisors[i]
        if a >= target:
            continue
        rem4 = target - a
        for j in range(i + 1, num_divs):
            b = divisors[j]
            if b >= rem4:
                continue
            rem3 = rem4 - b
            for k_idx in range(j + 1, num_divs):
                c = divisors[k_idx]
                if c >= rem3:
                    continue
                rem2 = rem3 - c
                for l in range(k_idx + 1, num_divs):
                    d = divisors[l]
                    if d >= rem2:
                        continue
                    e = rem2 - d
                    if e in divisor_set and e != a and e != b and e != c and e != d:
                        return [a, b, c, d, e]
        if time.time() > deadline:
            break
    
    # Fallback: greedy solution (always valid)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    assert remainder == 0, f"Greedy failed: remainder={remainder}"
    return chosen
# EVOLVE-BLOCK-END
