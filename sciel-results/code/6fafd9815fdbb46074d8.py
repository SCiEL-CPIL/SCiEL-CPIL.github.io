import time
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()  # ascending
    div_set = set(divisors)
    m = len(divisors)
    
    # Try increasing number of divisors
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
        if k == 1:
            if target in div_set:
                return [target]
        elif k == 2:
            for i in range(m):
                d1 = divisors[i]
                if d1 >= target:
                    break
                rem = target - d1
                if rem in div_set and rem != d1:
                    return [d1, rem]
        elif k == 3:
            for i in range(m):
                d1 = divisors[i]
                if d1 >= target:
                    break
                rem2 = target - d1
                for j in range(i + 1, m):
                    d2 = divisors[j]
                    if d2 >= rem2:
                        break
                    d3 = rem2 - d2
                    if d3 <= d2:
                        continue
                    if d3 in div_set:
                        return [d1, d2, d3]
        elif k == 4:
            # Meet-in-the-middle: group pairs by sum
            pair_by_sum = defaultdict(list)
            for i in range(m):
                for j in range(i + 1, m):
                    s = divisors[i] + divisors[j]
                    if s < target:
                        pair_by_sum[s].append((i, j))
            
            found = None
            for s1 in pair_by_sum:
                s2 = target - s1
                if s2 not in pair_by_sum:
                    continue
                if s1 > s2:
                    continue
                for (i, j) in pair_by_sum[s1]:
                    for (k_idx, l_idx) in pair_by_sum[s2]:
                        if len({i, j, k_idx, l_idx}) == 4:
                            found = [divisors[i], divisors[j], divisors[k_idx], divisors[l_idx]]
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                return found
        else:
            found = None
            for combo in combinations(divisors, k):
                if sum(combo) == target:
                    found = list(combo)
                    break
                if time.time() > deadline:
                    break
            if found:
                return found
    
    # Fallback: greedy
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    assert remainder == 0, f"Could not reach target, remainder={remainder}"
    return chosen
