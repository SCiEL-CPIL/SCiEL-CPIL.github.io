import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            v = _lcm(i, j)
            lcm_table[i][j] = v
            lcm_table[j][i] = v
    
    def is_valid(x, sel, lcm_table):
        L = len(sel)
        for i in range(L):
            a = sel[i]
            for j in range(i + 1, L):
                b = sel[j]
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    return False
        return True
    
    # Build prime power set
    pp = set()
    for x in range(2, n + 1):
        is_pp = False
        p = 2
        while p * p <= x:
            if x % p == 0:
                t = x
                while t % p == 0:
                    t //= p
                if t == 1:
                    is_pp = True
                    break
            p += 1
        if not is_pp and x > 1:
            is_pp = True
        if is_pp:
            pp.add(x)
    
    best = list(pp)
    best_set = set(best)
    
    # Greedy extension
    for x in range(1, n + 1):
        if x not in best_set and is_valid(x, best, lcm_table):
            best.append(x)
            best_set.add(x)
    
    # Local improvement: try adding any missing element
    if time.time() < deadline - 10:
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for x in range(1, n + 1):
                if x not in best_set and is_valid(x, best, lcm_table):
                    best.append(x)
                    best_set.add(x)
                    changed = True
                    break
    
    return best
