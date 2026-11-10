import time
import math
from math import factorial

def construct():
    deadline = time.time() + 55.0
    N = factorial(12)  # 479001600

    # divisors of N = 2^10 * 3^5 * 5^2 * 7 * 11
    primes_exp = [(2,10),(3,5),(5,2),(7,1),(11,1)]
    divisors = [1]
    for p, e in primes_exp:
        new = []
        pw = 1
        for _ in range(e+1):
            for d in divisors:
                new.append(d*pw)
            pw *= p
    divisors = sorted(set(d for d in divisors if d > 12))
    nd = len(divisors)

    best = [None]
    best_max = [N]

    # greedy init - try to pick factors that keep remaining balanced
    def greedy():
        cur = []
        rem = N
        last = 12
        while rem > 1:
            found = False
            # Try to find a factor that makes rem/f as small as possible while > last
            best_f = None
            best_rem = None
            for f in divisors:
                if f <= last:
                    continue
                if rem % f == 0:
                    R = rem // f
                    if R == 1:
                        cur.append(f)
                        rem = 1
                        found = True
                        break
                    if R <= last:
                        continue
                    if best_f is None or R < best_rem:
                        best_f = f
                        best_rem = R
            if not found:
                if best_f is not None:
                    cur.append(best_f)
                    rem = best_rem
                    last = best_f
                else:
                    return None
                continue
            if rem == 1:
                break
            if len(cur) >= 12:
                return None
        if rem == 1 and len(cur) <= 12 and len(cur) >= 1:
            return cur
        return None
    g = greedy()
    if g is not None:
        best[0] = g
        best_max[0] = max(g)

    import sys
    sys.setrecursionlimit(100000)

    def dfs(idx, remaining, cur, last_factor):
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if len(cur) <= 12:
                m = max(cur) if cur else 1
                if m < best_max[0]:
                    best_max[0] = m
                    best[0] = list(cur)
            return

        if len(cur) >= 12:
            return

        if remaining <= last_factor:
            return

        for i in range(idx, nd):
            f = divisors[i]
            if f <= last_factor:
                continue
            if f > best_max[0]:
                break
            
            if remaining % f != 0:
                continue
            
            R = remaining // f
            new_last = f
            new_len = len(cur) + 1
            slots = 12 - new_len
            
            if R == 1:
                if new_len <= 12:
                    m = max(cur + [f])
                    if m < best_max[0]:
                        best_max[0] = m
                        best[0] = cur + [f]
                continue

            if slots < 0:
                continue
                
            # Prune 1: Minimum product of 'slots' distinct integers > f
            if slots > 0:
                min_prod = 1
                for j in range(slots):
                    min_prod *= (f + 1 + j)
                    if min_prod > R:
                        break
                if min_prod > R:
                    continue
                
                # Prune 2: Upper bound on R
                if math.log(R) > slots * math.log(best_max[0]) + 1e-9:
                    continue

            cur.append(f)
            dfs(i+1, R, cur, new_last)
            cur.pop()

    dfs(0, N, [], 12)

    if best[0] is not None:
        return best[0]
    return [N]
