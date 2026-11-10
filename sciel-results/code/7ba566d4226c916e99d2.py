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

    log_best_max = math.log(best_max[0]) if best_max[0] > 1 else 0
    nodes = [0]

    # Precompute log of each divisor for fast pruning
    div_logs = [math.log(d) for d in divisors]

    def dfs(idx, remaining, cur):
        if time.time() > deadline:
            return
        nodes[0] += 1
        if remaining == 1:
            if len(cur) <= 12 and len(cur) >= 1:
                m = max(cur)
                if m < best_max[0]:
                    best_max[0] = m
                    best[0] = list(cur)
                    nonlocal log_best_max
                    log_best_max = math.log(m) if m > 1 else 0
            return
        # remaining_slots after this pick
        for i in range(idx, nd):
            f = divisors[i]
            if f > best_max[0]:
                break
            if f <= (cur[-1] if cur else 12):
                continue
            if len(cur) >= 12:
                break
            if remaining % f != 0:
                continue
            R = remaining // f
            # prune: R must be factorable into factors > f, <= best_max
            if R > 1 and R <= f:
                continue
            slots = 12 - len(cur) - 1
            if slots < 0:
                break
            # Tighter prune: minimum possible product of 'slots' factors each > f
            # Each factor must be at least f+1 (but also a divisor of N)
            # Use log: log(R) <= slots * log(best_max)
            if R > 1 and math.log(R) > slots * log_best_max + 1e-9:
                continue
            # Additional prune: if slots == 0, R must be 1
            if slots == 0 and R > 1:
                continue
            # If slots >= 1, R must be > f (since next factor > f)
            if slots >= 1 and R <= f:
                continue
            cur.append(f)
            dfs(i+1, R, cur)
            cur.pop()

    dfs(0, N, [])

    if best[0] is not None:
        return best[0]
    return [N]
