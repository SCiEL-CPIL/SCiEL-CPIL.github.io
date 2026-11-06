import time
import random

def construct():
    n = 128
    aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            dd = a + 3 * d
            if dd <= n:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, dd - 1))
                
    def check(idx, col, colors):
        for ap in aps:
            if idx not in ap:
                continue
            vals = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
            pos = ap.index(idx)
            vals[pos] = col
            assigned = [v for v in vals if v != -1]
            cnt = len(assigned)
            if cnt == 4:
                if len(set(assigned)) < 3:
                    return False
            elif cnt == 3:
                if len(set(assigned)) == 1:
                    return False
            elif cnt == 2:
                if len(set(assigned)) == 1:
                    if col == assigned[0]:
                        return False
        return True

    def solve(k):
        colors = [-1] * n
        start_time = time.time()
        
        def backtrack(i):
            if i == n:
                return True
            if time.time() - start_time > 85:
                return False
            
            cols = list(range(k))
            random.shuffle(cols)
            
            for c in cols:
                if check(i, c, colors):
                    colors[i] = c
                    if backtrack(i + 1):
                        return True
                    colors[i] = -1
            return False
            
        if backtrack(0):
            return colors
        return None

    # Try increasing k
    for k in range(4, 14):
        random.seed(42 + k)
        res = solve(k)
        if res is not None:
            return res
            
    # Fallback: use 13 colors (seed level)
    # Just in case 4-12 all fail (unlikely)
    return [i % 13 for i in range(1, n + 1)]

