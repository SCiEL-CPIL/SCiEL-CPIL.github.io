import time

def construct():
    start = time.time()
    LIMIT = 145.0
    
    def check(a):
        if 0 not in a or len(a) > 32:
            return False
        diffs = set()
        for i in range(len(a)):
            for j in range(i+1, len(a)):
                d = a[j] - a[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))
    
    # Greedy init
    a = [0]
    used = set()
    miss = set(range(1, 31))
    for _ in range(32):
        if not miss:
            break
        bx, bc = None, 0
        for x in range(a[-1]+1, 200):
            ok, cv = True, 0
            for y in a:
                d = x - y
                if d in used:
                    ok = False
                    break
                if d in miss:
                    cv += 1
            if ok and cv > bc:
                bc, bx = cv, x
        if bx is None:
            break
        a.append(bx)
        for y in a[:-1]:
            d = bx - y
            used.add(d)
            miss.discard(d)
    best = list(a) if not miss else None
    best_max = max(a) if best else 10**6
    
    def bt(cur, used, mask, lim):
        nonlocal best, best_max
        if time.time() - start > LIMIT:
            return None
        if mask == (1<<30) - 1:
            m = max(cur)
            if m < best_max:
                best_max = m
                best = list(cur)
            return list(cur)
        if len(cur) >= 32:
            return None
        miss_cnt = 30 - bin(mask).count('1')
        rem = 32 - len(cur)
        if rem * len(cur) + rem*(rem-1)//2 < miss_cnt:
            return None
        last = cur[-1]
        cands = []
        for x in range(last+1, lim+1):
            ok = True
            nm = mask
            for y in cur:
                d = x - y
                if d in used:
                    ok = False
                    break
                if 1 <= d <= 30:
                    nm |= 1 << (d-1)
            if not ok:
                continue
            cands.append((bin(nm ^ mask).count('1'), x))
        cands.sort(key=lambda t: (-t[0], t[1]))
        for _, x in cands:
            nu = set(used)
            nm = mask
            for y in cur:
                d = x - y
                nu.add(d)
                if 1 <= d <= 30:
                    nm |= 1 << (d-1)
            r = bt(cur+[x], nu, nm, lim)
            if r:
                return r
        return None
    
    for lim in range(30, best_max):
        if time.time() - start > LIMIT:
            break
        r = bt([0], set(), 0, lim)
        if r and check(r):
            return r
    
    if best and check(best):
        return best
    return [0]
