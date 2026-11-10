import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 512

    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    if time.time() > deadline:
        return list(range(1, 10))

    def verify_solution(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                g = gcd(lst[i], lst[j])
                if (lst[i] // g) * lst[j] > n:
                    return False
        return True

    # Find the number M <= 512 with the maximum number of divisors
    max_divs = 0
    best_M = 1
    divs_of_best_M = [1]
    
    for M in range(1, n + 1):
        count = 0
        for k in range(1, int(M**0.5) + 1):
            if M % k == 0:
                count += 2
                if k * k == M:
                    count -= 1
        if count > max_divs:
            max_divs = count
            best_M = M
            # Calculate divisors for best_M later or now
            divs = []
            for k in range(1, int(M**0.5) + 1):
                if M % k == 0:
                    divs.append(k)
                    if k * k != M:
                        divs.append(M // k)
            divs_of_best_M = divs

    # Initial best is the divisors of the most composite number
    best = list(divs_of_best_M)
    best_sz = len(best)
    
    # Quick verification of the divisor set
    if not verify_solution(best):
        # Should not happen, but fallback
        best = [1, 2]
        best_sz = 2

    # Greedy initial solution for a good lower bound (might find something else)
    def greedy_clique(order=None):
        if order is None:
            order = list(range(1, n + 1))
        P = 0
        for i in order:
            P |= (1 << i)
        R = []
        while P:
            best_v = 0
            best_deg = -1
            t = P
            while t:
                v = (t & -t).bit_length() - 1
                d = (adj[v] & P).bit_count()
                if d > best_deg:
                    best_deg = d
                    best_v = v
                t &= t - 1
            R.append(best_v)
            P &= adj[best_v]
        return R

    greedy = greedy_clique()
    if len(greedy) > best_sz:
        best_sz = len(greedy)
        best = list(greedy)

    g2 = greedy_clique(list(range(n, 0, -1)))
    if len(g2) > best_sz:
        best_sz = len(g2)
        best = list(g2)

    def expand(R, P):
        nonlocal best, best_sz
        if time.time() > deadline:
            return
        if len(R) > best_sz:
            best_sz = len(R)
            best = R.copy()
        if P == 0:
            return

        plist = []
        t = P
        while t:
            v = (t & -t).bit_length() - 1
            plist.append(v)
            t &= t - 1

        deg = [(v, (adj[v] & P).bit_count()) for v in plist]
        deg.sort(key=lambda x: -x[1])
        plist = [v for v, d in deg]

        color_classes = []
        for v in plist:
            placed = False
            for ci, cls in enumerate(color_classes):
                ok = True
                for u in cls:
                    if adj[v] & (1 << u):
                        ok = False
                        break
                if ok:
                    cls.append(v)
                    placed = True
                    break
            if not placed:
                color_classes.append([v])

        max_c = len(color_classes)
        if len(R) + max_c <= best_sz:
            return

        for ci in range(max_c, 0, -1):
            if len(R) + ci <= best_sz:
                break
            for v in color_classes[ci - 1]:
                if not (P & (1 << v)):
                    continue
                R.append(v)
                expand(R, P & adj[v])
                R.pop()
                P &= ~(1 << v)
                if time.time() > deadline:
                    return

    P0 = 0
    for i in range(1, n + 1):
        P0 |= (1 << i)

    try:
        expand([], P0)
    except Exception:
        if not best or not verify_solution(best):
            best = [2**i for i in range(10) if 2**i <= 512]

    if not best or not verify_solution(best):
        best = [2**i for i in range(10) if 2**i <= 512]

    res = list(best)
    res_set = set(res)
    ch = True
    while ch and time.time() < deadline - 3:
        ch = False
        for v in range(1, n + 1):
            if v in res_set:
                continue
            if all(adj[v] & (1 << u) for u in res):
                res.append(v)
                res_set.add(v)
                ch = True

    if not verify_solution(res):
        if verify_solution(best):
            return best
        return [1, 2]
    return res
