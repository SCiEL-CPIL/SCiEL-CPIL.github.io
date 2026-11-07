"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    start = time.time()
    dl = start + 80

    def valid_complete(col):
        for u, v in edges:
            if col[u] == col[v]:
                return False
        k = max(col) + 1
        for a in range(k):
            for b in range(a + 1, k):
                vs = [i for i in range(n) if col[i] in (a, b)]
                if len(vs) <= 1:
                    continue
                vs_set = set(vs)
                seen = set()
                for s in vs:
                    if s in seen:
                        continue
                    cnt_v = 0
                    cnt_e = 0
                    q = deque([s])
                    seen.add(s)
                    while q:
                        cur = q.popleft()
                        cnt_v += 1
                        for nb in adj[cur]:
                            if nb in vs_set:
                                if cur < nb:
                                    cnt_e += 1
                                if nb not in seen:
                                    seen.add(nb)
                                    q.append(nb)
                    if cnt_e >= cnt_v:
                        return False
        return True

    def solve(k):
        col = [-1] * n

        def ok(v, c):
            for u in adj[v]:
                if col[u] == c:
                    return False
            for d in range(k):
                if d == c:
                    continue
                nd = [u for u in adj[v] if col[u] == d]
                if len(nd) <= 1:
                    continue
                seen = set()
                q = deque([nd[0]])
                seen.add(nd[0])
                while q:
                    cur = q.popleft()
                    for nb in adj[cur]:
                        if nb == v or nb in seen:
                            continue
                        if col[nb] == c or col[nb] == d:
                            seen.add(nb)
                            q.append(nb)
                for u in nd[1:]:
                    if u in seen:
                        return False
            return True

        def bt(i):
            if time.time() > dl:
                return False
            if i == n:
                return True
            v = i
            for c in range(k):
                if not ok(v, c):
                    continue
                col[v] = c
                if bt(i + 1):
                    return True
                col[v] = -1
            return False

        return list(col) if bt(0) else None

    for k in range(2, 10):
        if time.time() > dl:
            break
        r = solve(k)
        if r is not None and valid_complete(r):
            return [int(x) for x in r]

    return list(range(n))

# EVOLVE-BLOCK-END
