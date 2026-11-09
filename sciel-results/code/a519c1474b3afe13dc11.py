import time, math, random
from collections import defaultdict

P = {'n': 50, 'edges': [[0,1],[0,7],[0,43],[0,49],[1,2],[1,8],[1,44],[2,3],[2,9],[2,45],[3,4],[3,10],[3,46],[4,5],[4,11],[4,47],[5,6],[5,12],[5,48],[6,7],[6,13],[6,49],[7,8],[7,14],[8,9],[8,15],[9,10],[9,16],[10,11],[10,17],[11,12],[11,18],[12,13],[12,19],[13,14],[13,20],[14,15],[14,21],[15,16],[15,22],[16,17],[16,23],[17,18],[17,24],[18,19],[18,25],[19,20],[19,26],[20,21],[20,27],[21,22],[21,28],[22,23],[22,29],[23,24],[23,30],[24,25],[24,31],[25,26],[25,32],[26,27],[26,33],[27,28],[27,34],[28,29],[28,35],[29,30],[29,36],[30,31],[30,37],[31,32],[31,38],[32,33],[32,39],[33,34],[33,40],[34,35],[34,41],[35,36],[35,42],[36,37],[36,43],[37,38],[37,44],[38,39],[38,45],[39,40],[39,46],[40,41],[40,47],[41,42],[41,48],[42,43],[42,49],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49]]}

n = P['n']
edges = P['edges']
adj = [set() for _ in range(n)]
for u, v in edges:
    adj[u].add(v); adj[v].add(u)

def _eval(c):
    improper = 0
    pair_edges = defaultdict(list)
    for u, v in edges:
        a, b = c[u], c[v]
        if a == b:
            improper += 1
        else:
            if a < b:
                pair_edges[(a, b)].append((u, v))
            else:
                pair_edges[(b, a)].append((u, v))
    cyc = 0
    for (a, b), el in pair_edges.items():
        V = 0
        for x in range(n):
            if c[x] == a or c[x] == b:
                V += 1
        E = len(el)
        parent = list(range(n))
        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in el:
            ru, rv = f(u), f(v)
            if ru != rv:
                parent[ru] = rv
        roots = set()
        for x in range(n):
            if c[x] == a or c[x] == b:
                roots.add(f(x))
        cyc += E - V + len(roots)
    return cyc, improper

def _polish(c, k, passes=300):
    for _ in range(passes):
        cyc, imp = _eval(c)
        base = cyc + 1000 * imp
        improved = False
        for v in range(n):
            old = c[v]
            for new in range(k):
                if new == old:
                    continue
                c[v] = new
                cyc2, imp2 = _eval(c)
                nc = cyc2 + 1000 * imp2
                if nc < base:
                    base = nc
                    c[v] = new
                    improved = True
                    break
                else:
                    c[v] = old
        if not improved:
            break
    return c

def _sa(k, budget, init, seed):
    random.seed(seed)
    t0 = time.time()
    c = init[:]
    cyc, imp = _eval(c)
    curc = cyc + 1000 * imp
    best = c[:]; bestc = curc
    T = 3.0; Tmin = 1e-4
    while time.time() - t0 < budget:
        v = random.randrange(n)
        old = c[v]
        new = random.randrange(k)
        if new == old:
            continue
        c[v] = new
        cyc, imp = _eval(c)
        nc = cyc + 1000 * imp
        d = nc - curc
        if d <= 0 or random.random() < math.exp(-d / T):
            curc = nc
            if nc < bestc:
                bestc = nc; best = c[:]
        else:
            c[v] = old
        T = max(Tmin, T * 0.9998)
    return best, bestc

def construct():
    t0 = time.time()
    deadline = t0 + 80
    best_overall = None
    best_distinct = n + 1
    ks = [4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 30, 50]
    base = {4: 26, 5: 16, 6: 10, 7: 8, 8: 5, 9: 3, 10: 2, 12: 1.5, 16: 1, 20: 1, 30: 1, 50: 1}
    for k in ks:
        if time.time() > deadline - 4:
            break
        remaining = deadline - 4 - time.time()
        if remaining <= 0:
            break
        budget = min(base.get(k, 1.0), remaining)
        if budget < 0.4:
            budget = remaining
        t_start = time.time()
        r = 0
        found = None
        while time.time() - t_start < budget and found is None:
            sub = min(2.5, budget - (time.time() - t_start))
            if sub <= 0.05:
                break
            if r == 0:
                init = [i % k for i in range(n)]
            else:
                init = [random.randrange(k) for _ in range(n)]
            bb, bc = _sa(k, sub, init, 100000 + k * 1000 + r)
            bb = _polish(bb, k)
            cyc, imp = _eval(bb)
            if cyc == 0 and imp == 0:
                found = bb
            r += 1
        if found is not None:
            d = len(set(found))
            if d < best_distinct:
                best_distinct = d
                best_overall = found
            if best_distinct == 4:
                break
    if best_overall is None:
        best_overall = list(range(n))
    return [int(x) for x in best_overall]
