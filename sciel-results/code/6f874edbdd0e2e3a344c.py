"""Long-path search on the pinned G(100,1.8/100) graph (seed 748)."""
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)
DEADLINE = time.time() + 78.0  # safety margin under the 90s wall-clock kill


def _components():
    seen = set()
    comps = []
    for s in range(N):
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in ADJ[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def _greedy_extend(path, seen, mode):
    """Grow path from its tail using a heuristic; return new list (or same)."""
    p = list(path)
    s = set(seen)
    while True:
        cur = p[-1]
        cand = ADJ[cur] - s
        if not cand:
            break
        if mode == 0:  # Warnsdorff: fewest onward options first (avoid traps)
            nxt = min(cand, key=lambda w: (len(ADJ[w] - s), w))
        else:  # high degree first (explore dense area)
            nxt = min(cand, key=lambda w: (-len(ADJ[w]), w))
        p.append(nxt)
        s.add(nxt)
    return p


def _try_insert(path, seen, budget):
    """Insert a single unused vertex between consecutive path vertices if possible."""
    p = list(path)
    s = set(seen)
    used = set(p)
    unused = [v for v in range(N) if v not in used]
    improved = True
    steps = 0
    while improved and steps < budget and time.time() < DEADLINE:
        improved = False
        steps += 1
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            if a == b:
                continue
            for v in unused:
                if v in ADJ[a] and v in ADJ[b] and v not in used:
                    p.insert(i + 1, v)
                    used.add(v)
                    unused.remove(v)
                    improved = True
                    break
            if improved:
                break
    return p


def _dfs_extend(path, seen, depth_limit):
    """Bounded DFS from path tail to find a longer extension; return best path."""
    p = list(path)
    s = set(seen)
    best = list(p)
    best_len = len(p)
    nodes = [0]

    def dfs(cur, depth):
        nodes[0] += 1
        if nodes[0] % 2048 == 0 and time.time() >= DEADLINE:
            return
        if len(p) > best_len:
            best_len = len(p)
            best = list(p)
        if depth >= depth_limit:
            return
        cur = p[-1]
        cand = sorted(ADJ[cur] - s, key=lambda w: (len(ADJ[w] - s), w))
        for nxt in cand:
            p.append(nxt)
            s.add(nxt)
            dfs(nxt, depth + 1)
            p.pop()
            s.discard(nxt)

    dfs(p[-1], 0)
    return best


def construct():
    t0 = time.time()
    comps = _components()
    best = []
    # Work on the largest component (a path can't cross components).
    for comp in comps:
        if time.time() >= DEADLINE:
            break
        cset = set(comp)
        # Try several seeds and greedy modes, plus DFS and insertion.
        seeds = list(comp)
        # order seeds by degree desc for a good quick start
        seeds.sort(key=lambda v: -len(ADJ[v]))
        for start in seeds:
            if time.time() >= DEADLINE:
                break
            for mode in (0, 1):
                if time.time() >= DEADLINE:
                    break
                path = _greedy_extend([start], {start}, mode)
                if len(path) > len(best):
                    best = list(path)
            # DFS extension from a Warnsdorff greedy base
            if time.time() >= DEADLINE:
                break
            base = _greedy_extend([start], {start}, 0)
            d = _dfs_extend(base, set(base), depth_limit=10)
            if len(d) > len(best):
                best = list(d)
            # Try inserting unused vertices into the best-so-far path
            if time.time() >= DEADLINE:
                break
            ins = _try_insert(best, set(best), budget=8)
            if len(ins) > len(best):
                best = list(ins)
        # After scanning seeds, a final global insertion pass on best
        if time.time() >= DEADLINE:
            break
        ins = _try_insert(best, set(best), budget=20)
        if len(ins) > len(best):
            best = list(ins)

    # Final safety: if nothing found (shouldn't happen), return a single vertex.
    if not best:
        best = [comps[0][0]] if comps else [0]
    return best
