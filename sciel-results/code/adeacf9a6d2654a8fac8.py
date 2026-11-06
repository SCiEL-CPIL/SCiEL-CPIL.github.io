"""C4-free construction: polarity-graph seed + bitmask local search + fast C4 check."""
import time
import random
import numpy as np


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _prime_power(n):
    """Largest prime power <= n (n>=2)."""
    if n < 2:
        return 1
    best = 1
    for p in range(2, n + 1):
        if _is_prime(p):
            pk = p
            while pk <= n:
                best = pk
                pk *= p
    return best


def _polarity_graph(q, n):
    """Build a C4-free polarity graph on n vertices.

    Uses the standard polarity of the projective plane of order q.
    A point is adjacent to a point p' iff the dot product is 0 (self-adjacency
    is excluded). We take the first n points.
    """
    if q == 2:
        gmul = lambda a, b: (a & b)
        gadd = lambda a, b: (a ^ b)
        ginv = lambda a: a
    else:
        if _is_prime(q):
            gmul = lambda a, b: (a * b) % q
            gadd = lambda a, b: (a + b) % q
            ginv = lambda a: pow(a, q - 2, q)
        else:
            # Fallback for prime powers: we'll just return a random graph seed instead
            return None

    # Enumerate points: 1-dim subspaces of GF(q)^3
    def all_points():
        reps = []
        seen = set()
        for z in range(q):
            for y in range(q):
                for x in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        invx = ginv(x)
                        rep = (1, gmul(y, invx), gmul(z, invx))
                    elif y != 0:
                        invy = ginv(y)
                        rep = (0, 1, gmul(z, invy))
                    else:
                        rep = (0, 0, 1)
                    if rep not in seen:
                        seen.add(rep)
                        reps.append(rep)
        return reps

    pts = all_points()
    if n > len(pts):
        n = len(pts)
    sel = pts[:n]

    adj = [0] * n
    for i in range(n):
        x, y, z = sel[i]
        for j in range(i + 1, n):
            x2, y2, z2 = sel[j]
            s = gadd(gadd(gmul(x, x2), gmul(y, y2)), gmul(z, z2))
            if s == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _is_c4_free_fast(adj, n):
    """Fast C4 check via (A^2)[i][j] for i!=j: count common neighbors."""
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        a = adj[i]
        tmp = a
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            A[i][idx] = 1
            tmp ^= lsb
    A2 = A @ A
    np.fill_diagonal(A2, 0)
    return not (A2 >= 2).any()


def _is_c4_free(adj, n):
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai & adj[j]).bit_count() >= 2:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = 145

    def elapsed():
        return time.time() - start_time

    def build_edges(adj, n):
        edges = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    edges.append([i, j])
        return edges

    def min_deg(adj, n):
        m = n
        for i in range(n):
            d = adj[i].bit_count()
            if d < m:
                m = d
        return m

    def local_search(adj, n, target, budget):
        """Add edges to raise min degree while preserving C4-freeness."""
        degs = [adj[i].bit_count() for i in range(n)]
        mask_all = (1 << n) - 1
        for _ in range(300):
            if elapsed() > budget:
                break
            if min(degs) >= target:
                break
            progressed = False
            low = [i for i in range(n) if degs[i] <= min(degs) + 1]
            random.shuffle(low)
            for u in low:
                if degs[u] >= target:
                    continue
                non = (~adj[u]) & mask_all
                non &= ~(1 << u)
                if not non:
                    continue
                tmp = non
                cands = []
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    cands.append(idx)
                    tmp ^= lsb
                random.shuffle(cands)
                for v in cands:
                    if (adj[u] & adj[v]).bit_count() < 2:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degs[u] += 1
                        degs[v] += 1
                        progressed = True
                        if degs[u] >= target:
                            break
                if degs[u] >= target:
                    continue
            if not progressed:
                break
        return adj

    def edge_repair(adj, n, max_iters=100):
        """Remove edges to break C4s, choosing edges whose removal causes least degree damage."""
        degs = [adj[i].bit_count() for i in range(n)]
        for _ in range(max_iters):
            if elapsed() > deadline - 2:
                break
            # Find a C4
            found = False
            for i in range(n):
                ai = adj[i]
                for j in range(i + 1, n):
                    common = ai & adj[j]
                    if common.bit_count() >= 2:
                        # Collect two common neighbors
                        c_list = []
                        tmp = common
                        while tmp and len(c_list) < 2:
                            lsb = tmp & -tmp
                            idx = lsb.bit_length() - 1
                            c_list.append(idx)
                            tmp ^= lsb
                        if len(c_list) < 2:
                            continue
                        k, l = c_list[0], c_list[1]
                        # C4: i-k-j-l-i. Remove one of the 4 edges.
                        # Choose the edge whose endpoints have highest degrees (least relative damage)
                        candidates = [(i, k), (k, j), (j, l), (l, i)]
                        # Score: sum of degrees (higher = safer to remove)
                        candidates.sort(key=lambda ab: degs[ab[0]] + degs[ab[1]], reverse=True)
                        a, b = candidates[0]
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        degs[a] -= 1
                        degs[b] -= 1
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return adj

    def try_n(n, budget):
        target = n - 39
        if target < 1:
            target = 1
        best = None
        best_md = -1

        # Seed 1: polarity graph
        try:
            pg = _polarity_graph(_prime_power(n - 1), n)
            if pg is not None and _is_c4_free_fast(pg, n):
                md = min_deg(pg, n)
                if md > best_md:
                    best_md = md
                    best = pg[:]
                if md < target and elapsed() < budget:
                    ls = local_search(pg, n, target, budget)
                    if _is_c4_free(ls, n):
                        md2 = min_deg(ls, n)
                        if md2 > best_md:
                            best_md = md2
                            best = ls[:]
                        if md2 >= target:
                            return n, build_edges(ls, n)
        except Exception:
            pass

        # Seed 2+: random graphs with repair + local search
        ps = [target / n, (target + 1) / n, (target + 2) / n, 0.55]
        ps = [p for p in ps if 0 < p < 0.9]
        if not ps:
            ps = [0.5]

        for p in ps:
            for attempt in range(5):
                if elapsed() > budget:
                    break
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)

                # Repair: remove edges to break C4s
                adj = edge_repair(adj, n, max_iters=200)

                if not _is_c4_free(adj, n):
                    continue

                md = min_deg(adj, n)
                if md > best_md:
                    best_md = md
                    best = adj[:]

                if md < target and elapsed() < budget:
                    ls = local_search(adj, n, target, budget)
                    if _is_c4_free(ls, n):
                        md2 = min_deg(ls, n)
                        if md2 > best_md:
                            best_md = md2
                            best = ls[:]
                        if md2 >= target:
                            return n, build_edges(ls, n)

        if best is not None and best_md >= target:
            return n, build_edges(best, n)
        return None

    # Try from largest n down
    ns = [47, 46, 45, 44, 43, 42, 41, 40]
    weights = [1.5, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
    total_w = sum(weights)
    alloc = [int((deadline - 2) * w / total_w) for w in weights]
    alloc = [max(5, a) for a in alloc]
    s = sum(alloc)
    if s > deadline - 2:
        scale = (deadline - 2) / s
        alloc = [max(3, int(a * scale)) for a in alloc]

    for n, b in zip(ns, alloc):
        if elapsed() > deadline - 1:
            break
        res = try_n(n, deadline - 1)
        if res is not None:
            return {'n': res[0], 'edges': res[1]}

    return {'n': 39, 'edges': []}
