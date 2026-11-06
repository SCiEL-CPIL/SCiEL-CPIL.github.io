import time
import random

def construct():
    n = 16
    target = 65
    deadline = time.time() + 110.0
    rng = random.Random()

    # ---- exact triangle enumeration via bitmasks (fast) ----
    def enumerate_triangles(m):
        tris = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (m[i] >> j) & 1:
                    continue
                common = m[i] & m[j]
                k = j + 1
                while common:
                    if common & 1:
                        if k > j:
                            tris.append((i, j, k))
                    common >>= 1
                    k += 1
        return tris

    def score_of(tris, m):
        best = 0
        for (i, j, k) in tris:
            s = ((m[i] & 1) * 0)  # placeholder
            deg_i = bin(m[i]).count('1')
            deg_j = bin(m[j]).count('1')
            deg_k = bin(m[k]).count('1')
            if deg_i + deg_j + deg_k > best:
                best = deg_i + deg_j + deg_k
        return best

    # ---- incremental flip scoring ----
    def flip_score(m, tris, edge, add):
        a, b = edge
        if add:
            if not ((m[a] >> b) & 1):
                deg = [bin(m[i]).count('1') for i in range(n)]
                # triangles that become present: {a,b,c} where c in common
                common = m[a] & m[b]
                new_tris = []
                c = 0
                while common:
                    if common & 1:
                        new_tris.append((a, b, c))
                    common >>= 1
                    c += 1
                # recompute max among new triangles and existing
                # existing triangles keep same degree sums (degrees unchanged for them)
                # but we need max over all triangles = max(existing_max, new_max)
                # existing_max we don't have stored; recompute is expensive.
                # Instead compute full max quickly:
                best = 0
                for (i, j, k) in tris:
                    s = bin(m[i]).count('1') + bin(m[j]).count('1') + bin(m[k]).count('1')
                    if s > best:
                        best = s
                for (i, j, k) in new_tris:
                    s = bin(m[i]).count('1') + bin(m[j]).count('1') + bin(m[k]).count('1')
                    if s > best:
                        best = s
                return best
        else:
            if (m[a] >> b) & 1:
                best = 0
                for (i, j, k) in tris:
                    if (i == a and j == b) or (i == b and j == a):
                        continue
                    s = bin(m[i]).count('1') + bin(m[j]).count('1') + bin(m[k]).count('1')
                    if s > best:
                        best = s
                return best
        return None

    # ---- helpers ----
    def random_graph():
        m = [0] * n
        ec = 0
        while ec < target:
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v or ((m[u] >> v) & 1):
                continue
            m[u] |= (1 << v)
            m[v] |= (1 << u)
            ec += 1
        return m

    def count_edges(m):
        c = 0
        for i in range(n):
            c += bin(m[i]).count('1')
        return c // 2

    def best_flip(m, tris, budget):
        """Find best single-edge flip (add or remove) that keeps edge count = target."""
        ec = count_edges(m)
        cur = score_of(tris, m)
        best = (cur, None, None)  # (score, edge, add)
        if ec < target:
            # need to add an edge
            for a in range(n):
                for b in range(a + 1, n):
                    if (m[a] >> b) & 1:
                        continue
                    s = flip_score(m, tris, (a, b), True)
                    if s is not None and s < best[0]:
                        best = (s, (a, b), True)
        elif ec > target:
            # need to remove an edge
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((m[a] >> b) & 1):
                        continue
                    s = flip_score(m, tris, (a, b), False)
                    if s is not None and s < best[0]:
                        best = (s, (a, b), False)
        return best

    def apply_flip(m, edge, add):
        a, b = edge
        if add:
            m[a] |= (1 << b)
            m[b] |= (1 << a)
        else:
            m[a] &= ~(1 << b)
            m[b] &= ~(1 << a)

    # ---- main search ----
    best_m = random_graph()
    best_tris = enumerate_triangles(best_m)
    best_score = score_of(best_tris, best_m)

    while time.time() < deadline:
        improved = False
        while time.time() < deadline:
            tris = enumerate_triangles(best_m)
            res = best_flip(best_m, tris, deadline)
            if res[1] is None:
                break
            s, edge, add = res
            if s >= best_score:
                break
            apply_flip(best_m, edge, add)
            best_score = s
            improved = True
        if not improved:
            # random restart
            cand = random_graph()
            ct = enumerate_triangles(cand)
            cs = score_of(ct, cand)
            if cs < best_score:
                best_m = cand
                best_score = cs
            # also try to improve this restart a bit
            for _ in range(30):
                if time.time() >= deadline:
                    break
                tris = enumerate_triangles(best_m)
                res = best_flip(best_m, tris, deadline)
                if res[1] is None:
                    break
                s, edge, add = res
                if s >= best_score:
                    break
                apply_flip(best_m, edge, add)
                best_score = s

    # ---- convert to edge list ----
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_m[i] >> j) & 1:
                edges.append([i, j])
    return {'n': 16, 'edges': edges}
