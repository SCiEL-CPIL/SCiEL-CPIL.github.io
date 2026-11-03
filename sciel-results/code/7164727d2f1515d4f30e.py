"""Portfolio SEARCH: explicit LUW family + fast girth check + random-regular repair."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys
from collections import deque
import numpy as np


def construct(k, g):
    deadline = time.time() + 112.0  # safety margin before 150s kill
    rng = random.Random(12345)

    def check_deadline():
        return time.time() > deadline

    # --- Fast girth validation: k-regular, simple, girth >= g ---
    def validate(n, edges, k, g):
        if n <= 0 or k <= 0:
            return False
        if n * k % 2 != 0:
            return False
        if k >= n:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            key = (u, v) if u < v else (v, u)
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # girth check
        g_minus_1 = g - 1
        for s in range(n):
            best = np.full(n, -1, dtype=np.int32)
            best[s] = 0
            frontier = adj[s]
            depth = 1
            while frontier and depth <= g_minus_1:
                next_frontier = []
                for u in frontier:
                    for v in adj[u]:
                        if best[v] >= 0:
                            if depth + 1 + best[v] < g:
                                return False
                        else:
                            best[v] = depth
                            next_frontier.append(v)
                frontier = next_frontier
                depth += 1
        return True

    def try_graph(n, edges, k, g):
        if not validate(n, edges, k, g):
            return None
        return n, edges

    # --- Family 1: Lazebnik-Ustimenko-Woldar (reliable baseline) ---
    def luw_family(q=None, d=None):
        if q is None:
            q = k
        if d is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)

        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # --- Generalized hexagon of order (q,q): 5-regular, girth 12, 3(q+1)^2 vertices ---
    def gen_hexagon(q):
        if q == 4:
            add_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            mul_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def gf_add(x, y): return add_tbl[x][y]
            def gf_mul(x, y): return mul_tbl[x][y]
            def gf_neg(x): return x
        else:
            def gf_add(x, y): return (x + y) % q
            def gf_mul(x, y): return (x * y) % q
            def gf_neg(x): return (-x) % q

        def collinear(a, b, c, d, e, f):
            return gf_add(gf_mul(a, e), gf_mul(b, f)) == 0 and \
                   gf_add(gf_mul(c, e), gf_mul(d, f)) == 0

        point_lines = {}
        lines = []
        point_id = {}
        all_points = []
        for side in range(3):
            for x in range(q):
                for y in range(q):
                    pid = len(all_points)
                    all_points.append((side, x, y))
                    point_id[(side, x, y)] = pid
                    point_lines[pid] = []

        for a in range(q):
            for b in range(q):
                for c in range(q):
                    for d in range(q):
                        det = gf_add(gf_mul(a, d), gf_neg(gf_mul(b, c)))
                        if det == 0:
                            continue
                        for e in range(q):
                            for f in range(q):
                                if collinear(a, b, c, d, e, f):
                                    line_id = len(lines)
                                    lines.append([(0, a, b), (1, c, d), (2, e, f)])
                                    for pt in lines[-1]:
                                        point_lines[point_id[pt]].append(line_id)
                                    break
                            else:
                                continue
                            break

        edges = []
        seen_edges = set()
        for pid in range(len(all_points)):
            for lid in point_lines[pid]:
                line_pts = lines[lid]
                for qid_pt in line_pts:
                    qid = point_id[qid_pt]
                    if qid == pid:
                        continue
                    key = (pid, qid) if pid < qid else (qid, pid)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append(key)
        return len(all_points), edges

    # --- Random k-regular graph via configuration model + 2-switches ---
    def generate_random_regular(n, k, rng):
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        for attempt in range(60):
            half_edges = []
            for v in range(n):
                half_edges.extend([v] * k)
            rng.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not valid:
                continue
            has_multi = False
            for u in range(n):
                if len(adj[u]) != len(set(adj[u])):
                    has_multi = True
                    break
            if has_multi:
                continue
            for _ in range(n * 2):
                a = rng.randint(0, n-1)
                b = rng.choice(adj[a])
                c = rng.randint(0, n-1)
                if c == a or c == b:
                    continue
                d = rng.choice(adj[c])
                if d == a or d == b:
                    continue
                if b in adj[c] or d in adj[a]:
                    continue
                adj[a].remove(b); adj[b].remove(a)
                adj[c].remove(d); adj[d].remove(c)
                adj[a].append(d); adj[d].append(a)
                adj[c].append(b); adj[b].append(c)
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        return None

    # --- Fast short-cycle finder (BFS, no path storage) ---
    def find_short_cycle(adj, g):
        n = len(adj)
        g_minus_1 = g - 1
        for s in range(n):
            best = np.full(n, -1, dtype=np.int32)
            best[s] = 0
            frontier = adj[s]
            depth = 1
            while frontier and depth <= g_minus_1:
                next_frontier = []
                for u in frontier:
                    for v in adj[u]:
                        if best[v] >= 0:
                            if depth + 1 + best[v] < g:
                                return _reconstruct_cycle(adj, s, v, depth + 1, best)
                        else:
                            best[v] = depth
                            next_frontier.append(v)
                frontier = next_frontier
                depth += 1
        return None

    def _reconstruct_cycle(adj, s, v, total_len, best):
        d1 = best[v]
        d2 = total_len - 1
        def dfs(start, target, length, forbidden):
            if length == 0:
                return [start] if start == target else None
            for nb in adj[start]:
                if nb == forbidden or nb in path_set:
                    continue
                path_set.add(nb)
                res = dfs(nb, target, length - 1, start)
                path_set.discard(nb)
                if res is not None:
                    return [start] + res
            return None
        path_set = set()
        path_set.add(s)
        p1 = dfs(s, v, d1, -1)
        path_set = set()
        path_set.add(v)
        p2 = dfs(v, s, d2, -1)
        if p1 is None or p2 is None:
            return None
        cycle = p1 + p2[1:]
        return cycle

    # --- Repair: break short cycles via 2-switches ---
    def repair_girth(n, edges, k, g, max_iters=300):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for _ in range(max_iters):
            if check_deadline():
                break
            cycle = find_short_cycle(adj, g)
            if cycle is None:
                break
            L = len(cycle)
            found_switch = False
            for _attempt in range(30):
                i = rng.randint(0, L-1)
                j = rng.randint(0, L-1)
                if i == j or (i + 1) % L == j or (j + 1) % L == i:
                    continue
                u1 = cycle[i]
                v1 = cycle[(i+1) % L]
                u2 = cycle[j]
                v2 = cycle[(j+1) % L]
                if v2 in adj[u1] or v1 in adj[u2]:
                    continue
                adj[u1].remove(v1); adj[v1].remove(u1)
                adj[u2].remove(v2); adj[v2].remove(u2)
                adj[u1].append(v2); adj[v2].append(u1)
                adj[u2].append(v1); adj[v1].append(u2)
                found_switch = True
                break
            if not found_switch:
                a = rng.randint(0, n-1)
                if not adj[a]:
                    continue
                b = rng.choice(adj[a])
                cand_a = [x for x in adj[b] if x != a and x != b and x not in adj[a]]
                cand_b = [x for x in adj[a] if x != b and x != a and x not in adj[b]]
                if not cand_a or not cand_b:
                    for _ in range(10):
                        a2 = rng.randint(0, n-1)
                        b2 = rng.choice(adj[a2])
                        c2 = rng.randint(0, n-1)
                        if c2 == a2 or c2 == b2:
                            continue
                        d2 = rng.choice(adj[c2])
                        if d2 == a2 or d2 == b2:
                            continue
                        if b2 in adj[c2] or d2 in adj[a2]:
                            continue
                        adj[a2].remove(b2); adj[b2].remove(a2)
                        adj[c2].remove(d2); adj[d2].remove(c2)
                        adj[a2].append(d2); adj[d2].append(a2)
                        adj[c2].append(b2); adj[b2].append(c2)
                        found_switch = True
                        break
                    continue
                na = rng.choice(cand_a)
                nb = rng.choice(cand_b)
                if nb in adj[a] or na in adj[b]:
                    continue
                adj[a].remove(b); adj[b].remove(a)
                adj[a].append(na); adj[na].append(a)
                adj[b].append(nb); adj[nb].append(b)
                found_switch = True

        if find_short_cycle(adj, g) is None:
            new_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        new_edges.append((u, v))
            return new_edges
        return None

    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if n < best_n:
            result = try_graph(n, edges, k, g)
            if result is not None:
                best = result
                best_n = n
        if check_deadline():
            raise TimeoutError

    # --- Moore bound ---
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        moore = k * sum((k - 1) ** i for i in range(r))

    # --- 1. LUW portfolio (reliable baseline) ---
    try:
        n_luw, e_luw = luw_family()
        consider(n_luw, e_luw)
    except Exception:
        pass

    for q_try in (k, k + 1, k + 2, k + 3):
        if check_deadline():
            break
        try:
            n_l, e_l = luw_family(q=q_try)
            consider(n_l, e_l)
        except Exception:
            pass

    # --- 1b. Generalized hexagon (k=5, g<=12): 330 vertices, girth 12 ---
    if k == 5 and g <= 12:
        for q_h in (4,):
            if check_deadline():
                break
            try:
                n_h, e_h = gen_hexagon(q_h)
                consider(n_h, e_h)
            except Exception:
                pass

    # --- 2. Random regular + repair ---
    if best is not None:
        n_upper = best_n - 1
    else:
        n_upper = 3000

    n_samples = []
    n = max(moore, 500)
    step = 20
    while n < n_upper and not check_deadline():
        n_samples.append(n)
        n += step
        if step < 100:
            step += 5
        elif step < 300:
            step += 20
        else:
            step += 50

    rng.shuffle(n_samples)
    n_samples.sort()

    for n in n_samples:
        if check_deadline():
            break
        if n >= best_n:
            continue
        if n * k % 2 != 0:
            continue
        if k >= n:
            continue
        for _restart in range(2):
            if check_deadline():
                break
            res = generate_random_regular(n, k, rng)
            if res is None:
                continue
            n_gen, edges_gen = res
            remaining = max(1.0, deadline - time.time())
            max_it = int(50 + 200 * min(1.0, remaining / 60.0))
            repaired = repair_girth(n_gen, edges_gen, k, g, max_iters=max_it)
            if repaired is not None:
                consider(n_gen, repaired)
                if best is not None and best_n == n_gen:
                    break

    if best is not None:
        return best

    # Fallback: LUW
    n, e = luw_family()
    return n, e

# EVOLVE-BLOCK-END
