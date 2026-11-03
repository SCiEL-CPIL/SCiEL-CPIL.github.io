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
    # Proper BFS girth check: for each start s, do BFS up to depth g-1.
    # If any vertex v (v != s) is reached at depth d1 from s and also at
    # depth d2 >= d1 from s (i.e., encountered again), then d1 + d2 is a
    # cycle length. If d1 + d2 < g, fail. Also if we encounter s at depth
    # d < g, that's a cycle of length d < g.
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
        # girth check via BFS from each vertex
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            parent = [-1] * n
            queue = deque([s])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # v already visited; cycle length = du + 1 + dist[v]
                        # but only if v is not the parent of u (backtracking)
                        if v == parent[u]:
                            continue
                        cycle_len = du + 1 + dist[v]
                        if cycle_len < g:
                            return False
        return True

    def try_graph(n, edges, k, g):
        if not validate(n, edges, k, g):
            return None
        return n, edges

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

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

    # --- Fast short-cycle finder: returns the cycle as a list of vertices ---
    def find_short_cycle(adj, g):
        n = len(adj)
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v == parent[u]:
                            continue
                        cycle_len = du + 1 + dist[v]
                        if cycle_len < g:
                            # Reconstruct cycle: path s->u (via parent), edge u-v, path v->s (via parent)
                            # Path from s to u
                            path1 = []
                            cur = u
                            while cur != -1:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.reverse()  # now s -> ... -> u
                            # Path from s to v
                            path2 = []
                            cur = v
                            while cur != -1:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.reverse()  # now s -> ... -> v
                            # Cycle: u -> v -> ... -> s -> ... -> u
                            # path2 gives s->...->v, so v->...->s is reversed path2[1:]
                            # path1 gives s->...->u
                            # cycle = [u, v] + path2[1:][::-1] + path1[1:][::-1]
                            # Actually: cycle goes u -> v (edge), then v back to s via path2 reversed (skip s), then s to u via path1 (skip s)
                            # path2 = [s, ..., v], reversed = [v, ..., s]
                            # path1 = [s, ..., u], reversed = [u, ..., s]
                            # cycle: start at u, go to v (edge u-v), then follow path2 reversed from v to s (excluding s), then follow path1 from s to u (excluding s)
                            # path2[1:][::-1] = nodes from v's predecessor back to s's successor... no
                            # Let me think again:
                            # path2 = [s, n1, n2, ..., v] (dist[v] steps)
                            # To go from v back to s: v, ..., n2, n1, s = path2[::-1]
                            # path1 = [s, m1, m2, ..., u] (dist[u] steps)
                            # To go from s to u: s, m1, ..., u = path1
                            # Cycle: u -> v (edge), v -> ... -> s (path2 reversed), s -> ... -> u (path1)
                            # But we want a simple cycle. The cycle is:
                            # u, v, path2[-2], path2[-3], ..., path2[1], path1[1], path1[2], ..., u
                            # i.e., [u] + [v] + path2[-2::-1] + path1[1:]
                            # Wait, path2 = [s, ..., v], so path2 reversed = [v, ..., s]
                            # We want from v to s: v, path2[-2], path2[-3], ..., path2[1], s
                            # That's path2[::-1] which is [v, ..., s]
                            # Then from s to u: s, path1[1], ..., u = path1
                            # Full cycle: u -> v -> ... -> s -> ... -> u
                            # Nodes: u, v, path2[-2], ..., path2[1], path1[1], ..., u
                            # But u appears at start and end. Let's just return the nodes without the closing u.
                            cycle_nodes = [u, v]
                            # Add path from v back to s (excluding s and v itself which is already there)
                            for node in path2[-2::-1]:
                                cycle_nodes.append(node)
                            # Add path from s to u (excluding s which is already there, and u which closes the cycle)
                            for node in path1[1:-1]:
                                cycle_nodes.append(node)
                            # cycle_nodes should have length cycle_len
                            if len(cycle_nodes) != cycle_len:
                                # Fallback: just return a marker
                                return [u, v] + path2[-2::-1] + path1[1:-1]
                            return cycle_nodes
        return None

    # --- Repair: break short cycles via targeted 2-switches ---
    def repair_girth(n, edges, k, g, max_iters=500):
        adj = build_adj(n, edges)
        adj_set = [set(adj[i]) for i in range(n)]  # for fast membership

        def do_2switch(a, b, c, d):
            """Remove edges (a,b) and (c,d), add edges (a,d) and (c,b)."""
            adj[a].remove(b); adj[b].remove(a)
            adj[c].remove(d); adj[d].remove(c)
            adj[a].append(d); adj[d].append(a)
            adj[c].append(b); adj[b].append(c)
            adj_set[a].discard(b); adj_set[b].discard(a)
            adj_set[c].discard(d); adj_set[d].discard(c)
            adj_set[a].add(d); adj_set[d].add(a)
            adj_set[c].add(b); adj_set[b].add(c)

        for _iter in range(max_iters):
            if check_deadline():
                break
            cycle = find_short_cycle(adj, g)
            if cycle is None:
                break
            L = len(cycle)
            if L < 3:
                break
            found_switch = False
            # Try multiple random 2-switches on this cycle
            for _attempt in range(50):
                i = rng.randint(0, L-1)
                j = rng.randint(0, L-1)
                # Need non-adjacent edges on the cycle
                if abs(i - j) == 1 or abs(i - j) == L - 1:
                    continue
                u1, v1 = cycle[i], cycle[(i+1) % L]
                u2, v2 = cycle[j], cycle[(j+1) % L]
                # Check that new edges don't already exist
                if v2 in adj_set[u1] or v1 in adj_set[u2]:
                    continue
                # Also ensure we're not creating a multi-edge
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                do_2switch(u1, v1, u2, v2)
                found_switch = True
                break
            if not found_switch:
                # Try a random 2-switch anywhere in the graph
                for _ in range(20):
                    a = rng.randint(0, n-1)
                    if len(adj[a]) < 1:
                        continue
                    b = rng.choice(adj[a])
                    c = rng.randint(0, n-1)
                    if c == a or c == b:
                        continue
                    if len(adj[c]) < 1:
                        continue
                    d = rng.choice(adj[c])
                    if d == a or d == b:
                        continue
                    if b in adj_set[c] or d in adj_set[a]:
                        continue
                    if a == c or a == d or b == c or b == d:
                        continue
                    do_2switch(a, b, c, d)
                    found_switch = True
                    break
            if not found_switch:
                break

        # Final check
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
            if best is not None:
                return best
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

    # Try a few LUW parameter variants if time permits
    for q_try in (k, k + 1, k + 2, k + 3):
        if check_deadline():
            break
        try:
            n_l, e_l = luw_family(q=q_try)
            consider(n_l, e_l)
        except Exception:
            pass

    # --- 2. Random regular + repair, focused on feasible n range ---
    if best is not None:
        n_upper = best_n - 1
    else:
        n_upper = 3000

    # Sample n values: dense near moore, sparser further out
    n_samples = []
    n = max(moore + 1, 500)
    step = 15
    while n < n_upper and not check_deadline():
        n_samples.append(n)
        n += step
        if step < 80:
            step += 5
        elif step < 250:
            step += 15
        else:
            step += 40

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
        for _restart in range(3):
            if check_deadline():
                break
            res = generate_random_regular(n, k, rng)
            if res is None:
                continue
            n_gen, edges_gen = res
            repaired = repair_girth(n_gen, edges_gen, k, g, max_iters=400)
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
