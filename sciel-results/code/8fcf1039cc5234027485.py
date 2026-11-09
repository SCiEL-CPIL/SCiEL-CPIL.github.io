import time

def construct():
    from random import Random
    import numpy as np
    from scipy.optimize import linprog

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_lp(edges, n, deadline):
        m = len(edges)
        if m == 0:
            return []
        c = -np.ones(2 * m)
        A_rows = []
        b_vals = []
        in_adj = {i: [] for i in range(n)}
        out_adj = {i: [] for i in range(n)}
        for i, (u, v) in enumerate(edges):
            arc_uv = 2 * i
            arc_vu = 2 * i + 1
            in_adj[v].append(arc_uv)
            in_adj[u].append(arc_vu)
            out_adj[u].append(arc_uv)
            out_adj[v].append(arc_vu)
        for v in range(n):
            row = np.zeros(2 * m)
            for arc_idx in in_adj[v]:
                row[arc_idx] = 1
            A_rows.append(row)
            b_vals.append(1)
        for v in range(n):
            row = np.zeros(2 * m)
            for arc_idx in out_adj[v]:
                row[arc_idx] = 1
            A_rows.append(row)
            b_vals.append(1)
        for i in range(m):
            row = np.zeros(2 * m)
            row[2 * i] = 1
            row[2 * i + 1] = 1
            A_rows.append(row)
            b_vals.append(1)
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        bounds = [(0, 1)] * (2 * m)
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not result.success:
            return [[u, v] for (u, v) in edges]
        x = result.x
        adj = {i: [] for i in range(n)}
        for i, (u, v) in enumerate(edges):
            if x[2 * i] > 0.5:
                adj[u].append(v)
            if x[2 * i + 1] > 0.5:
                adj[v].append(u)
        visited = [False] * n
        cycles = []
        for start in range(n):
            if visited[start] or not adj[start]:
                continue
            path = [start]
            visited[start] = True
            cur = start
            while True:
                if not adj[cur]:
                    break
                nxt = adj[cur][0]
                if nxt == start:
                    if len(path) >= 3:
                        cycles.append(list(path))
                    break
                elif visited[nxt]:
                    break
                else:
                    path.append(nxt)
                    visited[nxt] = True
                    cur = nxt
        used_edges = set()
        for cyc in cycles:
            k = len(cyc)
            for i in range(k):
                a = cyc[i]
                b = cyc[(i + 1) % k]
                used_edges.add((min(a, b), max(a, b)))
        pieces = []
        for cyc in cycles:
            pieces.append(cyc)
        for (u, v) in edges:
            if (u, v) not in used_edges:
                pieces.append([u, v])
        return pieces

    def decompose_greedy(edges, n, deadline):
        remaining = set()
        for (u, v) in edges:
            remaining.add((min(u, v), max(u, v)))
        pieces = []

        def build_cur_adj():
            cur_adj = {i: [] for i in range(n)}
            for (u, v) in remaining:
                cur_adj[u].append(v)
                cur_adj[v].append(u)
            for i in range(n):
                cur_adj[i].sort()
            return cur_adj

        def find_longest_cycle():
            cur_adj = build_cur_adj()
            best_cycle = None
            best_len = 0
            for start in range(n):
                if not cur_adj[start]:
                    continue
                visited = [False] * n
                in_stack = [False] * n
                depth = [-1] * n
                parent = [-1] * n
                cycles_found = []

                def dfs(u):
                    visited[u] = True
                    in_stack[u] = True
                    for v in cur_adj[u]:
                        if not visited[v]:
                            parent[v] = u
                            depth[v] = depth[u] + 1
                            dfs(v)
                        elif in_stack[v]:
                            if depth[u] - depth[v] >= 2:
                                cycle = []
                                cur = u
                                while cur != v:
                                    cycle.append(cur)
                                    cur = parent[cur]
                                cycle.append(v)
                                cycle.reverse()
                                cycles_found.append(cycle)
                    in_stack[u] = False

                depth[start] = 0
                dfs(start)
                for cyc in cycles_found:
                    L = len(cyc)
                    if L > best_len:
                        best_len = L
                        best_cycle = cyc
            return best_cycle

        while True:
            if time.time() > deadline:
                break
            cyc = find_longest_cycle()
            if cyc is None:
                break
            k = len(cyc)
            if k < 3 or len(set(cyc)) != k:
                break
            ok = True
            for i in range(k):
                a = cyc[i]
                b = cyc[(i + 1) % k]
                e = (min(a, b), max(a, b))
                if e not in remaining:
                    ok = False
                    break
            if not ok:
                break
            for i in range(k):
                a = cyc[i]
                b = cyc[(i + 1) % k]
                e = (min(a, b), max(a, b))
                remaining.discard(e)
            pieces.append(list(cyc))

        for (u, v) in sorted(remaining):
            pieces.append([u, v])
        return pieces

    results = []
    deadline = time.time() + 140

    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            n = spec[0]
            edges = instance(*spec)
            results.append([[u, v] for (u, v) in edges])
            continue
        n = spec[0]
        edges = instance(*spec)
        try:
            pieces = decompose_lp(edges, n, deadline)
            used = set()
            valid = True
            for piece in pieces:
                if len(piece) == 2:
                    e = (min(piece[0], piece[1]), max(piece[0], piece[1]))
                    if e in used:
                        valid = False
                        break
                    used.add(e)
                elif len(piece) >= 3:
                    if len(set(piece)) != len(piece):
                        valid = False
                        break
                    for i in range(len(piece)):
                        a = piece[i]
                        b = piece[(i + 1) % len(piece)]
                        e = (min(a, b), max(a, b))
                        if e in used:
                            valid = False
                            break
                        used.add(e)
                    if not valid:
                        break
            if valid and used == {(min(u, v), max(u, v)) for (u, v) in edges}:
                results.append(pieces)
                continue
        except Exception:
            pass
        pieces = decompose_greedy(edges, n, deadline)
        results.append(pieces)

    return results
