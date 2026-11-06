import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def verify_and_fix(paths, edges):
        """Ensure paths are simple, use valid edges, edge-disjoint, cover all edges."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        valid_paths = []
        for p in paths:
            if len(p) != len(set(p)):
                seen = set()
                fixed = []
                for v in p:
                    if v in seen:
                        break
                    seen.add(v)
                    fixed.append(v)
                p = fixed
            if len(p) < 2:
                continue
            ok = True
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set or e in used:
                    ok = False
                    break
            if ok:
                for i in range(len(p) - 1):
                    used.add(tuple(sorted((p[i], p[i+1]))))
                valid_paths.append(p)
        for e in edge_set:
            if e not in used:
                valid_paths.append([e[0], e[1]])
                used.add(e)
        return valid_paths

    def merge_paths(paths, edges):
        """Aggressively merge paths that can be concatenated."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        improved = True
        max_iter = 100
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            new_current = []
            used = [False] * len(current)
            for i in range(len(current)):
                if used[i]:
                    continue
                p1 = current[i]
                merged = False
                for j in range(len(current)):
                    if i == j or used[j]:
                        continue
                    p2 = current[j]
                    # Try p1 + p2[1:]
                    if p1[-1] == p2[0]:
                        merged_path = p1 + p2[1:]
                        if len(merged_path) == len(set(merged_path)) and all(
                            tuple(sorted((merged_path[k], merged_path[k+1]))) in edge_set
                            for k in range(len(merged_path) - 1)
                        ):
                            new_current.append(merged_path)
                            used[j] = True
                            merged = True
                            improved = True
                            break
                    # Try p2 + p1[1:]
                    if p2[-1] == p1[0]:
                        merged_path = p2 + p1[1:]
                        if len(merged_path) == len(set(merged_path)) and all(
                            tuple(sorted((merged_path[k], merged_path[k+1]))) in edge_set
                            for k in range(len(merged_path) - 1)
                        ):
                            new_current.append(merged_path)
                            used[j] = True
                            merged = True
                            improved = True
                            break
                if not merged:
                    new_current.append(p1)
            current = new_current
        return current

    def eulerian_decompose_with_pairing(edges, odd_pairing, deadline):
        """
        Perform Eulerian decomposition given a specific pairing of odd vertices.
        """
        if not edges:
            return []
        adj = build_adj(edges)
        
        # Add virtual edges based on the provided pairing
        virtual_edges = []
        for a, b in odd_pairing:
            # BFS for shortest path between a and b
            dist = {a: 0}
            parent = {}
            q = [a]
            for u in q:
                for nb in adj[u]:
                    if nb not in dist:
                        dist[nb] = dist[u] + 1
                        parent[nb] = u
                        q.append(nb)
            if b in dist:
                path = [b]
                cur = b
                while cur != a:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                for k in range(len(path) - 1):
                    virtual_edges.append(tuple(sorted((path[k], path[k+1]))))
            else:
                # Should not happen if graph is connected
                pass

        all_edges = list(edges) + virtual_edges
        m = len(all_edges)
        m_adj = defaultdict(list)
        for eid, (u, v) in enumerate(all_edges):
            m_adj[u].append((v, eid))
            m_adj[v].append((u, eid))

        used = [False] * m

        def hierholzer(start):
            stack = [start]
            circuit = []
            idx = defaultdict(int)
            while stack:
                v = stack[-1]
                while idx[v] < len(m_adj[v]) and used[m_adj[v][idx[v]][1]]:
                    idx[v] += 1
                if idx[v] < len(m_adj[v]):
                    nb, eid = m_adj[v][idx[v]]
                    idx[v] += 1
                    used[eid] = True
                    stack.append(nb)
                else:
                    circuit.append(stack.pop())
            circuit.reverse()
            return circuit

        circuits = []
        for eid in range(m):
            if not used[eid]:
                u, v = all_edges[eid]
                circuit = hierholzer(u)
                circuits.append(circuit)

        paths = []
        for circuit in circuits:
            current = [circuit[0]]
            seen = {circuit[0]}
            for k in range(1, len(circuit)):
                vtx = circuit[k]
                if vtx in seen:
                    if len(current) >= 2:
                        paths.append(current)
                    current = [vtx]
                    seen = {vtx}
                else:
                    current.append(vtx)
                    seen.add(vtx)
            if len(current) >= 2:
                paths.append(current)

        return paths

    def decompose_one(edges, deadline):
        if not edges:
            return []
        best = None
        best_count = len(edges)
        
        adj = build_adj(edges)
        degree = {v: len(adj[v]) for v in adj}
        odd = [v for v in degree if degree[v] % 2 == 1]
        
        # Strategy 1: Eulerian with multiple random pairings
        if time.time() < deadline - 5:
            # Try a few random pairings
            # If len(odd) is small, we can even try all permutations?
            # len(odd) is usually even. For n=20, max odd is 20. 20! is too big.
            # Random sampling is good.
            num_attempts = 50
            for _ in range(num_attempts):
                if time.time() > deadline - 2:
                    break
                if len(odd) < 2:
                    pairing = []
                else:
                    # Random shuffle of odd vertices and pair consecutive
                    shuffled_odd = odd[:]
                    random.shuffle(shuffled_odd)
                    pairing = [(shuffled_odd[i], shuffled_odd[i+1]) for i in range(0, len(shuffled_odd), 2)]
                
                paths = eulerian_decompose_with_pairing(edges, pairing, deadline)
                paths = verify_and_fix(paths, edges)
                paths = merge_paths(paths, edges)
                
                if len(paths) < best_count:
                    best_count = len(paths)
                    best = paths
        
        # Strategy 2: Greedy (kept as fallback or competitor)
        # ... (existing greedy code would go here, but we can simplify if Eulerian is strong)
        # Let's add a simple greedy if Eulerian didn't find a good solution or if time permits
        
        if best is None or time.time() < deadline - 10:
             # Simple Greedy
             def greedy_extend(start, adj, remaining, rng, warnsdorff=True, max_steps=500):
                 path = [start]
                 visited = {start}
                 for _ in range(max_steps):
                     cur = path[-1]
                     nbs = [nb for nb in adj[cur] if nb not in visited and (tuple(sorted((cur, nb)))) in remaining]
                     if not nbs:
                         break
                     if warnsdorff and len(nbs) > 1:
                         def score(nb):
                             return sum(1 for x in adj[nb] if x not in visited and (tuple(sorted((nb, x)))) in remaining)
                         nbs.sort(key=score)
                         top = min(3, len(nbs))
                         choice = rng.choice(nbs[:top])
                     else:
                         choice = rng.choice(nbs)
                     path.append(choice)
                     visited.add(choice)
                 return path

             for seed in range(10):
                 if time.time() > deadline - 2:
                     break
                 rng = random.Random(seed * 7919 + 13)
                 remaining = set(tuple(sorted(e)) for e in edges)
                 g_adj = build_adj(list(remaining))
                 g_degree = {v: len(g_adj[v]) for v in g_adj}
                 paths = []

                 while remaining:
                     if time.time() > deadline - 1:
                         break
                     odd_verts = [v for v in g_adj if g_degree.get(v, 0) % 2 == 1 and g_adj[v]]
                     if odd_verts:
                         start = rng.choice(odd_verts)
                     else:
                         cands = [v for v in g_adj if g_adj[v]]
                         if not cands:
                             break
                         cands.sort(key=lambda v: g_degree.get(v, 0), reverse=True)
                         start = rng.choice(cands[:min(5, len(cands))])

                     path = greedy_extend(start, g_adj, remaining, rng, warnsdorff=True)

                     if len(path) < 2:
                         nbs = [nb for nb in g_adj[start] if (tuple(sorted((start, nb)))) in remaining]
                         if nbs:
                             path = [start, rng.choice(nbs)]
                         else:
                             e = next(iter(remaining))
                             path = [e[0], e[1]]

                     path_edges = set()
                     for i in range(len(path) - 1):
                         e = tuple(sorted((path[i], path[i+1])))
                         if e in remaining:
                             path_edges.add(e)

                     if not path_edges:
                         e = next(iter(remaining))
                         path = [e[0], e[1]]
                         path_edges = {e}

                     for e in path_edges:
                         remaining.discard(e)
                         u, v = e
                         g_adj[u].discard(v)
                         g_adj[v].discard(u)
                         g_degree[u] = max(0, g_degree.get(u, 0) - 1)
                         g_degree[v] = max(0, g_degree.get(v, 0) - 1)
                     for v in list(g_adj.keys()):
                         if not g_adj[v]:
                             del g_adj[v]

                     paths.append(path)

                 paths = verify_and_fix(paths, edges)
                 paths = merge_paths(paths, edges)
                 if len(paths) < best_count:
                     best_count = len(paths)
                     best = paths

        if best is None:
            return [[u, v] for u, v in edges]
        return best

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    for i, spec in enumerate(specs):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        remaining_time = deadline - time.time()
        per = remaining_time / (3 - len(results))
        paths = decompose_one(edges, time.time() + per)
        results.append(paths)

    while len(results) < 3:
        spec = specs[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])

    return results
