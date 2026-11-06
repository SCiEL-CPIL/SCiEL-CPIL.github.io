import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin below the 150s kill

    def remaining():
        return TIME_LIMIT - (time.time() - start_time)

    def solve_instance(n, p, seed):
        # --- generate graph ---
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

        def is_clique(vertices):
            # vertices: list of ints
            if len(vertices) < 2:
                return True
            m = None
            for v in vertices:
                m = adj[v] if m is None else (m & adj[v])
            for v in vertices:
                if not (m >> v) & 1:
                    return False
            return True

        def grow_clique(start_edge, forbidden, forbidden_mask):
            a, b = start_edge
            if a in forbidden or b in forbidden:
                return None
            verts = [a, b]
            while True:
                # find a candidate vertex adjacent to all current verts, not forbidden, not in verts
                best = -1
                best_deg = -1
                for v in range(n):
                    if v in verts or v in forbidden:
                        continue
                    ok = True
                    for v2 in verts:
                        if not (adj[v] >> v2) & 1:
                            ok = False
                            break
                    if ok:
                        deg = bin(adj[v]).count('1')
                        if deg > best_deg:
                            best_deg = deg
                            best = v
                if best < 0:
                    break
                verts.append(best)
            return verts

        def build_partition():
            forbidden = set()
            forbidden_mask = 0
            cliques = []
            edge_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        cn = bin(adj[u] & adj[v]).count('1')
                        edge_list.append((cn, u, v))
            edge_list.sort()
            for cn, u, v in edge_list:
                if (u in forbidden) or (v in forbidden):
                    continue
                verts = grow_clique((u, v), forbidden, forbidden_mask)
                if verts is None:
                    continue
                cliques.append(verts)
                for x in verts:
                    forbidden.add(x)
                    forbidden_mask |= (1 << x)
            return cliques

        def refine(cliques, budget):
            cliques = [list(c) for c in cliques]
            improved = True
            while improved and (time.time() - start_time) < budget:
                improved = False
                for i in range(len(cliques)):
                    if (time.time() - start_time) > budget:
                        break
                    for j in range(i + 1, len(cliques)):
                        union = cliques[i] + cliques[j]
                        if is_clique(union):
                            cliques[i] = union
                            del cliques[j]
                            improved = True
                            break
                    if improved:
                        break
            return cliques

        cliques = build_partition()
        budget = min(remaining() - 3.0, 20.0)
        if budget > 1.0:
            cliques = refine(cliques, budget)
        return [list(c) for c in cliques]

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if remaining() < 10.0:
            rng = random.Random(seed)
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            trivial = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        trivial.append([u, v])
            results.append(trivial)
        else:
            res = solve_instance(n, p, seed)
            results.append(res)

    return results
