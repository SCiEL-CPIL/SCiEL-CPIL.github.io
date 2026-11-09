import time
import random

def construct():
    n = 15
    deadline = time.time() + 85.0
    best_edges = None
    best_count = -1

    def build_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def check(edges):
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        # For each edge (u,v): forbidden if N(u)\{v} and N(v)\{u} share a degree value
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                # degrees of neighbors of u excluding v
                du_set = set()
                au = adj[u]
                for s in range(n):
                    if s == v:
                        continue
                    if (au >> s) & 1:
                        du_set.add(deg[s])
                if not du_set:
                    continue
                av = adj[v]
                for t in range(n):
                    if t == u:
                        continue
                    if (av >> t) & 1:
                        if deg[t] in du_set:
                            return False
        return True

    def try_improve(start_edges):
        edges = list(start_edges)
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u in range(n):
                if time.time() > deadline - 2:
                    break
                for v in range(u + 1, n):
                    if (u, v) in present:
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                    if check(edges + [[u, v]]):
                        present.add((u, v))
                        edges.append([u, v])
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        deg[u] -= 1
                        deg[v] -= 1
        return edges

    def local_search(start_edges):
        import math
        edges = list(start_edges)
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        best = len(edges)
        best_copy = [e[:] for e in edges]

        # Greedy add
        for u in range(n):
            if time.time() > deadline - 3:
                return best_copy
            for v in range(u + 1, n):
                if (u, v) in present:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                if check(edges + [[u, v]]):
                    present.add((u, v))
                    edges.append([u, v])
                    if len(edges) > best:
                        best = len(edges)
                        best_copy = [e[:] for e in edges]
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    deg[u] -= 1
                    deg[v] -= 1

        # Simulated annealing with remove+add moves
        if time.time() < deadline - 8:
            sa_edges = [e[:] for e in best_copy]
            sa_adj = build_adj(sa_edges)
            sa_deg = [bin(sa_adj[i]).count('1') for i in range(n)]
            sa_present = set()
            for u, v in sa_edges:
                a, b = (u, v) if u < v else (v, u)
                sa_present.add((a, b))

            # Count violations for current state
            def count_violations(adj, deg):
                cnt = 0
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (adj[u] >> v) & 1:
                            continue
                        du_set = set()
                        au = adj[u]
                        for s in range(n):
                            if s == v:
                                continue
                            if (au >> s) & 1:
                                du_set.add(deg[s])
                        if not du_set:
                            continue
                        av = adj[v]
                        for t in range(n):
                            if t == u:
                                continue
                            if (av >> t) & 1:
                                if deg[t] in du_set:
                                    cnt += 1
                return cnt

            sa_viol = count_violations(sa_adj, sa_deg)
            sa_best = len(sa_edges)
            sa_best_copy = [e[:] for e in sa_edges]
            T = 2.0
            T_min = 0.01
            cooling = 0.999

            while time.time() < deadline - 4 and T > T_min:
                if random.random() < 0.6:
                    # Try to add a random missing edge
                    missing = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in sa_present]
                    if not missing:
                        break
                    a, b = random.choice(missing)
                    sa_adj[a] |= (1 << b)
                    sa_adj[b] |= (1 << a)
                    sa_deg[a] += 1
                    sa_deg[b] += 1
                    sa_viol_new = count_violations(sa_adj, sa_deg)
                    delta = sa_viol_new - sa_viol
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                        sa_edges.append([a, b])
                        sa_present.add((a, b))
                        sa_viol = sa_viol_new
                        if sa_viol == 0 and len(sa_edges) > sa_best:
                            sa_best = len(sa_edges)
                            sa_best_copy = [e[:] for e in sa_edges]
                    else:
                        sa_adj[a] &= ~(1 << b)
                        sa_adj[b] &= ~(1 << a)
                        sa_deg[a] -= 1
                        sa_deg[b] -= 1
                else:
                    # Try to remove a random edge
                    if not sa_edges:
                        break
                    idx = random.randint(0, len(sa_edges) - 1)
                    ru, rv = sa_edges[idx]
                    sa_edges.pop(idx)
                    sa_adj[ru] &= ~(1 << rv)
                    sa_adj[rv] &= ~(1 << ru)
                    sa_deg[ru] -= 1
                    sa_deg[rv] -= 1
                    sa_present.discard((ru, rv) if ru < rv else (rv, ru))
                    sa_viol_new = count_violations(sa_adj, sa_deg)
                    delta = sa_viol_new - sa_viol
                    # Removing always accepted if it reduces violations, else with prob
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                        sa_viol = sa_viol_new
                    else:
                        # Revert
                        sa_edges.append([ru, rv])
                        sa_adj[ru] |= (1 << rv)
                        sa_adj[rv] |= (1 << ru)
                        sa_deg[ru] += 1
                        sa_deg[rv] += 1
                        sa_present.add((ru, rv) if ru < rv else (rv, ru))
                        sa_viol = sa_viol_new

                T *= cooling

            if sa_best > best:
                best = sa_best
                best_copy = sa_best_copy

        # Final greedy polish on best_copy
        if time.time() < deadline - 2:
            pol_edges = [e[:] for e in best_copy]
            pol_adj = build_adj(pol_edges)
            pol_deg = [bin(pol_adj[i]).count('1') for i in range(n)]
            pol_present = set()
            for u, v in pol_edges:
                a, b = (u, v) if u < v else (v, u)
                pol_present.add((a, b))
            improved_pol = True
            while improved_pol and time.time() < deadline - 1:
                improved_pol = False
                for u in range(n):
                    if time.time() > deadline - 1:
                        break
                    for v in range(u + 1, n):
                        if (u, v) in pol_present:
                            continue
                        pol_adj[u] |= (1 << v)
                        pol_adj[v] |= (1 << u)
                        pol_deg[u] += 1
                        pol_deg[v] += 1
                        if check(pol_edges + [[u, v]]):
                            pol_present.add((u, v))
                            pol_edges.append([u, v])
                            improved_pol = True
                            if len(pol_edges) > best:
                                best = len(pol_edges)
                                best_copy = [e[:] for e in pol_edges]
                        else:
                            pol_adj[u] &= ~(1 << v)
                            pol_adj[v] &= ~(1 << u)
                            pol_deg[u] -= 1
                            pol_deg[v] -= 1

        return best_copy

    # Generate candidates
    candidates = []

    # Bipartite graphs
    for a in range(1, 15):
        edges = []
        A = list(range(a))
        B = list(range(a, 15))
        for x in A:
            for y in B:
                edges.append([x, y])
        candidates.append(edges)

    # Complete graph
    complete = [[u, v] for u in range(15) for v in range(u + 1, 15)]
    candidates.append(complete)

    # Empty
    candidates.append([])

    # Random graphs with various densities
    for seed in range(20):
        random.seed(seed)
        density = 0.15 + 0.02 * seed
        edges = []
        for u in range(15):
            for v in range(u + 1, 15):
                if random.random() < density:
                    edges.append([u, v])
        candidates.append(edges)

    # Degree-diversity biased random: try to create varied degrees
    for seed in range(5):
        random.seed(seed + 100)
        edges = []
        for u in range(15):
            target_deg = random.randint(1, 10)
            candidates_v = [v for v in range(15) if v != u]
            random.shuffle(candidates_v)
            for v in candidates_v[:target_deg]:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in set((min(e[0],e[1]), max(e[0],e[1])) for e in edges):
                    edges.append([u, v])
        candidates.append(edges)

    def fix_invalid(edges):
        if check(edges):
            return edges
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            if check(edges):
                break
            # Find all violations and count which edge appears most
            adj = build_adj(edges)
            deg = [bin(adj[i]).count('1') for i in range(n)]
            edge_violation_count = {}
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        continue
                    for s in range(n):
                        if s == u or s == v:
                            continue
                        if not (adj[u] >> s) & 1:
                            continue
                        for t in range(n):
                            if t == u or t == v or t == s:
                                continue
                            if not (adj[v] >> t) & 1:
                                continue
                            if deg[s] == deg[t]:
                                edge_violation_count[(u, v)] = edge_violation_count.get((u, v), 0) + 1
            if not edge_violation_count:
                break
            # Remove the edge with most violations
            worst_edge = max(edge_violation_count, key=lambda e: edge_violation_count[e])
            edges = [e for e in edges if not ((e[0] == worst_edge[0] and e[1] == worst_edge[1]) or
                                              (e[0] == worst_edge[1] and e[1] == worst_edge[0]))]
            changed = True
        return edges

    for cand in candidates:
        if time.time() > deadline - 3:
            break
        cand = fix_invalid(cand)
        improved = try_improve(cand)
        if check(improved):
            cnt = len(improved)
            if cnt > best_count:
                best_count = cnt
                best_edges = improved

        if best_edges is not None and time.time() < deadline - 15:
            ls_result = local_search(best_edges)
            if check(ls_result):
                cnt = len(ls_result)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = ls_result

    if best_edges is None:
        return [[0, v] for v in range(1, 15)]

    out = [[int(u), int(v)] for u, v in best_edges]
    return out
