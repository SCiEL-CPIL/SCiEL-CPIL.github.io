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

    def get_deg(adj):
        return [bin(adj[i]).count('1') for i in range(n)]

    def check_fast(adj, deg, edges):
        """Check validity: for each edge (u,v), no s in N(u)\\{v}, t in N(v)\\{u} with deg[s]==deg[t]."""
        for u, v in edges:
            # Collect degrees of neighbors of u (excluding v)
            du_deg_set = set()
            au = adj[u]
            # Mask out v
            au_masked = au & ~(1 << v)
            s = au_masked
            while s:
                low = s & (-s)
                si = low.bit_length() - 1
                du_deg_set.add(deg[si])
                s ^= low
            if not du_deg_set:
                continue
            # Check neighbors of v (excluding u)
            av = adj[v] & ~(1 << u)
            t = av
            while t:
                low = t & (-t)
                ti = low.bit_length() - 1
                if deg[ti] in du_deg_set:
                    return False
                t ^= low
        return True

    def check_edges(edges):
        adj = build_adj(edges)
        deg = get_deg(adj)
        return check_fast(adj, deg, edges)

    def try_add_edges(edges):
        """Greedy: try to add missing edges one by one."""
        edges = [e[:] for e in edges]
        adj = build_adj(edges)
        deg = get_deg(adj)
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for u in range(n):
                if time.time() > deadline - 3:
                    break
                for v in range(u + 1, n):
                    if (u, v) in present:
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                    if check_fast(adj, deg, edges + [[u, v]]):
                        present.add((u, v))
                        edges.append([u, v])
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        deg[u] -= 1
                        deg[v] -= 1
        return edges

    def fix_invalid(edges):
        """Remove edges to make graph valid."""
        if check_edges(edges):
            return [e[:] for e in edges]
        edges = [e[:] for e in edges]
        while time.time() < deadline - 5:
            if check_edges(edges):
                break
            adj = build_adj(edges)
            deg = get_deg(adj)
            # Find most violating edge
            edge_viol = {}
            for u, v in edges:
                count = 0
                au = adj[u] & ~(1 << v)
                av = adj[v] & ~(1 << u)
                s = au
                while s:
                    low = s & (-s)
                    si = low.bit_length() - 1
                    t = av
                    while t:
                        low2 = t & (-t)
                        ti = low2.bit_length() - 1
                        if deg[si] == deg[ti]:
                            count += 1
                        t ^= low2
                    s ^= low
                edge_viol[(u, v)] = count
            if not edge_viol or max(edge_viol.values()) == 0:
                break
            worst = max(edge_viol, key=lambda e: edge_viol[e])
            edges = [e for e in edges if not ((e[0] == worst[0] and e[1] == worst[1]) or (e[0] == worst[1] and e[1] == worst[0]))]
        return edges

    # Generate diverse candidates
    candidates = []

    # K_{a,b} for various splits
    for a in range(2, 14):
        b = 15 - a
        if a == b:
            continue
        edges = []
        for x in range(a):
            for y in range(a, 15):
                edges.append([x, y])
        candidates.append(edges)

    # K_{7,8} plus internal edges in partition A
    A = list(range(7))
    B = list(range(7, 15))
    base = []
    for x in A:
        for y in B:
            base.append([x, y])
    candidates.append(base)

    # K_{7,8} plus some internal edges in A
    for seed in range(10):
        random.seed(seed)
        edges = [e[:] for e in base]
        # Add random internal edges in A
        for i in range(7):
            for j in range(i+1, 7):
                if random.random() < 0.5:
                    edges.append([i, j])
        candidates.append(edges)

    # K_{7,8} plus some internal edges in B
    for seed in range(10):
        random.seed(seed + 50)
        edges = [e[:] for e in base]
        for i in range(7):
            for j in range(i+1, 8):
                if random.random() < 0.5:
                    edges.append([7+i, 7+j])
        candidates.append(edges)

    # Random graphs with degree-diversity bias
    for seed in range(15):
        random.seed(seed + 200)
        edges = []
        present_set = set()
        for u in range(n):
            target = random.randint(2, 12)
            cands = [v for v in range(n) if v != u]
            random.shuffle(cands)
            for v in cands[:target]:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in present_set:
                    edges.append([u, v])
                    present_set.add((a, b))
        candidates.append(edges)

    # Try to build graph with as many distinct degrees as possible
    for seed in range(10):
        random.seed(seed + 300)
        edges = []
        present_set = set()
        # Start with a path to give varied degrees
        for i in range(n - 1):
            edges.append([i, i + 1])
            present_set.add((i, i + 1))
        # Add random edges
        for _ in range(30):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in present_set:
                edges.append([u, v])
                present_set.add((a, b))
        candidates.append(edges)

    # Complete bipartite with added cross-edges variations
    for seed in range(10):
        random.seed(seed + 400)
        a = random.randint(5, 10)
        b = 15 - a
        edges = []
        present_set = set()
        for x in range(a):
            for y in range(a, 15):
                edges.append([x, y])
                present_set.add((x, y))
        # Add some internal edges
        for i in range(a):
            for j in range(i + 1, a):
                if random.random() < 0.3:
                    edges.append([i, j])
                    present_set.add((i, j))
        for i in range(a, 15):
            for j in range(i + 1, 15):
                if random.random() < 0.3:
                    edges.append([i, j])
                    present_set.add((i, j))
        candidates.append(edges)

    # Process candidates
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        fixed = fix_invalid(cand)
        improved = try_add_edges(fixed)
        if check_edges(improved):
            cnt = len(improved)
            if cnt > best_count:
                best_count = cnt
                best_edges = [e[:] for e in improved]

        # If we have a good result, try more improvements
        if best_edges is not None and best_count >= 55 and time.time() < deadline - 10:
            # Try multiple greedy passes with different orderings
            for trial in range(5):
                if time.time() > deadline - 5:
                    break
                random.seed(trial + 500)
                edges = [e[:] for e in best_edges]
                adj = build_adj(edges)
                deg = get_deg(adj)
                present = set()
                for u, v in edges:
                    a, b = (u, v) if u < v else (v, u)
                    present.add((a, b))
                # Try adding in random order
                missing = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in present]
                random.shuffle(missing)
                for u, v in missing:
                    if time.time() > deadline - 5:
                        break
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                    if check_fast(adj, deg, edges + [[u, v]]):
                        present.add((u, v))
                        edges.append([u, v])
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        deg[u] -= 1
                        deg[v] -= 1
                if check_edges(edges) and len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = [e[:] for e in edges]

    # Final polish
    if best_edges is not None and time.time() < deadline - 3:
        final = try_add_edges(best_edges)
        if check_edges(final) and len(final) > best_count:
            best_count = len(final)
            best_edges = final

    if best_edges is None:
        # Fallback: K_{7,8}
        best_edges = []
        for x in range(7):
            for y in range(7, 15):
                best_edges.append([x, y])

    out = [[int(u), int(v)] for u, v in best_edges]
    return out
