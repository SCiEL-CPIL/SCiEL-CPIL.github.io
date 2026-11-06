import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    def has_c5(adj, n):
        # Fast characterization: C5-free iff every edge's common-neighborhood is a clique.
        for u in range(n):
            nu = adj[u]
            for v in nu:
                if v <= u:
                    continue
                common = nu & adj[v]
                # common must be a clique
                cl = list(common)
                for i in range(len(cl)):
                    a = cl[i]
                    for j in range(i + 1, len(cl)):
                        if cl[j] not in adj[a]:
                            return True
        return False

    def _has_c5_brute(adj, n):
        # Fallback: direct 5-cycle search (used only to validate the fast check).
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v1:
                        continue
                    for v3 in adj[v2]:
                        if v3 == v0 or v3 == v1 or v3 == v2:
                            continue
                        for v4 in adj[v3]:
                            if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                                continue
                            if v0 in adj[v4]:
                                return True
        return False

    def _clique_size(adj, cand, cur_size):
        # Maximum clique size in adj restricted to cand (greedy coloring bound).
        best = cur_size
        while cand:
            if not cand:
                break
            if len(cand) + cur_size <= best:
                break
            # Order by degree within cand
            verts = sorted(cand, key=lambda x: len(adj[x] & cand))
            # Greedy coloring
            color = {}
            color_class = []
            for v in verts:
                placed = False
                for ci, cc in enumerate(color_class):
                    if not (adj[v] & cc):
                        color[v] = ci
                        cc.add(v)
                        placed = True
                        break
                if not placed:
                    color[v] = len(color_class)
                    color_class.append({v})
            # Process in reverse color order
            for ci in range(len(color_class) - 1, -1, -1):
                cc = color_class[ci]
                if cur_size + len(cc) <= best:
                    break
                for v in cc:
                    if cur_size + 1 > best:
                        best = cur_size + 1
                    new_cand = cand & adj[v]
                    s = _clique_size(adj, new_cand, cur_size + 1)
                    if s > best:
                        best = s
                    cand = cand - {v}
                    if len(cand) + cur_size <= best:
                        break
        return best

    def has_independent_set_9(adj, n):
        # alpha(G) = max clique of complement. Build complement adjacency.
        comp = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    comp[i].add(j)
                    comp[j].add(i)
        return _clique_size(comp, set(range(n)), 0) >= 9

    def is_valid(n, adj):
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges

    def make_8k4():
        n = 32
        adj = [set() for _ in range(n)]
        for b in range(8):
            for i in range(4):
                for j in range(i + 1, 4):
                    u = 4 * b + i
                    v = 4 * b + j
                    adj[u].add(v)
                    adj[v].add(u)
        return n, adj

    # Validate the fast C5 check against brute force on the known solution.
    n_base, adj_base = make_8k4()
    assert has_c5(adj_base, n_base) == _has_c5_brute(adj_base, n_base)
    assert is_valid(n_base, adj_base)

    n_best, adj_best = n_base, [set(s) for s in adj_base]

    random.seed(12345)

    # ---- Try to reach n = 33 ----
    # Structured: start from 8xK4, add a vertex with a random connection pattern,
    # then do local flips to reduce alpha and eliminate C5.
    if time_left() > 20:
        base_n, base_adj = make_8k4()
        for attempt in range(60):
            if time_left() < 15:
                break
            adj = [set(s) for s in base_adj]
            adj.append(set())
            n = 33
            # Connect vertex 32 to a random subset
            for i in range(32):
                if random.random() < 0.55:
                    adj[32].add(i)
                    adj[i].add(32)
            # Local search: flip edges to remove C5 and reduce alpha
            for it in range(300):
                if time_left() < 15:
                    break
                if not has_c5(adj, n) and not has_independent_set_9(adj, n):
                    n_best, adj_best = n, adj
                    break
                # Pick a random edge to flip
                u = random.randrange(n)
                v = random.randrange(n)
                if u == v:
                    continue
                if v in adj[u]:
                    adj[u].discard(v)
                    adj[v].discard(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)

    # ---- Random search for n = 33, 34, 35 ----
    for target_n in [33, 34, 35]:
        if n_best >= target_n:
            continue
        if time_left() < 15:
            break
        for attempt in range(400):
            if time_left() < 10:
                break
            p = random.uniform(0.65, 0.9)
            adj = [set() for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
            if has_c5(adj, target_n):
                continue
            if has_independent_set_9(adj, target_n):
                continue
            n_best, adj_best = target_n, adj
            break
        if n_best == target_n:
            break

    return {'n': n_best, 'edges': adj_to_edges(adj_best, n_best)}
