import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # safety margin before 150s kill

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    bit = [1 << i for i in range(n)]

    # ---- Incremental common-neighbor tracking state ----
    # adj[i] is a bitmask of neighbors of i
    # cn[u][v] = number of common neighbors of u and v (for u != v)
    adj = [0] * n
    cn = [[0] * n for _ in range(n)]
    edge_count = 0

    def popcount(x):
        return x.bit_count()

    def is_valid_add(u, v):
        """O(1)-ish validity: adding (u,v) keeps all cn <= 2."""
        if (adj[u] >> v) & 1:
            return False
        # u and v must end with <= 2 common neighbors
        if cn[u][v] > 2:
            return False
        # For each w in N(u), w != v: u becomes a common neighbor of v,w
        # so cn[v][w] must be <= 1 currently
        nu = adj[u] & ~bit[v]
        while nu:
            low = nu & (-nu)
            w = low.bit_length() - 1
            if cn[v][w] > 1:
                return False
            nu ^= low
        # For each w in N(v), w != u: v becomes a common neighbor of u,w
        nv = adj[v] & ~bit[u]
        while nv:
            low = nv & (-nv)
            w = low.bit_length() - 1
            if cn[u][w] > 1:
                return False
            nv ^= low
        return True

    def add_edge(u, v):
        """Add edge, update cn incrementally. Assumes validity already checked."""
        nonlocal edge_count
        adj[u] |= bit[v]
        adj[v] |= bit[u]
        # common neighbors of u and v all gain u and v as common-neighbor partners
        common = adj[u] & adj[v]
        while common:
            low = common & (-common)
            w = low.bit_length() - 1
            # u is now a common neighbor of v and w
            cn[v][w] += 1
            cn[w][v] += 1
            # v is now a common neighbor of u and w
            cn[u][w] += 1
            cn[w][u] += 1
            # u and v now share w as a common neighbor
            cn[u][v] += 1
            cn[v][u] += 1
            common ^= low
        edge_count += 1

    def remove_edge(u, v):
        """Remove edge, update cn incrementally."""
        nonlocal edge_count
        if not ((adj[u] >> v) & 1):
            return
        # common neighbors BEFORE removal
        common = adj[u] & adj[v]
        adj[u] &= ~bit[v]
        adj[v] &= ~bit[u]
        while common:
            low = common & (-common)
            w = low.bit_length() - 1
            cn[v][w] -= 1
            cn[w][v] -= 1
            cn[u][w] -= 1
            cn[w][u] -= 1
            cn[u][v] -= 1
            cn[v][u] -= 1
            common ^= low
        edge_count -= 1

    def get_present_edges():
        res = []
        for i in range(n):
            mask = adj[i]
            while mask:
                low = mask & (-mask)
                j = low.bit_length() - 1
                if i < j:
                    res.append((i, j))
                mask ^= low
        return res

    def get_missing_edges():
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj[i] >> j) & 1):
                    res.append((i, j))
        return res

    def snapshot():
        return (adj[:], [row[:] for row in cn], edge_count)

    def restore(snap):
        nonlocal edge_count
        adj[:] = snap[0][:]
        for i in range(n):
            cn[i][:] = snap[1][i][:]
        edge_count = snap[2]

    def greedy_fill(order=None):
        """Add all valid edges from order (default: all missing, shuffled)."""
        if order is None:
            order = get_missing_edges()
            random.shuffle(order)
        for u, v in order:
            if is_valid_add(u, v):
                add_edge(u, v)

    def spectral_init():
        """
        Spectral heuristic seed: build a random symmetric matrix, take the
        top eigenvector directions as a low-dim vertex embedding, and connect
        vertices whose embedding dot product exceeds a threshold. This tends
        to produce dense, correlated graphs that give the local search a
        strong starting point. We then greedily repair to K2,3-freeness.
        """
        adj[:] = [0] * n
        for i in range(n):
            cn[i][:] = [0] * n
        edge_count = 0

        # Build a random symmetric matrix
        M = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                val = random.gauss(0.0, 1.0)
                M[i][j] = val
                M[j][i] = val

        # Power iteration to get top eigenvector (and a couple more via
        # deflation) to form a 2D embedding.
        def power_iter(vec, iters=200):
            for _ in range(iters):
                new = [sum(vec[k] * M[k][j] for k in range(n)) for j in range(n)]
                norm = math.sqrt(sum(x * x for x in new))
                if norm < 1e-12:
                    return vec
                new = [x / norm for x in new]
                vec = new
            return vec

        # First eigenvector
        v1 = [random.gauss(0, 1) for _ in range(n)]
        v1 = power_iter(v1)
        # Deflate
        M2 = [row[:] for row in M]
        for i in range(n):
            for j in range(n):
                M2[i][j] -= v1[i] * v1[j]
        v2 = [random.gauss(0, 1) for _ in range(n)]
        v2 = power_iter(v2, 150)

        # Embedding: vertex i -> (v1[i], v2[i])
        # Connect if dot product > threshold; choose threshold to target density
        # Try a few thresholds, keep the one that gives the most valid edges.
        best_seed = None
        best_seed_count = -1
        for thr in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
            cand_adj = [0] * n
            cand_cn = [[0] * n for _ in range(n)]
            cand_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dot = v1[i] * v1[j] + v2[i] * v2[j]
                    if dot > thr:
                        # quick validity check in candidate
                        ok = True
                        if (cand_adj[i] & cand_adj[j]).bit_count() > 2:
                            ok = False
                        if ok:
                            for w in range(n):
                                if w == i or w == j:
                                    if (cand_adj[i] >> w) & 1:
                                        if (cand_adj[j] & cand_adj[w]).bit_count() > 1:
                                            ok = False
                                            break
                                    if (cand_adj[j] >> w) & 1:
                                        if (cand_adj[i] & cand_adj[w]).bit_count() > 1:
                                            ok = False
                                            break
                        if ok:
                            cand_adj[i] |= bit[j]
                            cand_adj[j] |= bit[i]
                            cand_count += 1
            if cand_count > best_seed_count:
                best_seed_count = cand_count
                best_seed = (cand_adj, cand_count)

        if best_seed is not None:
            adj[:] = best_seed[0][:]
            edge_count = best_seed[1]
            # rebuild cn
            for i in range(n):
                cn[i][:] = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    c = (adj[i] & adj[j]).bit_count()
                    cn[i][j] = c
                    cn[j][i] = c
            greedy_fill()
            return edge_count
        return edge_count

    # ---- Local search with incremental tracking ----
    def local_search(max_steps=60000):
        """Simulated-annealing-style local search using fast incremental moves."""
        best_adj, best_cn, best_count = snapshot()
        T = 4.0
        T_end = 0.02
        steps = max_steps
        for step in range(steps):
            if time.time() > deadline - 4:
                break
            progress = step / steps
            T = T_end + (4.0 - T_end) * ((T_end / 4.0) ** progress)

            present = get_present_edges()
            missing = get_missing_edges()
            r = random.random()

            if r < 0.45:
                # Try to add a random missing edge
                if not missing:
                    continue
                u, v = random.choice(missing)
                if is_valid_add(u, v):
                    add_edge(u, v)
                    if edge_count > best_count:
                        best_adj, best_cn, best_count = snapshot()
            elif r < 0.75:
                # Remove a random edge then greedily refill (kick)
                if not present:
                    continue
                u, v = random.choice(present)
                backup = snapshot()
                remove_edge(u, v)
                greedy_fill()
                delta = edge_count - backup[2]
                if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                    if edge_count > best_count:
                        best_adj, best_cn, best_count = snapshot()
                else:
                    restore(backup)
            else:
                # Swap: remove one, add another, then refill
                if not present or not missing:
                    continue
                u1, v1 = random.choice(present)
                u2, v2 = random.choice(missing)
                backup = snapshot()
                remove_edge(u1, v1)
                if is_valid_add(u2, v2):
                    add_edge(u2, v2)
                greedy_fill()
                delta = edge_count - backup[2]
                if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                    if edge_count > best_count:
                        best_adj, best_cn, best_count = snapshot()
                else:
                    restore(backup)

        restore((best_adj, best_cn, best_count))
        return edge_count

    # ---- Crossover: union + repair using fast tracking ----
    def crossover(a_snap, b_snap):
        """Union of two graphs, then repair violations by removing edges."""
        adj[:] = [0] * n
        for i in range(n):
            adj[i] = a_snap[0][i] | b_snap[0][i]
        edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
        for i in range(n):
            cn[i][:] = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                c = (adj[i] & adj[j]).bit_count()
                cn[i][j] = c
                cn[j][i] = c

        # Repair: while any pair has > 2 common neighbors, remove an edge
        improved = True
        guard = 0
        while improved and time.time() < deadline - 6 and guard < 5000:
            improved = False
            guard += 1
            # find a violating pair
            viol = None
            for u in range(n):
                for v in range(u + 1, n):
                    if cn[u][v] > 2:
                        viol = (u, v)
                        break
                if viol:
                    break
            if not viol:
                break
            u, v = viol
            common = adj[u] & adj[v]
            # pick a common neighbor w and remove one of (u,w) or (v,w)
            w_list = []
            m = common
            while m:
                low = m & (-m)
                w_list.append(low.bit_length() - 1)
                m ^= low
            w = random.choice(w_list)
            if random.random() < 0.5:
                remove_edge(u, w)
            else:
                remove_edge(v, w)
            improved = True

        greedy_fill()
        return snapshot()

    # ---- Main: build diverse population, crossover, local search ----
    best_adj, best_cn, best_count = [0] * n, [[0] * n for _ in range(n)], 0

    # Phase 0: spectral seeds
    if time.time() < deadline - 60:
        for _ in range(4):
            spectral_init()
            if edge_count > best_count:
                best_adj, best_cn, best_count = snapshot()

    # Phase 1: random greedy seeds
    population = []
    for _ in range(12):
        if time.time() > deadline - 50:
            break
        adj[:] = [0] * n
        for i in range(n):
            cn[i][:] = [0] * n
        edge_count = 0
        order = all_pairs[:]
        random.shuffle(order)
        for u, v in order:
            if is_valid_add(u, v):
                add_edge(u, v)
        greedy_fill()
        population.append(snapshot())
        if edge_count > best_count:
            best_adj, best_cn, best_count = snapshot()

    population.sort(key=lambda s: -s[2])
    if population and population[0][2] > best_count:
        best_adj, best_cn, best_count = population[0]

    # Phase 2: evolutionary loop with crossover + local search
    pop_size = 10
    while time.time() < deadline - 25 and len(population) >= 2:
        # elitism
        new_pop = [population[0], population[1]]
        while len(new_pop) < pop_size and time.time() < deadline - 25:
            # tournament select two parents
            def tselect(k=3):
                cands = random.sample(population, min(k, len(population)))
                return max(cands, key=lambda s: s[2])
            p1 = tselect()
            p2 = tselect()
            child = crossover(p1, p2)
            # local search on child
            restore(child)
            local_search(max_steps=15000)
            child = snapshot()
            new_pop.append(child)
            if edge_count > best_count:
                best_adj, best_cn, best_count = snapshot()
        population = sorted(new_pop, key=lambda s: -s[2])[:pop_size]

    # Phase 3: final polish on the best solution with more steps
    if time.time() < deadline - 8:
        restore((best_adj, best_cn, best_count))
        local_search(max_steps=120000)
        best_adj, best_cn, best_count = snapshot()

    # Build output edges
    edges = []
    for i in range(n):
        mask = adj[i]
        while mask:
            low = mask & (-mask)
            j = low.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= low

    # Final verification + fallback
    if edges:
        vadj = [0] * n
        for u, v in edges:
            vadj[u] |= bit[v]
            vadj[v] |= bit[u]
        valid = True
        for u in range(n):
            for v in range(u + 1, n):
                if (vadj[u] & vadj[v]).bit_count() > 2:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback: simple greedy
            adj[:] = [0] * n
            for i in range(n):
                cn[i][:] = [0] * n
            edge_count = 0
            order = all_pairs[:]
            random.shuffle(order)
            for u, v in order:
                if is_valid_add(u, v):
                    add_edge(u, v)
            greedy_fill()
            edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    low = mask & (-mask)
                    j = low.bit_length() - 1
                    if i < j:
                        edges.append([i, j])
                    mask ^= low

    return edges


if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Found {len(res)} edges in {time.time() - start:.2f}s")
    n = 20
    adj = [0] * n
    for u, v in res:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            c = (adj[u] & adj[v]).bit_count()
            if c > 2:
                valid = False
                print(f"Invalid: {u},{v} have {c} common neighbors")
                break
        if not valid:
            break
    print(f"Valid: {valid}")
