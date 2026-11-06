import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    def is_valid_add(adj, u, v):
        # 1. u and v must not have > 2 common neighbours
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        # 2. For all w in N(u) \ {v}, |N(v) & N(w)| <= 1
        m = adj[u] & ~(1 << v)
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if (adj[v] & adj[w]).bit_count() > 1:
                return False
            m ^= lsb
        # 3. For all x in N(v) \ {u}, |N(u) & N(x)| <= 1
        m = adj[v] & ~(1 << u)
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            if (adj[u] & adj[x]).bit_count() > 1:
                return False
            m ^= lsb
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    def greedy_build(seed, base_adj=None):
        rng = random.Random(seed)
        adj = [0] * n if base_adj is None else list(base_adj)
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def local_search(adj, rng, time_budget):
        adj = list(adj)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False

            # 1) Add any missing edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_valid_add(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue

            # Build present/missing lists for swaps
            present = []
            missing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        present.append((u, v))
                    else:
                        missing.append((u, v))
            if not present or not missing:
                break

            # 2) 1-swap
            found = False
            for _ in range(150):
                if time.time() - start > time_budget:
                    break
                u, v = rng.choice(present)
                x, y = rng.choice(missing)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                if is_valid_add(adj, x, y):
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    improved = True
                    found = True
                    break
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            if found:
                continue

            # 3) 2-swap
            found = False
            for _ in range(60):
                if time.time() - start > time_budget:
                    break
                (u1, v1), (u2, v2) = rng.sample(present, 2)
                (x1, y1), (x2, y2) = rng.sample(missing, 2)
                adj[u1] &= ~(1 << v1); adj[v1] &= ~(1 << u1)
                adj[u2] &= ~(1 << v2); adj[v2] &= ~(1 << u2)
                ok = is_valid_add(adj, x1, y1)
                if ok:
                    adj[x1] |= (1 << y1); adj[y1] |= (1 << x1)
                    ok = is_valid_add(adj, x2, y2)
                    if ok:
                        adj[x2] |= (1 << y2); adj[y2] |= (1 << x2)
                        improved = True
                        found = True
                        break
                    else:
                        adj[x1] &= ~(1 << y1); adj[y1] &= ~(1 << x1)
                adj[u1] |= (1 << v1); adj[v1] |= (1 << u1)
                adj[u2] |= (1 << v2); adj[v2] |= (1 << u2)
            if found:
                continue

            # 4) Simulated-annealing-style removal to escape (accept a temporary drop)
            found = False
            for _ in range(30):
                if time.time() - start > time_budget:
                    break
                u, v = rng.choice(present)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                # Try to add two missing edges to recover and possibly improve
                added = 0
                for x, y in missing:
                    if added >= 2:
                        break
                    if is_valid_add(adj, x, y):
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
                        added += 1
                if added == 2:
                    improved = True
                    found = True
                    break
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            if found:
                continue

            break

        return adj

    def structured_initial(seed):
        rng = random.Random(seed)
        adj = [0] * n
        # Random base: each vertex connects to ~3 random others
        for i in range(n):
            candidates = [j for j in range(n) if j != i and not ((adj[i] >> j) & 1)]
            rng.shuffle(candidates)
            for j in candidates[:3]:
                if is_valid_add(adj, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def crossover(a, b, rng):
        # Combine disjoint edge subsets from two valid graphs, then fill greedily.
        # Take edges present in exactly one parent (symmetric difference) to avoid
        # re-introducing conflicts, then greedily add any further valid edges.
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                in_a = (a[u] >> v) & 1
                in_b = (b[u] >> v) & 1
                if in_a != in_b:  # disjoint-ish: only one parent has it
                    if is_valid_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return greedy_build(rng.randint(0, 10**9), base_adj=adj)

    def final_polish(adj):
        # Tighter add-only pass with multiple orders to extract remaining gains.
        adj = list(adj)
        for _ in range(3):
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_valid_add(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
            if not changed:
                break
        return adj

    def valid_final(edges):
        # Independent verification that no two vertices share 3 common neighbours.
        adj = [0] * n
        for u, v in edges:
            if u == v:
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            if adj[u] & (1 << u):
                return False
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() > 2:
                    return False
        return True

    best_adj = None
    best_count = -1

    def consider(adj):
        nonlocal best_adj, best_count
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)

    # Initial constructions: structured + greedy fills
    for s in range(6):
        if time.time() > deadline - 8.0:
            break
        base = structured_initial(s)
        adj = greedy_build(s, base_adj=base)
        consider(adj)

    # Local search on the best few
    for s in range(6):
        if time.time() > deadline - 8.0:
            break
        base = structured_initial(s)
        adj = greedy_build(s, base_adj=base)
        rng = random.Random(s * 1000 + 7)
        budget = min(2.0, max(0.3, (deadline - 8.0 - time.time()) * 0.05))
        if budget > 0.1:
            adj_ls = local_search(adj, rng, budget)
            consider(adj_ls)

    # Genetic crossover: combine top graphs via disjoint subsets, then search
    population = []
    for s in range(6):
        base = structured_initial(s)
        population.append(greedy_build(s, base_adj=base))
    # Keep top by count
    population.sort(key=edge_count, reverse=True)
    population = population[:4]

    for gen in range(4):
        if time.time() > deadline - 6.0:
            break
        rng = random.Random(gen * 777 + 1)
        children = []
        for i in range(2):
            if time.time() > deadline - 6.0:
                break
            a, b = rng.sample(population, 2)
            child = crossover(a, b, rng)
            consider(child)
            children.append(child)
        # Local search on children
        for child in children:
            if time.time() > deadline - 6.0:
                break
            rng = random.Random(random.randint(0, 10**9))
            budget = min(1.5, max(0.2, (deadline - 6.0 - time.time()) * 0.04))
            if budget > 0.1:
                consider(local_search(child, rng, budget))
        # Update population with best found
        population.append(best_adj)
        population.sort(key=edge_count, reverse=True)
        population = population[:4]

    # Final polish pass on the best
    if best_adj is not None:
        best_adj = final_polish(best_adj)
        consider(best_adj)

    if best_adj is None:
        best_adj = greedy_build(0)

    edges = adj_to_edges(best_adj)
    # Safety: if verification fails, fall back to a simple greedy result
    if not valid_final(edges):
        fb = greedy_build(0)
        edges = adj_to_edges(fb)
    return edges
