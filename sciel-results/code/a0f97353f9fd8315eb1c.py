"""Combined program: fast bitmask C6 detection + simulated-annealing kicks + priority ordering + polish."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    n = 40

    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def degree(u):
        return adj_mask[u].bit_count()

    def total_edges():
        return sum(adj_mask[u].bit_count() for u in range(n)) // 2

    def has_c6_fast(u, v):
        """Check if adding edge (u,v) creates a C6 using bitmasks (iterative DFS to depth 4)."""
        # Fast necessary-condition filter: a C6 needs a path u-a-b-c-d-v with d in N(v).
        # If no neighbor of v can reach u in <=3 hops, no C6.
        nv = adj_mask[v] & ~(1 << u)
        if not nv:
            return False

        a = adj_mask[u] & ~(1 << v)
        if not a:
            return False

        # depth 1: a in N(u), a != v
        while a:
            la = a & (-a)
            a ^= la
            ai = la.bit_length() - 1

            nb = adj_mask[ai] & ~(1 << u) & ~(1 << v) & ~(1 << ai)
            b = nb
            while b:
                lb = b & (-b)
                b ^= lb
                bi = lb.bit_length() - 1

                nc = adj_mask[bi] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << bi)
                c = nc
                while c:
                    lc = c & (-c)
                    c ^= lc
                    ci = lc.bit_length() - 1

                    nd = adj_mask[ci] & nv & ~(1 << u) & ~(1 << ai) & ~(1 << bi) & ~(1 << ci)
                    if nd:
                        return True

        return False

    def get_all_edges(mask):
        """Get list of edges from adjacency mask."""
        edges = []
        for u in range(n):
            m = mask[u]
            while m:
                lb = m & (-m)
                m ^= lb
                v = lb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
        return edges

    def compute_priority(adj):
        """Rank missing edges: prefer high-degree endpoints, penalize common neighbors."""
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((1 << j) & adj[i]):
                    du = adj[i].bit_count()
                    dv = adj[j].bit_count()
                    common = (adj[i] & adj[j]).bit_count()
                    score = (du + dv) * 1.0 - common * 2.5 + random.random() * 0.2
                    missing.append((score, i, j))
        missing.sort(reverse=True)
        return [(i, j) for _, i, j in missing]

    def greedy_build(initial_mask, time_limit):
        """Greedy edge addition with C6 checking."""
        adj_mask[:] = list(initial_mask)
        best_mask = list(adj_mask)
        best_count = total_edges()

        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if not ((1 << j) & adj_mask[i])]
        random.shuffle(non_edges)

        for (u, v) in non_edges:
            if time.time() > time_limit:
                break
            if (1 << v) & adj_mask[u]:
                continue
            if has_c6_fast(u, v):
                continue
            add_edge(u, v)
            current_count = total_edges()
            if current_count > best_count:
                best_count = current_count
                best_mask = list(adj_mask)

        return best_mask, best_count

    def local_search_sa(initial_mask, time_limit):
        """Local search with simulated-annealing kicks to escape local optima.
        Uses a C6-check cache per pass so the polish pass reuses checks."""
        adj_mask[:] = list(initial_mask)
        best_mask = list(adj_mask)
        best_count = total_edges()

        temperature = 5.0
        cooling_rate = 0.985
        min_temperature = 0.15

        while time.time() < time_limit - 2:
            # Rank missing edges, then shuffle for diversity
            ranked = compute_priority(adj_mask)
            if not ranked:
                break
            random.shuffle(ranked)

            cache = {}
            added_in_pass = 0
            for (u, v) in ranked:
                if time.time() > time_limit - 1:
                    break
                if (1 << v) & adj_mask[u]:
                    continue
                key = (u, v)
                if key not in cache:
                    cache[key] = has_c6_fast(u, v)
                if cache[key]:
                    continue
                add_edge(u, v)
                added_in_pass += 1
                current_count = total_edges()
                if current_count > best_count:
                    best_count = current_count
                    best_mask = list(adj_mask)

            # Final polish pass: try all missing edges once more (reuses cache)
            if added_in_pass > 0:
                for i in range(n):
                    for j in range(i + 1, n):
                        if time.time() > time_limit - 1:
                            break
                        if (1 << j) & adj_mask[i]:
                            continue
                        key = (i, j)
                        if key not in cache:
                            cache[key] = has_c6_fast(i, j)
                        if cache[key]:
                            continue
                        add_edge(i, j)
                        current_count = total_edges()
                        if current_count > best_count:
                            best_count = current_count
                            best_mask = list(adj_mask)

            if added_in_pass == 0:
                # Stuck: simulated-annealing kick. Remove a random edge (worse move)
                # to escape the local optimum; accept with probability based on temperature.
                current_edges = get_all_edges(adj_mask)
                if not current_edges:
                    break
                u, v = random.choice(current_edges)
                remove_edge(u, v)
                # Cool down
                temperature = max(min_temperature, temperature * cooling_rate)

        return best_mask, best_count

    # Initial state: 8 disjoint K5s (104 edges)
    init_adj = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                init_adj[5 * k + a] |= (1 << (5 * k + b))
                init_adj[5 * k + b] |= (1 << (5 * k + a))

    best_overall_mask = list(init_adj)
    best_overall_count = total_edges()

    # Phase 1: Greedy build from initial state
    t1_end = time.time() + 5
    mask1, count1 = greedy_build(init_adj, t1_end)
    if count1 > best_overall_count:
        best_overall_count = count1
        best_overall_mask = mask1

    # Phase 2: Simulated-annealing local search with restarts
    remaining_time = deadline - time.time() - 5
    num_restarts = max(1, int(remaining_time / 10))  # ~10s per restart

    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break

        if r == 0:
            start_mask = list(best_overall_mask)
        else:
            start_mask = list(best_overall_mask)
            current_edges = get_all_edges(start_mask)
            if current_edges:
                edges_to_remove = random.sample(current_edges, min(3, len(current_edges)))
                for (u, v) in edges_to_remove:
                    start_mask[u] &= ~(1 << v)
                    start_mask[v] &= ~(1 << u)

        t_end = time.time() + min(10, remaining_time - (r * 10))
        if t_end < time.time() + 1:
            break
        mask, count = local_search_sa(start_mask, t_end)

        if count > best_overall_count:
            best_overall_count = count
            best_overall_mask = mask

    # Build final edge list from the best state
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (1 << v) & best_overall_mask[u]:
                edges.append([u, v])

    return edges
# EVOLVE-BLOCK-END
