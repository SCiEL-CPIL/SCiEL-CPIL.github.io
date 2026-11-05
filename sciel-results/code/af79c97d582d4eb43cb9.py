import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    n = 20
    rng = random.Random(12345)

    # Precompute all triples and map vertices to their triples
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    num_triples = len(triples)

    # vertex_triples[v] = list of (triple_idx, other1, other2) for triples containing v
    vertex_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        vertex_triples[i].append((idx, j, k))
        vertex_triples[j].append((idx, i, k))
        vertex_triples[k].append((idx, i, j))

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]

    def bit_count(x):
        return x.bit_count()

    def count_edges(adj):
        return sum(bit_count(adj[i]) for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) to a valid graph keeps it K3,3-free.
        Only triples containing u or v can gain a common neighbor (by exactly 1).
        So we need: no triple containing u has v as a common neighbor AND already has >=2 others.
        Similarly for v.
        """
        au = adj[u]
        av = adj[v]

        # For each triple (u, y, z): v becomes a common neighbor if v in N(y) and v in N(z).
        # If so, the triple's common-neighbor count increases by 1.
        # We need the old count (without v) to be <= 1, i.e., bit_count(au & adj[y] & adj[z]) <= 1.
        for idx, y, z in vertex_triples[u]:
            if (av >> y) & 1 and (av >> z) & 1:
                # v is a common neighbor of y and z, so adding (u,v) adds v to N(u)&N(y)&N(z)
                # Old common neighbors of (u,y,z) = au & adj[y] & adj[z] (v not in au yet)
                if bit_count(au & adj[y] & adj[z]) > 1:
                    return False

        # For each triple (v, y, z): u becomes a common neighbor if u in N(y) and u in N(z).
        for idx, y, z in vertex_triples[v]:
            if (au >> y) & 1 and (au >> z) & 1:
                if bit_count(av & adj[y] & adj[z]) > 1:
                    return False

        return True

    def is_valid(adj):
        """Full validity check for all triples."""
        for (i, j, k) in triples:
            cm = adj[i] & adj[j] & adj[k]
            mask = (1 << i) | (1 << j) | (1 << k)
            cm &= ~mask
            if bit_count(cm) >= 3:
                return False
        return True

    best_edges = 0
    best_adj = None

    def try_update_best(adj):
        nonlocal best_edges, best_adj
        ce = count_edges(adj)
        if ce > best_edges:
            best_edges = ce
            best_adj = [a for a in adj]

    def greedy_add(adj, cur_edges):
        """Add all possible edges in best-improvement order."""
        improved = True
        while improved:
            improved = False
            # Try to add edges; pick the one that enables most further additions
            # Simple: just try all, add if valid
            for (u, v) in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    cur_edges += 1
                    improved = True
        return cur_edges

    def local_search(adj, cur_edges, time_limit):
        """Systematic best-improvement local search: add-only, then swaps."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Add-only (try all missing edges, add if valid)
            for (u, v) in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    cur_edges += 1
                    improved = True
            try_update_best(adj)

            if time.time() > time_limit:
                break

            # Phase 2: Swaps (remove one edge, try to add a different one)
            # For each existing edge, remove it, then try to add any missing edge
            for (u1, v1) in all_edges:
                if time.time() > time_limit:
                    break
                if not ((adj[u1] >> v1) & 1):
                    continue

                remove_edge(adj, u1, v1)
                cur_edges -= 1

                found_swap = False
                for (u2, v2) in all_edges:
                    if (u2, v2) == (u1, v1):
                        continue
                    if (adj[u2] >> v2) & 1:
                        continue
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        cur_edges += 1
                        found_swap = True
                        improved = True
                        try_update_best(adj)
                        break

                if not found_swap:
                    # Revert
                    add_edge(adj, u1, v1)
                    cur_edges += 1

                if found_swap:
                    # After a swap, restart the search from this state
                    # (the graph changed, so re-run phases)
                    break

        return cur_edges

    def two_opt_search(adj, cur_edges, time_limit):
        """Enhanced local search with 2-edge swaps and random perturbations."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Add-only
            for (u, v) in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    cur_edges += 1
                    improved = True
            try_update_best(adj)

            if time.time() > time_limit:
                break

            # Phase 2: Single-edge swaps
            for (u1, v1) in all_edges:
                if time.time() > time_limit:
                    break
                if not ((adj[u1] >> v1) & 1):
                    continue

                remove_edge(adj, u1, v1)
                cur_edges -= 1

                found_swap = False
                for (u2, v2) in all_edges:
                    if (u2, v2) == (u1, v1):
                        continue
                    if (adj[u2] >> v2) & 1:
                        continue
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        cur_edges += 1
                        found_swap = True
                        improved = True
                        try_update_best(adj)
                        break

                if not found_swap:
                    add_edge(adj, u1, v1)
                    cur_edges += 1

                if found_swap:
                    break

            if time.time() > time_limit:
                break

            # Phase 3: Random 2-edge swaps (perturbation to escape local optima)
            if not improved:
                # Try a few random 2-edge swaps
                for _ in range(100):
                    if time.time() > time_limit:
                        break
                    # Pick two random existing edges to remove
                    e1 = rng.choice(all_edges)
                    e2 = rng.choice(all_edges)
                    if e1 == e2:
                        continue
                    if not ((adj[e1[0]] >> e1[1]) & 1) or not ((adj[e2[0]] >> e2[1]) & 1):
                        continue

                    remove_edge(adj, e1[0], e1[1])
                    remove_edge(adj, e2[0], e2[1])
                    cur_edges -= 2

                    # Try to add two new edges
                    added = 0
                    for (u, v) in all_edges:
                        if (u, v) == e1 or (u, v) == e2:
                            continue
                        if (adj[u] >> v) & 1:
                            continue
                        if is_valid_add(adj, u, v):
                            add_edge(adj, u, v)
                            cur_edges += 1
                            added += 1
                            if added == 2:
                                break

                    if added == 2:
                        improved = True
                        try_update_best(adj)
                    else:
                        # Revert
                        add_edge(adj, e1[0], e1[1])
                        add_edge(adj, e2[0], e2[1])
                        cur_edges += 2

        return cur_edges

    # === Main search loop ===
    num_starts = 30
    per_start_time = 3.0

    for start_idx in range(num_starts):
        if time.time() > deadline - 5:
            break

        # Create diverse starting graphs
        adj = [0] * n

        if start_idx % 3 == 0:
            # Random dense start
            for (u, v) in all_edges:
                if rng.random() < 0.6:
                    add_edge(adj, u, v)
        elif start_idx % 3 == 1:
            # Paley-like structured start (edges based on quadratic residues)
            for i in range(n):
                for j in range(i+1, n):
                    diff = (j - i) % n
                    # Use a simple heuristic: include edge if diff is "small" or random
                    if diff <= 3 or rng.random() < 0.3:
                        add_edge(adj, i, j)
        else:
            # Sparse random start
            for (u, v) in all_edges:
                if rng.random() < 0.35:
                    add_edge(adj, u, v)

        # Clean: remove edges until valid
        max_clean = 500
        for _ in range(max_clean):
            if time.time() > deadline - 5:
                break
            violated = False
            for (i, j, k) in triples:
                cm = adj[i] & adj[j] & adj[k]
                mask = (1 << i) | (1 << j) | (1 << k)
                cm &= ~mask
                if bit_count(cm) >= 3:
                    # Remove a random edge between the triple and a common neighbor
                    v_list = [x for x in range(n) if (cm >> x) & 1]
                    u_list = [i, j, k]
                    ru = rng.choice(u_list)
                    rv = rng.choice(v_list)
                    remove_edge(adj, ru, rv)
                    violated = True
                    break
            if not violated:
                break

        # Verify validity
        if not is_valid(adj):
            continue

        cur_edges = count_edges(adj)
        try_update_best(adj)

        # Greedy add
        if time.time() < deadline - 5:
            cur_edges = greedy_add(adj, cur_edges)
            try_update_best(adj)

        # Local search
        if time.time() < deadline - 5:
            start_ls_time = time.time()
            ls_budget = min(per_start_time, deadline - 5 - start_ls_time)
            if ls_budget > 0.5:
                cur_edges = two_opt_search(adj, cur_edges, start_time + start_ls_time + ls_budget)
                try_update_best(adj)

    # Final: try one more intensive local search on the best graph
    if best_adj is not None and time.time() < deadline - 10:
        adj = [a for a in best_adj]
        cur_edges = count_edges(adj)
        final_time = deadline - 5
        cur_edges = two_opt_search(adj, cur_edges, final_time)
        try_update_best(adj)

    # Convert to edge list
    if best_adj is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges
