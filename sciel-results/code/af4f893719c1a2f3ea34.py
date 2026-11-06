import random
import time

def popcount(x):
    return bin(x).count('1')

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def is_valid_addition(adj, u, v):
        # Check if adding edge (u, v) keeps the graph K2,3-free
        # 1. Common neighbors of u and v must be < 3
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False

        # 2. For any w in N(v), v becomes a common neighbor of u and w.
        #    So common neighbors of (u, w) must currently be < 2.
        neighbors_v = adj[v]
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = (lsb.bit_length() - 1)
            if w != u:
                common_uw = adj[u] & adj[w]
                if popcount(common_uw) >= 2:
                    return False
            neighbors_v ^= lsb

        # 3. Similarly for w in N(u)
        neighbors_u = adj[u]
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = (lsb.bit_length() - 1)
            if w != v:
                common_vw = adj[v] & adj[w]
                if popcount(common_vw) >= 2:
                    return False
            neighbors_u ^= lsb

        return True

    def try_add_edge(adj, u, v):
        if is_valid_addition(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    def greedy_build(adj, order):
        for u, v in order:
            try_add_edge(adj, u, v)

    def local_search_add(adj):
        improved = True
        while improved:
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n)
                       if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            for u, v in missing:
                if try_add_edge(adj, u, v):
                    improved = True

    def swap_search(adj, max_iters=200):
        """Try removing one edge and adding up to two others to gain net +1 edge."""
        n_edges = sum(popcount(adj[i]) for i in range(n)) // 2
        best_edges = n_edges
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random existing edge to remove
            existing = [(i, j) for i in range(n) for j in range(i+1, n)
                        if (adj[i] >> j) & 1]
            if not existing:
                break
            u, v = random.choice(existing)
            # Remove edge (u, v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

            # Try to add as many missing edges as possible
            missing = [(i, j) for i in range(n) for j in range(i+1, n)
                       if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            added = 0
            for a, b in missing:
                if try_add_edge(adj, a, b):
                    added += 1

            new_edges = sum(popcount(adj[i]) for i in range(n)) // 2
            if new_edges > best_edges:
                best_edges = new_edges
            else:
                # Revert: remove added edges and restore (u,v)
                # This is complex; simpler to just track best state
                pass

        return best_edges

    best_edges = []
    best_count = 0

    for restart in range(500):
        if time.time() > deadline:
            break

        adj = [0] * n

        # Greedy construction with random edge order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        greedy_build(adj, edges)

        # Local search: add any missing edges
        local_search_add(adj)

        # Swap-based local search: remove one edge, try to add more
        for _ in range(30):
            if time.time() > deadline:
                break
            # Pick a random existing edge to remove
            existing = [(i, j) for i in range(n) for j in range(i+1, n)
                        if (adj[i] >> j) & 1]
            if not existing:
                break
            u, v = random.choice(existing)
            # Remove edge (u, v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n)
                       if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            for a, b in missing:
                try_add_edge(adj, a, b)

            # Run local search again
            local_search_add(adj)

        # Count edges
        edge_count = sum(popcount(adj[i]) for i in range(n)) // 2
        if edge_count > best_count:
            best_count = edge_count
            current_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = (lsb.bit_length() - 1)
                    if i < j:
                        current_edges.append([i, j])
                    mask ^= lsb
            best_edges = current_edges

    return best_edges
