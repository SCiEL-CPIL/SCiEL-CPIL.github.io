def construct():
    import time
    import random
    import numpy as np
    from scipy.sparse.linalg import eigsh

    start_time = time.time()
    DEADLINE = start_time + 140.0  # 10s safety margin before 150s kill

    def time_up():
        return (time.time() - start_time) > DEADLINE

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique_bk(adj, n, active_mask, seed_vertex):
        """
        Find max clique containing seed_vertex in the subgraph induced by active_mask.
        Uses Bron-Kerbosch with pivot and coloring.
        """
        if active_mask == 0:
            return []

        # Ensure seed is in active
        if not (active_mask & (1 << seed_vertex)):
            return []

        best_clique = [seed_vertex]
        best_size = 1

        # Vertices in active_mask
        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb

        if not verts:
            return []

        # Sort by degree in active_mask descending for better coloring/order
        verts.sort(key=lambda v: (adj[v] & active_mask).bit_count(), reverse=True)

        def color_sort(P_mask):
            """Greedy coloring of vertices in P_mask, return list of (color, vertex) sorted by color."""
            pm = P_mask
            verts_p = []
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                verts_p.append(idx)
                pm ^= lsb

            if not verts_p:
                return []

            # Sort by degree in P_mask
            verts_p.sort(key=lambda v: (adj[v] & P_mask).bit_count(), reverse=True)

            v_color = {}
            result = []
            for v in verts_p:
                used = set()
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    if u in v_color:
                        used.add(v_color[u])
                    nm ^= lsb
                c = 1
                while c in used:
                    c += 1
                v_color[v] = c
                result.append((c, v))

            result.sort(key=lambda x: x[0])
            return result

        def bk(R, P, X):
            nonlocal best_clique, best_size

            if P == 0 and X == 0:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = list(R)
                return

            if time_up():
                return

            colored = color_sort(P)
            for color, v in reversed(colored):
                if len(R) + color <= best_size:
                    break

                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P ^= (1 << v)
                X |= (1 << v)

                if time_up():
                    return

        # Initialize with seed
        # P = neighbors of seed in active_mask
        # X = empty
        P = adj[seed_vertex] & active_mask
        X = 0
        R = [seed_vertex]
        
        bk(R, P, X)
        
        return best_clique

    def solve_instance(n, p, seed):
        adj_orig = generate_graph(n, p, seed)
        rem = adj_orig[:]  # Copy of adjacency for remaining edges
        cliques = []

        # Compute initial spectral centrality
        # Use adjacency matrix for eigsh
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            mask = adj_orig[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                A[u, v] = 1.0
                mask ^= lsb
        A = (A + A.T) / 2.0

        try:
            vals, vecs = eigsh(A, k=1, which='LA')
            centrality = np.abs(vecs[:, 0])
            if np.max(centrality) > 0:
                centrality = centrality / np.max(centrality)
            else:
                centrality = np.ones(n) / n
        except Exception:
            centrality = np.ones(n) / n

        # Track active vertices (those with at least one remaining edge)
        active_mask = 0
        for i in range(n):
            if rem[i]:
                active_mask |= (1 << i)

        while active_mask and not time_up():
            # Select seed vertex: highest centrality among active vertices
            # We need to iterate active vertices to find max centrality
            best_seed = -1
            max_cent = -1.0
            
            m = active_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if centrality[u] > max_cent:
                    max_cent = centrality[u]
                    best_seed = u
                m ^= lsb
            
            if best_seed == -1:
                break

            # Find max clique containing best_seed in the subgraph of active_mask
            # Note: The clique must consist of vertices in active_mask
            clique = max_clique_bk(rem, n, active_mask, best_seed)

            if not clique or len(clique) < 2:
                # Fallback: pick any neighbor
                nb = rem[best_seed] & active_mask
                if nb:
                    lsb = nb & -nb
                    v = lsb.bit_length() - 1
                    clique = [best_seed, v]
                else:
                    # Should not happen if active_mask is correct, but safety
                    break

            cliques.append(clique)

            # Remove edges of the clique from rem
            for u in clique:
                for v in clique:
                    if u != v:
                        rem[u] &= ~(1 << v)

            # Update active_mask
            active_mask = 0
            for i in range(n):
                if rem[i]:
                    active_mask |= (1 << i)

        # Fallback: cover remaining edges with pairs if time runs out or logic fails
        if active_mask:
            # Iterate through remaining edges
            for u in range(n):
                nb = rem[u]
                while nb:
                    lsb = nb & -nb
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        rem[u] &= ~(1 << v)
                        rem[v] &= ~(1 << u)
                    nb ^= lsb

        return cliques

    # Instance 1
    cliques1 = solve_instance(32, 0.6, 601)
    
    # Instance 2
    cliques2 = solve_instance(64, 0.55, 602)
    
    return [cliques1, cliques2]
