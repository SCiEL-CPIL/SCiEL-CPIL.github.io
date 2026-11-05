def construct():
    import time
    import random
    import sys
    import numpy as np

    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

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

    def max_clique_bk(adj, n, active_mask, priority_order=None):
        """Bron-Kerbosch with coloring to find max clique in subgraph induced by active_mask."""
        if active_mask == 0:
            return []

        best = [0]  # best clique size found
        best_clique = []

        # Get list of active vertices
        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb

        if not verts:
            return []

        # Sort by priority if given, otherwise by degree
        if priority_order is not None:
            # priority_order is a dict mapping vertex -> priority (higher is better)
            verts.sort(key=lambda v: priority_order.get(v, 0), reverse=True)
        else:
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

            # Use a list indexed by vertex for color storage (faster than dict)
            v_color = [0] * n
            result = []
            for v in verts_p:
                # Collect used colors via bitmask
                used_colors = 0
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    cu = v_color[u]
                    if cu:
                        used_colors |= (1 << cu)
                    nm ^= lsb
                c = 1
                while (used_colors >> c) & 1:
                    c += 1
                v_color[v] = c
                result.append((c, v))

            result.sort(key=lambda x: x[0])
            return result

        def bk(R, P, X):
            nonlocal best, best_clique

            if P == 0 and X == 0:
                if len(R) > best[0]:
                    best[0] = len(R)
                    best_clique = list(R)
                return

            if time_up():
                return

            colored = color_sort(P)
            for color, v in reversed(colored):
                if len(R) + color <= best[0]:
                    break

                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P ^= (1 << v)
                X |= (1 << v)

                if time_up():
                    return

        bk([], active_mask, 0)
        return best_clique

    def spectral_priority(adj, n):
        """Compute eigenvector centrality using power iteration for priority ordering."""
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    A[u, v] = 1.0
                    A[v, u] = 1.0
        
        # Power iteration for eigenvector centrality
        x = np.ones(n, dtype=np.float64)
        x /= np.linalg.norm(x)
        
        for _ in range(50):  # 50 iterations should be enough for convergence
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm < 1e-10:
                break
            x_new /= norm
            if np.linalg.norm(x_new - x) < 1e-8:
                x = x_new
                break
            x = x_new
        
        # Return priority dict
        return {i: float(x[i]) for i in range(n)}

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        rem = adj[:]  # remaining edges
        cliques = []

        # Compute initial active mask
        active = 0
        for i in range(n):
            if rem[i]:
                active |= (1 << i)

        # Compute spectral priority for initial ordering
        priority = spectral_priority(adj, n)

        while active and not time_up():
            # Find max clique in remaining graph with priority ordering
            clique = max_clique_bk(rem, n, active, priority)

            if not clique:
                # Shouldn't happen if active has edges, but fallback
                # Find any edge
                found = False
                m = active
                while m and not found:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    nb = rem[u] & active
                    if nb:
                        lsb2 = nb & -nb
                        v = lsb2.bit_length() - 1
                        clique = [u, v]
                        found = True
                    m ^= lsb
                if not found:
                    break

            cliques.append(clique)

            # Remove clique edges from rem
            if len(clique) >= 2:
                # Build mask of all vertices in clique
                clique_mask = 0
                for v in clique:
                    clique_mask |= (1 << v)
                for u in clique:
                    rem[u] &= ~clique_mask

            # Update active mask
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

            # Update priority based on remaining graph
            if active:
                priority = spectral_priority(rem, n)

        # Fallback: cover remaining edges as pairs
        if active:
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

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
