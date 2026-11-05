def construct():
    import time
    import random
    import math

    start_time = time.time()
    time_limit = 148.0  # Safety margin before 150s kill

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique_branch_and_bound(adj, n, active_mask, time_limit):
        """
        Find a maximum clique in the subgraph induced by active_mask.
        Uses Bron-Kerbosch with coloring bound.
        """
        if active_mask == 0:
            return []

        best_clique = []
        best_size = 0

        # Pre-compute degrees within active_mask for initial sorting
        def get_degree(v):
            return (adj[v] & active_mask).bit_count()

        # Initial vertex ordering by degree (descending)
        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb

        verts.sort(key=get_degree, reverse=True)

        def color_sort(P_mask):
            """
            Greedy coloring of P_mask to get an upper bound on clique size.
            Returns list of (color, vertex) sorted by color.
            """
            if P_mask == 0:
                return []

            verts_p = []
            m = P_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                verts_p.append(idx)
                m ^= lsb

            # Order by degree within P
            verts_p.sort(key=lambda v: (adj[v] & P_mask).bit_count(), reverse=True)

            v_color = {}
            colored = []
            for v in verts_p:
                used_colors = set()
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    if u in v_color:
                        used_colors.add(v_color[u])
                    nm ^= lsb

                c = 1
                while c in used_colors:
                    c += 1
                v_color[v] = c
                colored.append((c, v))

            # Sort by color ascending
            colored.sort(key=lambda x: x[0])
            return colored

        def bron_kerbosch(R, P, X):
            nonlocal best_clique, best_size

            if P == 0:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = list(R)
                return

            # Check time
            if time.time() > time_limit:
                return

            colored = color_sort(P)

            for color, v in reversed(colored):
                # Prune: if max possible clique size <= best_size, stop
                if len(R) + color < best_size:
                    break

                R.append(v)
                P_new = P & adj[v]
                X_new = X & adj[v]

                bron_kerbosch(R, P_new, X_new)

                R.pop()
                P ^= (1 << v)
                X |= (1 << v)

        bron_kerbosch([], active_mask, 0)
        return best_clique

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        rem_adj = adj[:]
        cliques = []

        active_mask = (1 << n) - 1

        # Find initial max clique
        clique = max_clique_branch_and_bound(rem_adj, n, active_mask, time_limit)

        if not clique:
            # Fallback: if no clique found (shouldn't happen if edges exist), pick edges
            for u in range(n):
                neigh = rem_adj[u]
                while neigh:
                    lsb = neigh & -neigh
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
                    neigh ^= lsb
            return cliques

        # Greedy loop: repeatedly find max clique in remaining graph
        while True:
            # Check if any edges remain
            has_edge = False
            m = active_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if rem_adj[u] & active_mask:
                    has_edge = True
                    break
                m ^= lsb

            if not has_edge:
                break

            # Check time
            if time.time() > time_limit - 5.0:
                # Fallback: cover remaining edges as pairs
                for u in range(n):
                    neigh = rem_adj[u] & active_mask
                    while neigh:
                        lsb = neigh & -neigh
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_adj[u] &= ~(1 << v)
                            rem_adj[v] &= ~(1 << u)
                        neigh ^= lsb
                return cliques

            # Find max clique in remaining graph
            clique = max_clique_branch_and_bound(rem_adj, n, active_mask, time_limit - 5.0)

            if not clique:
                # Fallback: pick any edge
                m = active_mask
                edge_found = None
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    neighbors = rem_adj[u] & active_mask
                    if neighbors:
                        lsb_n = neighbors & -neighbors
                        v = lsb_n.bit_length() - 1
                        edge_found = [u, v]
                        break
                    m ^= lsb

                if edge_found:
                    clique = edge_found
                else:
                    break

            # Add clique to result
            cliques.append(clique)

            # Remove edges in clique from rem_adj
            for u in clique:
                for v in clique:
                    if u != v:
                        rem_adj[u] &= ~(1 << v)

            # Update active_mask
            new_active = 0
            for i in range(n):
                if rem_adj[i]:
                    new_active |= (1 << i)
            active_mask = new_active

        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
