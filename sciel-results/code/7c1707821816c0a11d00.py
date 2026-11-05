def construct():
    import time
    import random
    import sys

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

    def max_clique_bk(adj, n, active_mask, must_include=None):
        """Bron-Kerbosch with coloring to find max clique in subgraph induced by active_mask.
        If must_include is given, the clique must contain all vertices in that set."""
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

        # Precompute degree in active_mask for sorting
        deg_in_active = {}
        for v in verts:
            deg_in_active[v] = (adj[v] & active_mask).bit_count()
        verts.sort(key=lambda v: deg_in_active[v], reverse=True)

        def color_sort(P_mask):
            """Greedy coloring of vertices in P_mask using bitmasks for speed."""
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
            deg_in_P = {}
            for v in verts_p:
                deg_in_P[v] = (adj[v] & P_mask).bit_count()
            verts_p.sort(key=lambda v: deg_in_P[v], reverse=True)

            # Use a list indexed by vertex for color lookup (faster than dict for small n)
            v_color = [-1] * n
            result = []
            for v in verts_p:
                used_colors = 0
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    cu = v_color[u]
                    if cu >= 0:
                        used_colors |= (1 << cu)
                    nm ^= lsb
                c = 0
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
                if len(R) + color + 1 <= best[0]:
                    break

                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P ^= (1 << v)
                X |= (1 << v)

                if time_up():
                    return

        if must_include:
            # All must_include vertices must be in the clique
            # Start with R = must_include, P = intersection of neighbors, X = empty
            R0 = list(must_include)
            P0 = active_mask
            X0 = 0
            for v in R0:
                P0 &= adj[v]
                X0 |= (1 << v)
            # Remove R0 vertices from P0
            for v in R0:
                P0 &= ~(1 << v)
            bk(R0, P0, X0)
        else:
            bk([], active_mask, 0)
        return best_clique

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        rem = adj[:]  # remaining edges
        cliques = []

        # Compute initial active mask
        active = 0
        for i in range(n):
            if rem[i]:
                active |= (1 << i)

        while active and not time_up():
            # Find the vertex with maximum remaining degree as seed
            best_seed = -1
            best_deg = -1
            m = active
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                d = (rem[u] & active).bit_count()
                if d > best_deg:
                    best_deg = d
                    best_seed = u
                m ^= lsb

            if best_seed < 0:
                break

            # Find max clique that must include the seed
            clique = max_clique_bk(rem, n, active, must_include=[best_seed])

            if not clique:
                # Fallback: find any edge
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

            # Local improvement: try to extend the clique by adding vertices
            clique_set = set(clique)
            clique_mask = 0
            for v in clique:
                clique_mask |= (1 << v)
            
            # Try to add vertices that are connected to all current clique members
            changed = True
            while changed and not time_up():
                changed = False
                # Find candidates: vertices connected to all clique members
                cand_mask = active
                for v in clique:
                    cand_mask &= rem[v]
                cand_mask &= ~clique_mask
                
                cm = cand_mask
                while cm:
                    lsb = cm & -cm
                    u = lsb.bit_length() - 1
                    # Check if u is connected to all clique members
                    ok = True
                    for v in clique:
                        if not (rem[u] >> v) & 1:
                            ok = False
                            break
                    if ok:
                        clique.append(u)
                        clique_mask |= (1 << u)
                        changed = True
                        break
                    cm ^= lsb

            cliques.append(clique)

            # Remove clique edges from rem using bit operations
            for u in clique:
                mask = 0
                for v in clique:
                    if u != v:
                        mask |= (1 << v)
                rem[u] &= ~mask

            # Update active mask
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

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
