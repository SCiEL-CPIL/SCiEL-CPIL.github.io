def construct():
    import time
    import random
    import sys
    import numpy as np

    start_time = time.time()
    DEADLINE = 142.0  # Safety margin before 150s kill

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

    def max_clique_bk(adj, n, active_mask):
        """Bron-Kerbosch with pivot and coloring to find max clique."""
        if active_mask == 0:
            return []

        best = [0]
        best_clique = []

        # Get list of active vertices sorted by degree descending
        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb

        if not verts:
            return []

        verts.sort(key=lambda v: (adj[v] & active_mask).bit_count(), reverse=True)

        def color_sort(P_mask):
            pm = P_mask
            verts_p = []
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                verts_p.append(idx)
                pm ^= lsb

            if not verts_p:
                return []

            verts_p.sort(key=lambda v: (adj[v] & P_mask).bit_count(), reverse=True)

            v_color = {}
            result = []
            for v in verts_p:
                used_colors = 0
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    if u in v_color:
                        used_colors |= (1 << v_color[u])
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

            # Pivot: choose u in P∪X with max |N(u)∩P|
            union = P | X
            pivot = -1
            max_deg = -1
            m = union
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                deg = (adj[u] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
                m ^= lsb

            if pivot == -1:
                candidates = P
            else:
                candidates = P & ~(adj[pivot])

            if candidates == 0:
                return

            colored = color_sort(candidates)
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

    def find_all_maximal_cliques(adj, n, active_mask, max_cliques=500):
        """Find up to max_cliques maximal cliques using BK with pivot."""
        cliques = []

        def bk(R, P, X):
            if time_up() or len(cliques) >= max_cliques:
                return

            if P == 0 and X == 0:
                if len(R) >= 2:
                    cliques.append(list(R))
                return

            # Pivot
            union = P | X
            pivot = -1
            max_deg = -1
            m = union
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                deg = (adj[u] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
                m ^= lsb

            if pivot == -1:
                candidates = P
            else:
                candidates = P & ~(adj[pivot])

            m = candidates
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb

                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()

                if time_up() or len(cliques) >= max_cliques:
                    return

        bk([], active_mask, 0)
        return cliques

    def is_clique(adj, vertices):
        """Check if vertices form a clique."""
        for i, u in enumerate(vertices):
            for v in vertices[i+1:]:
                if not (adj[u] & (1 << v)):
                    return False
        return True

    def clique_edges_mask(adj, vertices):
        """Return bitmask of edges covered by clique (for each vertex)."""
        masks = [0] * len(adj)
        for i, u in enumerate(vertices):
            for v in vertices[i+1:]:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
        return masks

    def solve_instance(n, p, seed, time_budget=70.0):
        adj = generate_graph(n, p, seed)
        rem = adj[:]  # remaining edges
        cliques = []

        # Compute initial active mask
        active = 0
        for i in range(n):
            if rem[i]:
                active |= (1 << i)

        # Phase 1: Greedy max-clique extraction
        instance_start = time.time()
        while active and not time_up() and (time.time() - instance_start) < time_budget:
            clique = max_clique_bk(rem, n, active)

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

            cliques.append(clique)

            # Remove clique edges from rem
            for u in clique:
                for v in clique:
                    if u != v:
                        rem[u] &= ~(1 << v)

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

        # Phase 2: Local improvement - try to merge pairs of cliques
        # If two cliques' union is a clique, and no other clique covers any edge
        # between V(C1)\V(C2) and V(C2)\V(C1), we can merge them.
        if not time_up():
            improved = True
            while improved and not time_up():
                improved = False
                new_cliques = list(cliques)
                for i in range(len(new_cliques)):
                    if improved:
                        break
                    for j in range(i + 1, len(new_cliques)):
                        if improved:
                            break
                        c1 = set(new_cliques[i])
                        c2 = set(new_cliques[j])
                        union = c1 | c2
                        
                        # Optimization: Only attempt merge if union size is small enough 
                        # or if we expect benefit. Here we just check validity.
                        
                        # Check if union is a clique
                        verts = list(union)
                        valid_clique = True
                        for a in range(len(verts)):
                            for b in range(a + 1, len(verts)):
                                if not (adj[verts[a]] & (1 << verts[b])):
                                    valid_clique = False
                                    break
                            if not valid_clique:
                                break
                        
                        if not valid_clique:
                            continue

                        # Check for edge conflicts with other cliques
                        # New edges covered by merge are between c1_only and c2_only
                        c1_only = c1 - c2
                        c2_only = c2 - c1
                        
                        conflict = False
                        for k in range(len(new_cliques)):
                            if k == i or k == j:
                                continue
                            ck = set(new_cliques[k])
                            # If ck has a vertex in c1_only AND a vertex in c2_only,
                            # then ck covers an edge between them, which is now covered by the merge too.
                            if (ck & c1_only) and (ck & c2_only):
                                conflict = True
                                break
                        
                        if not conflict:
                            new_cliques[i] = verts
                            del new_cliques[j]
                            cliques = new_cliques
                            improved = True
                            break

        return cliques

    c1 = solve_instance(32, 0.6, 601, time_budget=60.0)
    c2 = solve_instance(64, 0.55, 602, time_budget=60.0)

    return [c1, c2]
