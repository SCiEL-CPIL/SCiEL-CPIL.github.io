<<<<<<< SEARCH
    def _solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # Build neighbor sets
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        # Try multiple strategies and keep best
        best_cliques = None
        best_count = float('inf')

        # Strategy 1: Process I-vertices in decreasing degree order
        I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)

        for strategy_idx, I_order in enumerate([I_verts, list(reversed(I_verts)), sorted(range(k, n))]):
            if _time_up():
                break

            used_cc = set()
            cliques = []

            for v in I_order:
                neigh = list(N_C[v])
                d = len(neigh)
                if d == 0:
                    continue

                # Greedy packing: try to form largest possible cliques
                remaining = list(neigh)
                while remaining:
                    # Start with first element
                    group = [remaining.pop(0)]

                    # Try to add more elements that don't conflict
                    i = 0
                    while i < len(remaining):
                        cand = remaining[i]
                        ok = True
                        for g in group:
                            e = (cand, g) if cand < g else (g, cand)
                            if e in used_cc:
                                ok = False
                                break
                        if ok:
                            group.append(cand)
                            remaining.pop(i)
                        else:
                            i += 1

                    if len(group) == 1:
                        cliques.append([v, group[0]])
                    else:
                        cliques.append([v] + group)
                        for i in range(len(group)):
                            for j in range(i + 1, len(group)):
                                a, b = group[i], group[j]
                                e = (a, b) if a < b else (b, a)
                                used_cc.add(e)

            # Cover remaining C-C edges via coloring
            H_adj = [set() for _ in range(k)]
            for (a, b) in used_cc:
                H_adj[a].add(b)
                H_adj[b].add(a)

            uncolored = set(range(k))
            while uncolored:
                # Pick vertex with max degree in remaining subgraph
                best_v = None
                best_deg = -1
                for v in uncolored:
                    deg = len(H_adj[v] & uncolored)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v

                color_class = [best_v]
                uncolored.discard(best_v)

                # Add compatible vertices
                for u in list(uncolored):
                    if u not in uncolored:
                        continue
                    compatible = all(u not in H_adj[c] for c in color_class)
                    if compatible:
                        color_class.append(u)
                        uncolored.discard(u)

                if len(color_class) >= 2:
                    cliques.append(sorted(color_class))

            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

        if best_cliques is None:
            # Fallback
            best_cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    best_cliques.append([u, v])

        return best_cliques
=======
    def _solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # Build neighbor sets
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        # Try multiple strategies and keep best
        best_cliques = None
        best_count = float('inf')

        # Strategy 1: Process I-vertices in decreasing degree order
        I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)

        I_orders = [I_verts, list(reversed(I_verts)), sorted(range(k, n))]

        for strategy_idx, I_order in enumerate(I_orders):
            if _time_up():
                break

            used_cc = set()
            cliques = []

            for v in I_order:
                neigh = list(N_C[v])
                d = len(neigh)
                if d == 0:
                    continue

                # Greedy packing: try to form largest possible cliques
                remaining = list(neigh)
                while remaining:
                    # Start with first element
                    group = [remaining.pop(0)]

                    # Try to add more elements that don't conflict
                    i = 0
                    while i < len(remaining):
                        cand = remaining[i]
                        ok = True
                        for g in group:
                            e = (cand, g) if cand < g else (g, cand)
                            if e in used_cc:
                                ok = False
                                break
                        if ok:
                            group.append(cand)
                            remaining.pop(i)
                        else:
                            i += 1

                    if len(group) == 1:
                        cliques.append([v, group[0]])
                    else:
                        cliques.append([v] + group)
                        for i in range(len(group)):
                            for j in range(i + 1, len(group)):
                                a, b = group[i], group[j]
                                e = (a, b) if a < b else (b, a)
                                used_cc.add(e)

            # Cover remaining C-C edges via DSATUR coloring on the residual graph
            # Residual C-C edges are those NOT in used_cc
            # We need to partition the set of residual edges into cliques
            # Equivalent to coloring the complement: two C-vertices can be in same clique
            # iff their edge is NOT residual (i.e., already used)
            # So build graph G_res where edge (a,b) exists iff (a,b) is a residual C-C edge
            # We want to find minimum number of cliques covering all edges of G_res
            # This is equivalent to finding minimum number of independent sets in the
            # complement of G_res restricted to edges, i.e., coloring G_res.
            # Wait: a clique in the original graph covers a set of C-C edges that form a clique.
            # In G_res (residual edges), we want to partition edges into cliques.
            # Two vertices a,b can be in the same clique iff all pairs in the clique have
            # their edge in G_res. So we're finding a clique cover of G_res.
            # Clique cover of G_res = coloring of complement of G_res.
            # Complement of G_res on vertex set {0..k-1}: edge (a,b) exists iff (a,b) NOT in G_res
            # i.e., (a,b) IS in used_cc (or a==b).
            
            # Build adjacency for coloring: two vertices are "compatible" (can share a color)
            # iff their edge is in used_cc (already covered)
            # We want to color G_res so that adjacent vertices (in G_res) get different colors.
            # Color class = set of vertices with no G_res edges between them = clique in complement = 
            # all pairs have edges in used_cc.
            
            # G_res adjacency
            G_res_adj = [set() for _ in range(k)]
            for a in range(k):
                for b in range(a+1, k):
                    if (a, b) not in used_cc:
                        G_res_adj[a].add(b)
                        G_res_adj[b].add(a)
            
            # DSATUR coloring on G_res
            uncolored = set(range(k))
            color_of = {}
            colors = []  # list of color classes (sets of vertices)
            
            while uncolored:
                # DSATUR: pick vertex with max saturation degree, break ties by degree
                best_v = None
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    # Saturation: number of distinct colors among neighbors
                    sat_colors = set()
                    for nb in G_res_adj[v]:
                        if nb in color_of:
                            sat_colors.add(color_of[nb])
                    sat = len(sat_colors)
                    deg = len(G_res_adj[v] & uncolored)
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_v = v
                
                # Assign smallest available color
                used_colors = set()
                for nb in G_res_adj[best_v]:
                    if nb in color_of:
                        used_colors.add(color_of[nb])
                
                c = 0
                while c in used_colors:
                    c += 1
                color_of[best_v] = c
                uncolored.discard(best_v)
            
            # Build color classes
            max_color = max(color_of.values()) if color_of else 0
            for c in range(max_color + 1):
                cls = sorted(v for v in range(k) if color_of[v] == c)
                if len(cls) >= 2:
                    cliques.append(cls)

            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

        if best_cliques is None:
            # Fallback
            best_cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    best_cliques.append([u, v])

        return best_cliques
>>>>>>> REPLACE
