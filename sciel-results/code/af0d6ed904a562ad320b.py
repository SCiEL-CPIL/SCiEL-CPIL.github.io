def construct():
    import time
    from random import Random

    deadline = time.time() + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed):
        cross_edges = instance(n, k, p, seed)
        
        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)
            
        # Track which C-edges have been covered by mixed cliques
        # used_c_edges is a set of frozensets or tuples (u, w) with u < w
        used_c_edges = set()
        
        mixed_cliques = []
        
        # Sort I vertices by degree descending
        I_vertices = sorted(range(k, n), key=lambda x: len(N[x]), reverse=True)
        
        for v in I_vertices:
            if not N[v]:
                continue
                
            neighbors = N[v]
            
            # Build graph of conflicts among neighbors
            # Vertices: neighbors
            # Edge (u, w) exists if (u, w) is in used_c_edges
            # We need to color this graph. Each color class is a set of neighbors that can be in the same mixed clique.
            
            # Simple greedy coloring
            # Order neighbors by degree in conflict graph?
            # For now, just iterate.
            
            color_map = {} # neighbor -> color index
            colors = []    # list of lists of neighbors
            
            # Sort neighbors to ensure deterministic and potentially better coloring
            # Sort by number of conflicts?
            def get_conflict_count(u):
                cnt = 0
                for w in neighbors:
                    if u == w: continue
                    e = (min(u, w), max(u, w))
                    if e in used_c_edges:
                        cnt += 1
                return cnt
            
            sorted_neighbors = sorted(neighbors, key=get_conflict_count, reverse=True)
            
            for u in sorted_neighbors:
                # Find a color that doesn't conflict
                assigned = False
                for i in range(len(colors)):
                    # Check if u conflicts with any vertex in color i
                    conflict = False
                    for w in colors[i]:
                        e = (min(u, w), max(u, w))
                        if e in used_c_edges:
                            conflict = True
                            break
                    if not conflict:
                        colors[i].append(u)
                        color_map[u] = i
                        assigned = True
                        break
                
                if not assigned:
                    colors.append([u])
                    color_map[u] = len(colors) - 1
            
            # Now, for each color class, create a mixed clique
            for c in range(len(colors)):
                S = colors[c]
                if not S: continue
                
                # Add clique {v} + S
                mixed_cliques.append([int(v)] + [int(u) for u in S])
                
                # Mark C-edges within S as used
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        used_c_edges.add(e)

        # Now cover remaining C-edges with pure cliques
        # Remaining edges are all C-edges NOT in used_c_edges
        pure_cliques = []
        
        # Build set of remaining edges
        rem_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                e = (u, w)
                if e not in used_c_edges:
                    rem_edges.add(e)
        
        while rem_edges:
            if time.time() > deadline:
                # Fallback: add edges as 2-cliques
                for e in rem_edges:
                    pure_cliques.append([int(e[0]), int(e[1])])
                break
                
            # Find a maximal clique in the graph defined by rem_edges
            # Vertices involved in rem_edges
            vertices = set()
            for u, w in rem_edges:
                vertices.add(u)
                vertices.add(w)
                
            # Greedy maximal clique
            S = []
            S_set = set()
            cand_set = set(vertices)
            
            while cand_set:
                # Pick vertex with max degree in cand_set (within rem_edges)
                best_v = None
                best_deg = -1
                for w in cand_set:
                    deg = 0
                    for x in cand_set:
                        if x == w: continue
                        e = (min(w, x), max(w, x))
                        if e in rem_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = w
                            
                if best_v is None:
                    break
                
                S.append(best_v)
                S_set.add(best_v)
                
                # Update cand_set
                new_cand = set()
                for w in cand_set:
                    if w == best_v: continue
                    e = (min(best_v, w), max(best_v, w))
                    if e in rem_edges:
                        new_cand.add(w)
                cand_set = new_cand
                
            if S:
                pure_cliques.append([int(u) for u in S])
                # Remove edges in S from rem_edges
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        rem_edges.discard(e)
            else:
                # If no clique found (shouldn't happen if rem_edges not empty), add 2-clique
                # Pick any edge
                e = next(iter(rem_edges))
                pure_cliques.append([int(e[0]), int(e[1])])
                rem_edges.discard(e)

        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for (n, k, p, seed) in specs:
        if time.time() > deadline:
            # Fallback for safety
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
        else:
            result.append(solve_instance(n, k, p, seed))

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
