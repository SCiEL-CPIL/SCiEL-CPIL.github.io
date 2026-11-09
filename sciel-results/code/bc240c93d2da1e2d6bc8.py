import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def dsatur_color(h_vertices, h_adj, k):
        """DSATUR coloring of graph H. Returns list of color classes (lists of vertices)."""
        if not h_vertices:
            return []
        
        colors = {}
        color_set = {u: set() for u in h_vertices}
        uncolored = set(h_vertices)
        color_count = 0
        
        while uncolored:
            # Pick vertex with highest saturation degree, break ties by degree in H
            best = None
            best_sat = -1
            best_deg = -1
            for x in uncolored:
                sat = len(color_set[x])
                deg = len(h_adj[x] & uncolored)
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best = x
            
            # Find smallest available color
            c = 0
            while c in color_set[best]:
                c += 1
            colors[best] = c
            color_set[best].add(c)
            uncolored.discard(best)
            if c >= color_count:
                color_count = c + 1
            
            # Update saturation of neighbors
            for w in h_adj[best]:
                if w in uncolored:
                    if c not in color_set[w]:
                        color_set[w].add(c)
        
        # Group by color
        color_classes = {}
        for u, c in colors.items():
            if c not in color_classes:
                color_classes[c] = []
            color_classes[c].append(u)
        
        return [color_classes[c] for c in range(color_count)]

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        # Bitmask for clique edges in C
        edge_mask = [0] * k
        for u in range(k):
            for v in range(u + 1, k):
                edge_mask[u] |= (1 << v)
                edge_mask[v] |= (1 << u)
        
        partition = []
        
        # Group independent vertices by their neighborhood bitmask in C
        # This allows us to compute the clique partition once per distinct neighborhood
        neigh_groups = {}  # bitmask -> list of independent vertices
        for v in I:
            N_v = adj[v] & set(C)
            mask = 0
            for u in N_v:
                mask |= (1 << u)
            if mask == 0:
                continue
            if mask not in neigh_groups:
                neigh_groups[mask] = []
            neigh_groups[mask].append(v)
        
        # Track which C-edges have been "used" (covered by a clique involving an I vertex)
        # Actually, we need to track edges that are part of cliques we've already created
        used_c_edges = set()
        
        # Process each distinct neighborhood
        # Sort by size of neighborhood (ascending) to handle smaller ones first
        sorted_masks = sorted(neigh_groups.keys(), key=lambda m: bin(m).count('1'))
        
        for mask in sorted_masks:
            vertices = neigh_groups[mask]
            N_v = set()
            for u in range(k):
                if mask & (1 << u):
                    N_v.add(u)
            
            n_v = len(N_v)
            
            if n_v == 1:
                # Single vertex in neighborhood
                u = next(iter(N_v))
                for v in vertices:
                    partition.append(sorted([v, u]))
                # No C-edges used
                continue
            
            # Build complement graph on N_v: edge (u,w) in complement if NOT edge in G[C]
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i + 1, len(h_vertices)):
                    w = h_vertices[j]
                    # Check if (u,w) is NOT an edge in G[C]
                    if not (edge_mask[u] & (1 << w)):
                        h_adj[u].add(w)
                        h_adj[w].add(u)
            
            # DSATUR coloring of complement graph
            color_classes = dsatur_color(h_vertices, h_adj, k)
            
            # For each color class, create a clique {v} ∪ S for each v in vertices
            for S in color_classes:
                if not S:
                    continue
                S_sorted = sorted(S)
                for v in vertices:
                    clique = S_sorted + [v]
                    partition.append(sorted(clique))
                
                # Mark C-edges within S as used
                for i in range(len(S_sorted)):
                    u = S_sorted[i]
                    for j in range(i + 1, len(S_sorted)):
                        w = S_sorted[j]
                        used_c_edges.add((min(u, w), max(u, w)))
        
        # Remaining edges in C: partition into cliques
        remaining_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in used_c_edges and (edge_mask[u] & (1 << v)):
                    remaining_edges.append((u, v))
        
        rem_set = set(remaining_edges)
        
        while rem_set:
            # Pick edge with highest common neighborhood to grow large cliques
            best_edge = None
            best_score = -1
            for e in rem_set:
                u, v = e
                # Common neighbors in rem_set
                common = 0
                for w in range(k):
                    if w == u or w == v:
                        continue
                    if (min(u, w), max(u, w)) in rem_set and (min(v, w), max(v, w)) in rem_set:
                        common += 1
                if common > best_score:
                    best_score = common
                    best_edge = e
            
            if best_edge is None:
                best_edge = next(iter(rem_set))
            
            u, v = best_edge
            current_clique = {u, v}
            
            # Greedily extend clique
            changed = True
            while changed:
                changed = False
                for w in range(k):
                    if w in current_clique:
                        continue
                    can_add = True
                    for x in current_clique:
                        e = (min(w, x), max(w, x))
                        if e not in rem_set:
                            can_add = False
                            break
                    if can_add:
                        current_clique.add(w)
                        changed = True
            
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
            
            # Remove edges of this clique from rem_set
            for i in range(len(clique_list)):
                a = clique_list[i]
                for j in range(i + 1, len(clique_list)):
                    b = clique_list[j]
                    rem_set.discard((min(a, b), max(a, b)))
        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results
