"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency as bitmasks for fast intersection
        rem_mask = [0] * n
        for u, v in edges:
            rem_mask[u] |= (1 << v)
            rem_mask[v] |= (1 << u)

        remaining_count = len(edges)
        cliques = []
        
        # Helper to get list of neighbors from mask
        def mask_to_list(mask):
            res = []
            while mask:
                lsb = mask & (-mask)
                res.append(lsb.bit_length() - 1)
                mask ^= lsb
            return res

        def find_max_clique():
            """
            Finds a large clique using a branch-and-bound heuristic (Tomita-style).
            Returns a list of vertices forming a clique.
            """
            if not any(rem_mask):
                return []
                
            best_clique = []
            best_size = 0
            
            # Order vertices by degree for better pruning
            vertices = [v for v in range(n) if rem_mask[v]]
            vertices.sort(key=lambda v: bin(rem_mask[v]).count('1'), reverse=True)
            
            # Greedy coloring bound function
            # Colors the subgraph induced by P such that adjacent vertices have different colors.
            # The color number is an upper bound on the clique size in that subgraph.
            def greedy_coloring(P):
                colors = {}
                color_class = []
                for v in P:
                    # Find the smallest color that doesn't conflict with neighbors in P
                    # Neighbors of v in P
                    neigh_in_P = [u for u in P if u != v and (rem_mask[v] >> u) & 1]
                    used_colors = set()
                    for u in neigh_in_P:
                        if u in colors:
                            used_colors.add(colors[u])
                    
                    c = 1
                    while c in used_colors:
                        c += 1
                    colors[v] = c
                    color_class.append(c)
                    
                return colors, color_class

            def expand(R, P):
                nonlocal best_clique, best_size
                
                if not P:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = list(R)
                    return

                # Color P
                colors, color_class = greedy_coloring(P)
                max_color = max(color_class) if color_class else 0
                
                # Prune if current clique size + max color <= best size
                if len(R) + max_color <= best_size:
                    return
                
                # Process vertices in reverse color order (highest color first)
                # This is a heuristic to find large cliques faster
                # We need to group by color or just sort by color
                # Simple approach: sort P by color descending, then process
                # But standard Tomita uses color classes. Let's stick to a simpler version:
                # Iterate through P in order of decreasing color.
                
                P_sorted = sorted(P, key=lambda v: colors[v], reverse=True)
                
                # We will remove vertices from P as we process them
                # To maintain the bound, we should process carefully.
                # Simplified Branch and Bound:
                # Pick a vertex, recurse, remove it.
                # To be efficient, we pick vertices with high colors.
                
                i = 0
                while i < len(P_sorted):
                    v = P_sorted[i]
                    # If even the max color in remaining P_sorted[i:] + len(R) <= best_size, stop
                    # Note: color_class corresponds to the original P, so we need to be careful.
                    # A safer bound: if len(R) + colors[v] <= best_size, we can prune this branch?
                    # Not exactly, but it's a good heuristic.
                    if len(R) + colors[v] <= best_size:
                        break
                    
                    # Add v to R
                    R.append(v)
                    
                    # New P is intersection of P_sorted[i+1:] and neighbors of v
                    new_P = [u for u in P_sorted[i+1:] if (rem_mask[v] >> u) & 1]
                    
                    expand(R, new_P)
                    
                    R.pop()
                    i += 1

            # Start with empty clique and all vertices with degree > 0
            # We can start with the vertex with highest degree to speed up
            if vertices:
                start_v = vertices[0]
                expand([start_v], [u for u in vertices if u != start_v and (rem_mask[start_v] >> u) & 1])
                
            # Fallback if expand didn't find anything (shouldn't happen if edges exist)
            if not best_clique and vertices:
                best_clique = [vertices[0]]
                # Find a neighbor
                for u in vertices:
                    if u != vertices[0] and (rem_mask[vertices[0]] >> u) & 1:
                        best_clique.append(u)
                        break

            return best_clique

        while remaining_count > 0:
            if time.time() > deadline - 10:
                break

            clique = find_max_clique()

            if not clique or len(clique) < 2:
                # Fallback: take a single edge
                found = False
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            found = True
                            break
                        mask ^= lsb
                    if found:
                        break
                if not found:
                    break
                continue

            # Remove all edges within clique from remaining
            removed = 0
            for i, u in enumerate(clique):
                for v in clique[i + 1:]:
                    if (rem_mask[u] >> v) & 1:
                        rem_mask[u] &= ~(1 << v)
                        rem_mask[v] &= ~(1 << u)
                        removed += 1

            if removed == 0:
                # Shouldn't happen, but fallback
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            break
                        mask ^= lsb
                    if remaining_count < len(edges):
                        break
                continue

            cliques.append(clique)
            remaining_count -= removed

        # Handle any remaining edges as individual edge-cliques
        for u in range(n):
            mask = rem_mask[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    rem_mask[u] &= ~(1 << v)
                    rem_mask[v] &= ~(1 << u)
                mask ^= lsb

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            # Emergency fallback
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
# EVOLVE-BLOCK-END
