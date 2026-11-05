"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 140 

    def random_graph(n, p, seed):
        """Generate a random graph and return adjacency as a list of bitmasks."""
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def compute_spectral_order(adj, n):
        """
        Compute eigenvector centrality using power iteration.
        Returns a list of vertices sorted by centrality descending.
        """
        if n == 0:
            return []
        
        # Build dense adjacency matrix for NumPy
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            mask = adj[u]
            # Iterate through set bits
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                A[u, v] = 1.0
                m ^= lsb
        
        # Power iteration for principal eigenvector
        vec = np.ones(n, dtype=np.float64) / n
        for _ in range(50):
            new_vec = A @ vec
            norm = np.linalg.norm(new_vec)
            if norm == 0:
                break
            new_vec /= norm
            if np.linalg.norm(new_vec - vec) < 1e-10:
                break
            vec = new_vec
        
        # Sort vertices by centrality descending
        order = np.argsort(-vec)
        return [int(i) for i in order]

    def solve_instance(n, p, seed, time_limit):
        adj = random_graph(n, p, seed)
        rem = adj[:]  # Remaining edges as bitmasks
        cliques = []
        
        # Check if graph is empty
        if all(x == 0 for x in rem):
            return cliques

        # Compute initial spectral order
        spectral_order = compute_spectral_order(adj, n)
        spectral_rank = [0] * n
        for rank, v in enumerate(spectral_order):
            spectral_rank[v] = rank
            
        # Helper to get list of vertices from a bitmask
        def bits_to_list(mask):
            verts = []
            m = mask
            while m:
                lsb = m & -m
                verts.append(lsb.bit_length() - 1)
                m ^= lsb
            return verts

        def find_max_clique():
            """
            Find a maximum clique in the graph defined by 'rem' bitmasks.
            Uses Bron-Kerbosch with coloring and spectral ordering.
            """
            # Active vertices: those with at least one remaining edge
            active_mask = 0
            for i in range(n):
                if rem[i]:
                    active_mask |= (1 << i)

            if active_mask == 0:
                return []

            best_clique = []
            
            def color_sort(P):
                """
                Sort vertices in P by color (for pruning) and then by spectral rank.
                Returns a list of (color, vertex) sorted by color ascending.
                """
                verts = bits_to_list(P)
                # Sort by degree in P descending, then by spectral rank
                # This helps the greedy coloring produce better (lower) colors
                verts.sort(key=lambda v: (-(rem[v] & P).bit_count(), spectral_rank[v]))
                
                colors = {}
                colored = []
                for v in verts:
                    used = set()
                    # Neighbors of v in P
                    nm = rem[v] & P
                    while nm:
                        lsb = nm & -nm
                        u = lsb.bit_length() - 1
                        if u in colors:
                            used.add(colors[u])
                        nm ^= lsb
                    # Assign smallest available color
                    c = 1
                    while c in used:
                        c += 1
                    colors[v] = c
                    colored.append((c, v))
                
                # Sort by color for BK algorithm (process highest color last? 
                # Standard BK processes in reverse order of colors to prune effectively)
                colored.sort(key=lambda x: x[0])
                return colored

            def bk(R, P, X):
                nonlocal best_clique
                # Time check
                if time.time() > time_limit:
                    return
                
                if P == 0:
                    if len(R) > len(best_clique):
                        best_clique = list(R)
                    return
                
                colored = color_sort(P)
                
                # Process in reverse order (highest color first)
                # If len(R) + color <= len(best_clique), we cannot beat best_clique
                for color, v in reversed(colored):
                    if time.time() > time_limit:
                        return
                    if len(R) + color <= len(best_clique):
                        break
                    
                    R.append(v)
                    bk(R, P & rem[v], X & rem[v])
                    R.pop()
                    
                    P &= ~(1 << v)
                    X |= (1 << v)

            bk([], active_mask, 0)
            return best_clique

        # Main loop
        while time.time() < time_limit:
            # Check if edges remain
            has_edges = False
            for i in range(n):
                if rem[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            clique = find_max_clique()
            
            if not clique or len(clique) < 2:
                # If we can't find a clique of size >= 2, break to fallback
                break

            cliques.append(clique)

            # Remove edges of this clique from rem
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i + 1, len(clique)):
                    v = clique[j]
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)
            
            # Periodically update spectral order if significant edges removed?
            # For n=32/64, recomputing is cheap. Let's update every few iterations or if graph changes a lot.
            # Simple heuristic: update if > 10% edges removed since last update
            # To keep it simple and fast, we'll just use the static order from the start.
            # Dynamic spectral might be overkill for this size if static works well.

        # Fallback: cover remaining edges as 2-cliques
        # We need to iterate through remaining edges carefully to avoid duplicates
        covered = [False] * n
        for u in range(n):
            mask = rem[u]
            # Only consider neighbors v > u to avoid duplicates
            mask &= ~((1 << (u + 1)) - 1) 
            # Actually, simpler: just iterate all bits, but check if u < v
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v and rem[u] & (1 << v):
                    cliques.append([u, v])
                    # Remove this edge
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)
                m ^= lsb

        return cliques

    # Solve first instance (n=32)
    # Budget allocation: give 60% of time to first, 40% to second?
    # Or just run until time limit for each, but total must be < 140s.
    # Let's give 70s to first, 70s to second.
    
    budget1 = 70
    cliques1 = solve_instance(32, 0.6, 601, start_time + budget1)
    
    # Update deadline for second instance
    current_time = time.time()
    remaining_budget = deadline - current_time
    if remaining_budget < 10:
        # Not enough time, just return what we have for second as empty or minimal
        # But we must return valid partitions. 
        # If we don't have time, we should solve it very fast or fallback.
        # Let's give it whatever is left.
        budget2 = max(1, remaining_budget - 2)
    else:
        budget2 = 70

    cliques2 = solve_instance(64, 0.55, 602, start_time + budget1 + budget2)

    return [cliques1, cliques2]
# EVOLVE-BLOCK-END
