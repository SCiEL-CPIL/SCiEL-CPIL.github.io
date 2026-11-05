import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        # Create adjacency matrix or list
        # For speed in BK, we use bitmasks if n is small, or lists.
        # n <= 40, so bitmasks (integers) are perfect.
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique_size(adj, n):
        """
        Finds the size of the maximum clique in a graph represented by bitmasks.
        Uses Bron-Kerbosch with pivot and greedy coloring bound.
        """
        # We need to find max clique in adj.
        # Standard Tomita algorithm.
        
        # Pre-compute colors for initial bound? 
        # Or just use a simple bound: size of P.
        # For n=40, simple BK with pivot is likely fast enough.
        # Let's implement a recursive BK that returns the max size.
        
        best_size = 0
        
        # Color sort for bounding
        # We'll use a simple greedy coloring on the subgraph induced by P to get an upper bound.
        
        def color_sort(P, adj):
            # P is a bitmask
            # Returns list of (vertex, color) sorted by color
            uncolored = P
            color_classes = []
            while uncolored:
                # Pick a vertex
                # To speed up, pick vertex with lowest degree in uncolored? 
                # Or just lowest index.
                # For coloring bound, we want to color as many as possible with few colors.
                # Greedy: pick vertex, remove it and its neighbors from uncolored, assign next color.
                
                # Actually, standard Tomita uses a specific coloring.
                # Let's stick to a simpler bound first: popcount(P).
                # If that's too slow, we add coloring.
                # Given n=40, popcount bound might be sufficient for 120s.
                pass
        
        # Let's implement a robust Branch and Bound for Max Clique.
        
        def expand(R, P, X):
            nonlocal best_size
            if time.time() > deadline:
                return
            if not P and not X:
                if R.bit_count() > best_size:
                    best_size = R.bit_count()
                return
            
            # If P is empty, we have a maximal clique
            # But we want max, so we prune if |R| + |P| <= best_size
            if R.bit_count() + P.bit_count() <= best_size:
                return

            # Pivot selection
            # Choose u from P U X that maximizes |P ∩ N(u)|
            # This minimizes the branching factor.
            PX = P | X
            if PX:
                # Find vertex u in PX with max common neighbors in P
                # Iterate through bits of PX
                max_common = -1
                pivot = -1
                temp = PX
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    common = (P & adj[u]).bit_count()
                    if common > max_common:
                        max_common = common
                        pivot = u
                    temp ^= lsb
            else:
                pivot = -1

            # Branch on P \ N(pivot)
            if pivot != -1:
                candidates = P & ~(adj[pivot])
            else:
                candidates = P

            temp = candidates
            while temp:
                if time.time() > deadline:
                    return
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                # Recurse
                expand(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb

        # Initial call
        # P = all vertices, X = empty
        expand(0, (1 << n) - 1, 0)
        
        return best_size

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback to a simple heuristic if we run out of time
            # Just return all vertices (valid but large)
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in range(u+1, n) if (adj[u] >> v) & 1 for x in (u,v)}
            results.append(sorted(touched))
            continue

        adj = random_graph(n, p, seed)
        
        # We need Max Independent Set in G.
        # This is Max Clique in complement G_bar.
        # Build complement adj
        full_mask = (1 << n) - 1
        adj_bar = [0] * n
        for u in range(n):
            # Neighbors in G_bar are non-neighbors in G (excluding self)
            # adj[u] has 0 for self.
            # We want bits that are 0 in adj[u] and 1 in full_mask, except bit u.
            # comp = full_mask ^ adj[u]
            # comp has 1 at bit u. We must clear it.
            adj_bar[u] = (full_mask ^ adj[u]) & ~(1 << u)
        
        # Find Max Clique in adj_bar
        # Note: My expand function above is for Max Clique.
        # It uses adj (which should be the graph where we are finding cliques).
        # So we pass adj_bar.
        
        # Optimization: Check if graph is complete or empty quickly?
        # Not necessary for n=40.
        
        clique_size = max_clique_size(adj_bar, n)
        
        # The transversal is V \ MaxClique.
        # We need to retrieve the actual vertices of the max clique to exclude them.
        # My current max_clique_size only returns size.
        # I need the set of vertices.
        
        # Let's modify max_clique_size to return the set.
        
    # Redo with vertex retrieval
    
    def max_clique_vertices(adj, n):
        best_vertices = 0
        best_size = 0
        
        def expand(R, P, X):
            nonlocal best_vertices, best_size
            if time.time() > deadline:
                return
            if not P and not X:
                if R.bit_count() > best_size:
                    best_size = R.bit_count()
                    best_vertices = R
                return
            
            if R.bit_count() + P.bit_count() <= best_size:
                return

            PX = P | X
            if PX:
                max_common = -1
                pivot = -1
                temp = PX
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    common = (P & adj[u]).bit_count()
                    if common > max_common:
                        max_common = common
                        pivot = u
                    temp ^= lsb
            else:
                pivot = -1

            if pivot != -1:
                candidates = P & ~(adj[pivot])
            else:
                candidates = P

            temp = candidates
            while temp:
                if time.time() > deadline:
                    return
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                expand(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb

        expand(0, (1 << n) - 1, 0)
        
        # Convert bitmask to list of vertices
        vertices = []
        mask = best_vertices
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            vertices.append(v)
            mask ^= lsb
        return vertices

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in range(u+1, n) if (adj[u] >> v) & 1 for x in (u,v)}
            results.append(sorted(touched))
            continue

        adj = random_graph(n, p, seed)
        full_mask = (1 << n) - 1
        adj_bar = [0] * n
        for u in range(n):
            adj_bar[u] = (full_mask ^ adj[u]) & ~(1 << u)
        
        clique_vertices = max_clique_vertices(adj_bar, n)
        
        # Transversal is all vertices NOT in clique
        clique_set = set(clique_vertices)
        transversal = [v for v in range(n) if v not in clique_set]
        
        results.append(sorted(transversal))

    return results
