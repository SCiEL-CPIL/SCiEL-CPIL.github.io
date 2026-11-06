import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin within 60s limit
    
    N = 256
    # Build conflict graph
    # adj[i] is a set of nodes j such that (i+1, j+1) is a conflict
    # We use 0-indexed internally for the solver, but values are 1..256
    adj = [set() for _ in range(N)]
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    # Convert to lists for faster access in some cases, or keep sets
    # Tomita's algorithm benefits from bitsets or efficient neighbor iteration.
    # With N=256, we can use Python integers as bitmasks for adjacency.
    # This allows very fast set operations.
    
    adj_mask = [0] * N
    for i in range(N):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
        
    # Precompute greedy coloring for the whole graph to get an initial upper bound?
    # Or just use it inside the B&B.
    
    # Greedy Coloring (Tomita's style)
    # We need to color a set of vertices (given as a bitmask) such that adjacent vertices have different colors.
    # The number of colors is an upper bound on the clique size, which is the independence number of the complement.
    # Wait, MIS on G is Clique on G_complement.
    # Conflict graph G: edge if conflict. We want Independent Set in G.
    # Independent Set in G is Clique in G_complement.
    # Let's work directly on G.
    # Upper bound for Independent Set in a subgraph H is the clique number of H_complement.
    # Alternatively, we can use the bound: size of independent set <= number of colors in a greedy coloring of the subgraph?
    # No, greedy coloring gives an upper bound on the *clique* size of the graph being colored.
    # If we color G, the number of colors is an upper bound on the Clique Size of G.
    # We want Independent Set in G.
    # Independent Set in G is Clique in G_comp.
    # So we should color G_comp? Or use a different bound?
    
    # Actually, there is a standard bound for MIS:
    # If we greedily color the subgraph, the number of colors is an upper bound on the *clique number* of that subgraph.
    # This doesn't directly bound the independent set size.
    
    # Let's use the property:
    # Max Independent Set Size <= N - Min Vertex Cover Size.
    # Also, Max Independent Set Size is related to Coloring.
    # In fact, if we color the graph G, let the colors be 1..k.
    # This doesn't bound MIS directly.
    
    # Let's stick to the standard Tomita algorithm for Maximum Clique, and run it on the COMPLEMENT graph.
    # Let G_conflict be the graph where edges represent conflicts.
    # We want MIS(G_conflict).
    # MIS(G_conflict) = MaxClique(G_complement).
    
    # Let's build G_complement adjacency masks.
    # Total nodes N.
    # All_mask = (1 << N) - 1
    # adj_comp_mask[i] = All_mask ^ adj_mask[i] ^ (1 << i)  # Remove self and remove conflict neighbors
    
    All_mask = (1 << N) - 1
    adj_comp_mask = [0] * N
    for i in range(N):
        # Neighbors in complement are those NOT in adj_mask[i] and NOT i
        # adj_mask[i] contains conflict neighbors.
        # So comp neighbors = All_mask & ~(adj_mask[i] | (1 << i))
        adj_comp_mask[i] = All_mask & ~(adj_mask[i] | (1 << i))
        
    # Now we solve MaxClique on G_comp using Tomita's algorithm.
    
    def greedy_color(vertices, adj_mask):
        """
        Greedy coloring of the subgraph induced by 'vertices' (bitmask).
        Returns a list of colors assigned to each vertex in 'vertices', 
        and the maximum color used.
        The vertices are ordered by their index in the bitmask? 
        Tomita's algorithm usually orders vertices by degree or just processes them.
        Standard Tomita:
        1. Order vertices.
        2. Greedily assign colors.
        3. The number of colors is the upper bound.
        
        For efficiency, we can return the count of colors.
        """
        colors = [0] * N
        color_count = 0
        # We process vertices in the order they appear in the bitmask, 
        # but Tomita's algorithm specifically uses an ordering to improve the bound.
        # A simple ordering is by degree in the subgraph, or just natural order.
        # Let's try natural order first, or sort by degree.
        
        # To make it fast, we iterate through bits.
        # However, the "bound" quality depends on the ordering.
        # Tomita uses a specific ordering based on previous steps.
        # For simplicity and speed in Python, we might just use natural order or a static ordering.
        
        # Let's use a simple greedy coloring:
        # Iterate through vertices in 'vertices'.
        # For each vertex, assign the smallest available color.
        
        # To optimize, we can group vertices by the color they get.
        # But we just need the max color count.
        
        # Efficient approach:
        # Maintain a list of color classes (bitmasks).
        # For each vertex v:
        #   find first color class c such that v is not adjacent to any vertex in c.
        #   if found, add v to c.
        #   else, create new color class.
        
        color_classes = []
        # Order vertices: we can sort by degree in the subgraph to improve coloring quality?
        # Or just iterate.
        
        # Let's iterate through vertices in the order of their index.
        # This is O(N^2) per call. With N=256, this is fine.
        
        # Extract vertices
        verts = []
        v = vertices
        while v:
            lsb = v & -v
            idx = lsb.bit_length() - 1
            verts.append(idx)
            v ^= lsb
            
        # Optional: Sort by degree in the subgraph (descending) to get a tighter bound?
        # Degree of v in subgraph = popcount(adj_mask[v] & vertices)
        # This might be expensive. Let's try natural order first.
        
        for v in verts:
            placed = False
            for i, c_class in enumerate(color_classes):
                # Check if v is non-adjacent to all in c_class
                # adj_mask[v] & c_class == 0
                if (adj_mask[v] & c_class) == 0:
                    color_classes[i] |= (1 << v)
                    placed = True
                    break
            if not placed:
                color_classes.append(1 << v)
                
        return len(color_classes)

    # Tomita's Branch and Bound
    # State: current clique (list of vertices), candidates (bitmask)
    # We maximize the clique size.
    
    best_clique = []
    
    def expand(clique, candidates):
        nonlocal best_clique
        
        # If no candidates, we have a maximal clique
        if not candidates:
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            return
        
        # Upper bound: len(clique) + greedy_color(candidates, adj_comp_mask)
        # Note: greedy_color returns the number of colors, which is an upper bound on the clique size in the subgraph.
        bound = len(clique) + greedy_color(candidates, adj_comp_mask)
        
        if bound <= len(best_clique):
            return
            
        # Select a vertex to branch on
        # Tomita's choice: pick a vertex from the highest color class? 
        # Or just pick the first one?
        # To prune effectively, we should pick vertices from the last color class added in greedy coloring, 
        # or simply iterate through candidates.
        
        # Let's iterate through candidates. 
        # To improve, we can order them.
        
        # Extract candidates
        cand_list = []
        c = candidates
        while c:
            lsb = c & -c
            idx = lsb.bit_length() - 1
            cand_list.append(idx)
            c ^= lsb
            
        # We can sort cand_list to improve search? 
        # Often, picking high-degree vertices first helps.
        
        for v in cand_list:
            if time.time() > deadline:
                return
                
            # New candidates are those in 'candidates' that are neighbors of v
            new_candidates = candidates & adj_comp_mask[v]
            
            # Recurse
            expand(clique + [v], new_candidates)
            
            # Remove v from candidates
            candidates &= ~(1 << v)
            
            # Update bound
            # The bound might decrease as we remove vertices
            # We can recompute bound or just check if bound is still > best
            # For efficiency, we can recompute bound only if necessary or periodically.
            # Given N=256, recomputing is cheap enough if we do it often.
            # But greedy_color is O(N^2). 
            # Let's just check if the remaining candidates could possibly beat best.
            # A simple bound: len(clique) + popcount(candidates)
            if len(clique) + bin(candidates).count("1") <= len(best_clique):
                break

    # Initial candidates: all nodes
    initial_candidates = All_mask
    expand([], initial_candidates)
    
    # Convert best_clique (indices) to values (1..256)
    result = [v + 1 for v in best_clique]
    
    # Fallback: if the solver didn't find anything or timed out badly, 
    # use the previous heuristic as a safety net?
    # The solver should finish. But if it returns empty (shouldn't happen), 
    # or if we want to be safe:
    
    if not result:
        # Fallback to simple greedy
        S = set(range(1, N + 1, 2))
        evens = list(range(2, N + 1, 2))
        def can_add(x, S):
            for y in S:
                if (x * y) % (x + y) == 0:
                    return False
            return True
        for e in evens:
            if can_add(e, S):
                S.add(e)
        result = list(S)
        
    return result
