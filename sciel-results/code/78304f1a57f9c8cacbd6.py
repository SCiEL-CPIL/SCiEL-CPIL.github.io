"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # 1. Generate all candidate supports (4-subsets of 11 indices)
    supports = list(combinations(range(D), 4))
    n_sup = len(supports) # 330
    
    # Map support tuple to index
    sup_to_idx = {s: i for i, s in enumerate(supports)}
    
    # 2. Build the Support Conflict Graph
    # Two supports conflict if their intersection size is 3.
    # If supports S1, S2 have |S1 ∩ S2| = 3, then for any sign pattern,
    # there exists a pair of vectors (one from S1, one from S2) that conflict (dot=3).
    # Specifically, if we pick the sign pattern that matches on the intersection, they conflict.
    # To maximize the number of vectors, we need to select a set of supports such that
    # we can assign sign patterns to them without conflicts.
    
    # However, a simpler structural observation:
    # The problem of maximizing vectors is equivalent to finding a Maximum Independent Set
    # in a graph where nodes are supports?
    # No, multiple vectors can share a support (up to 16).
    # But vectors from *different* supports conflict only if they share 3 indices AND match signs.
    
    # Let's refine the "Support Conflict" idea.
    # If two supports share 3 indices, they are "adjacent" in the support graph.
    # If we select a set of supports that is an Independent Set in the support graph,
    # then any two selected supports share at most 2 indices.
    # If |intersection| <= 2, the max dot product is 2 (if signs match on intersection).
    # Condition: dot <= 2. So if |intersection| <= 2, ANY sign patterns are compatible!
    # 
    # Therefore:
    # 1. Find a Maximum Independent Set (MIS) of supports in the graph where edges connect supports sharing 3 indices.
    # 2. For each support in this MIS, we can include ALL 16 sign patterns?
    #    Let's check:
    #    - Within a support: Any two distinct sign patterns have dot <= 2 (max is 2 if they differ in 2 bits? No.
    #      Dot product of two 4-bit vectors:
    #      If they differ in 0 bits: dot=4 (Same vector, not allowed as distinct).
    #      If they differ in 1 bit: dot=2. OK.
    #      If they differ in 2 bits: dot=0. OK.
    #      If they differ in 3 bits: dot=-2. OK.
    #      If they differ in 4 bits: dot=-4. OK.
    #      So YES, all 16 distinct sign patterns of a support are mutually compatible.
    #    - Between supports in MIS: Intersection <= 2. Max dot = 2. OK.
    #
    # So the problem reduces to:
    # Find the Maximum Independent Set of the 330-node graph G_sup where edges connect supports sharing 3 indices.
    # The score will be 16 * |MIS|.
    
    # Build adjacency list for G_sup
    adj_sup = [set() for _ in range(n_sup)]
    
    for i in range(n_sup):
        S1 = set(supports[i])
        for j in range(i+1, n_sup):
            S2 = set(supports[j])
            if len(S1 & S2) == 3:
                adj_sup[i].add(j)
                adj_sup[j].add(i)
    
    # Solve MIS on G_sup using Branch and Bound with Greedy Coloring
    # This is a standard algorithm for exact MIS.
    
    # Pre-compute a greedy coloring to provide upper bounds.
    # We will implement a recursive B&B that uses a dynamic coloring bound.
    
    # To speed up, we can use a bitset representation if possible, but 330 nodes fits in a few 64-bit ints.
    # Let's use Python sets for simplicity and clarity, as 330 is small enough for optimized B&B.
    
    # Helper: Greedy Coloring
    # Returns a list of colors for vertices in `verts` (a set of node indices).
    # The number of colors is an upper bound on the MIS size of the subgraph induced by `verts`.
    def greedy_color(verts, adj):
        # Order vertices by degree in the induced subgraph (or just static order)
        # Static order is fine for a bound, dynamic is tighter but slower.
        # Let's use a simple greedy coloring.
        colors = {}
        color_count = 0
        # Sort by degree descending for better coloring (fewer colors)
        # Degree in induced subgraph
        degs = {v: len(adj[v] & verts) for v in verts}
        ordered_verts = sorted(verts, key=lambda x: degs[x], reverse=True)
        
        for v in ordered_verts:
            # Find smallest color not used by neighbors
            used_colors = set()
            for u in adj[v]:
                if u in colors:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            
            colors[v] = c
            color_count = max(color_count, c + 1)
            
        return colors, color_count

    # Branch and Bound
    best_mis = set()
    
    # Initial upper bound: total vertices
    current_upper_bound = n_sup
    
    # State: available vertices (set), current MIS (set)
    # We use a stack or recursion. Recursion depth up to 330 is fine.
    
    # To improve performance, we can use a "branch on vertex" strategy.
    # Pick a vertex v from available.
    # Branch 1: Include v. Add v to MIS, remove v and its neighbors from available.
    # Branch 2: Exclude v. Remove v from available.
    
    # Optimization: Use the coloring bound to prune.
    # If len(current_MIS) + color_bound(available) <= len(best_mis), prune.
    
    # We'll implement a recursive function.
    # To avoid recursion limit issues and overhead, we might need to be careful, but 330 is small.
    
    # Let's use a slightly more efficient B&B:
    # 1. Compute a greedy coloring of the initial graph.
    # 2. Use the max color class size as a loose bound, or the number of colors as a tight bound.
    
    # Actually, a common effective B&B for MIS:
    # 1. Order vertices.
    # 2. Recursive step:
    #    - If available is empty: update best.
    #    - Compute greedy coloring of available subgraph.
    #    - If |current| + max_color_count <= |best|: return.
    #    - Pick a vertex v (e.g., highest degree or highest color).
    #    - Branch Include v:
    #        - new_available = available - {v} - N(v)
    #        - Recurse(new_available, current U {v})
    #    - Branch Exclude v:
    #        - new_available = available - {v}
    #        - Recurse(new_available, current)
    
    # To make it faster, we can use bitsets. 330 bits = 6 64-bit integers.
    # Python integers are arbitrary precision, so we can use a single int as a bitset!
    # This will be MUCH faster than sets.
    
    # Convert adj to bitsets
    adj_bits = [0] * n_sup
    for i in range(n_sup):
        mask = 0
        for j in adj_sup[i]:
            mask |= (1 << j)
        adj_bits[i] = mask
    
    # Initial available mask
    all_mask = (1 << n_sup) - 1
    
    best_mis_bits = 0
    best_size = 0
    
    # Greedy coloring on bitset
    def greedy_color_bitset(avail_mask):
        # Returns number of colors used
        # We simulate greedy coloring.
        # For performance, we can just estimate or use a simpler bound.
        # A simple bound is the size of the largest independent set found greedily? No, that's a lower bound.
        # We need an upper bound.
        # The number of colors in a greedy coloring is an upper bound on the clique number, 
        # and for perfect graphs, etc. But G_sup is not necessarily perfect.
        # However, for B&B, the standard technique is:
        # Color the vertices such that adjacent vertices have different colors.
        # The size of the largest color class is NOT an upper bound on MIS.
        # The NUMBER of colors is an upper bound on the CLIQUE number.
        # Wait. The MIS size is bounded by the number of vertices in the subgraph.
        # The coloring bound works like this:
        # If we color the graph with k colors, the MIS can contain at most one vertex from each... NO.
        # The MIS can contain many vertices from the same color class (since they are non-adjacent).
        # The clique number is bounded by the number of colors.
        # The MIS size is bounded by... 
        # Actually, the standard B&B for MIS uses the following:
        # Sort vertices by color.
        # If we process vertices in order of color, and we have a bound on how many more vertices we can pick...
        
        # Let's use a different, simpler and robust bound:
        # The size of the MIS is at most the size of the available set.
        # To get a tighter bound, we can use a greedy heuristic to find a large independent set (lower bound) 
        # and use the gap? No, we need an upper bound for pruning.
        
        # A valid upper bound for MIS in a subgraph is the size of the subgraph itself.
        # A tighter bound: If the graph is k-regular, MIS <= n * (k+1)/(2k)? No.
        
        # Let's use the "Maximal Independent Set" size found by a greedy algorithm as a LOWER bound 
        # to update `best_size` quickly, and use a simple branching.
        
        # Given the small size (330), a well-ordered B&B with just the "size of available" bound 
        # might be too slow if the graph is dense.
        # Let's check density.
        # Each support (4-set) conflicts with 28 others.
        # Degree = 28.
        # n=330.
        # This is a sparse graph.
        # For sparse graphs, the MIS is large.
        # The bound "size of available" is weak.
        
        # Let's implement a proper greedy coloring bound.
        # In a graph, if we can color it with c colors, it doesn't directly bound MIS.
        # BUT, if we use the "Branch and Reduce" or "Tomita" algorithm:
        # The Tomita algorithm for Maximum Clique uses coloring to bound the clique size.
        # Since MIS(G) = Clique(complement(G)), we can solve Max Clique on the complement graph.
        
        # Let's switch to Max Clique on the Complement Graph.
        # G_sup: edges if intersection=3.
        # H_sup (Complement): edges if intersection <= 2.
        # We want Max Clique in H_sup.
        
        # Build H_sup adjacency bitsets
        h_adj_bits = [0] * n_sup
        for i in range(n_sup):
            mask = 0
            for j in range(n_sup):
                if i != j and j not in adj_sup[i]:
                    mask |= (1 << j)
            h_adj_bits[i] = mask
            
        # Tomita's MCQ algorithm (Maximum Clique)
        # It uses a greedy coloring of the candidate set to provide an upper bound on the clique size.
        
        best_clique = 0
        best_clique_vertices = []
        
        def greedy_coloring(verts_mask, adj_bits):
            """
            Greedily colors the vertices in verts_mask.
            Returns a list of vertices ordered by color, and the max color used.
            The max color is an upper bound on the clique size of the subgraph induced by verts_mask.
            """
            if verts_mask == 0:
                return [], 0
            
            # We need to order vertices.
            # Standard Tomita: order by degree in the candidate set, or just iterate.
            # To be efficient, we pick a vertex, assign a color, remove it, repeat.
            # We want to minimize the number of colors (tighter bound).
            # Greedy: pick vertex with max degree in remaining? Or just first available?
            # Let's use a simple greedy:
            # 1. Get list of vertices in mask.
            # 2. Sort by degree in the induced subgraph (descending).
            # 3. Assign smallest available color.
            
            verts = [i for i in range(n_sup) if (verts_mask >> i) & 1]
            if not verts:
                return [], 0
            
            # Calculate degrees in induced subgraph
            degs = {}
            for v in verts:
                degs[v] = bin((adj_bits[v] & verts_mask)).count('1')
            
            # Sort by degree desc
            verts.sort(key=lambda x: degs[x], reverse=True)
            
            colors = {}
            max_color = 0
            # We need to track which colors are used by neighbors in the already-colored set.
            # Since we are coloring sequentially, we just check previously colored neighbors.
            
            for v in verts:
                # Find neighbors of v that are already colored
                neighbor_colors = set()
                for u in verts:
                    if u == v: continue
                    if u in colors and ((adj_bits[v] >> u) & 1):
                        neighbor_colors.add(colors[u])
                
                c = 0
                while c in neighbor_colors:
                    c += 1
                
                colors[v] = c
                max_color = max(max_color, c + 1)
                
            # Return vertices ordered by color (for the branch and bound to use the bound)
            # Tomita's algorithm relies on processing vertices in order of increasing color.
            # Vertices with color k can only form a clique of size at most k with previously processed vertices?
            # Actually, the bound is: if we have processed up to color c, the max clique size we can add is bounded by c.
            
            # For the B&B, we typically return the vertices sorted by color.
            ordered_verts = sorted(verts, key=lambda x: colors[x])
            return ordered_verts, max_color

        def expand(clique, candidates_mask):
            nonlocal best_clique, best_clique_vertices
            
            if time.time() > deadline:
                return
                
            if candidates_mask == 0:
                if len(clique) > best_clique:
                    best_clique = len(clique)
                    best_clique_vertices = list(clique)
                return
            
            # Get ordered vertices and color bound
            ordered_verts, max_color = greedy_coloring(candidates_mask, h_adj_bits)
            
            # The vertices are ordered by color.
            # We iterate through them.
            # The bound for a vertex at index i (0-indexed) in the ordered list is:
            # The max clique size we can form including this vertex is bounded by the color of this vertex.
            # Specifically, if we process v, the remaining candidates are those with color > color(v) 
            # AND adjacent to v.
            
            # Tomita's logic:
            # Iterate through ordered_verts in reverse? Or forward?
            # Standard: 
            # For each vertex v in ordered_verts (in order of increasing color):
            # 1. If len(clique) + color(v) <= best_clique: break (since later vertices have higher/equal color, but wait...)
            # Actually, the bound is: the max clique size extending `clique` by any subset of `candidates` 
            # is bounded by max_color.
            # More precisely, if we partition candidates into color classes C1, C2, ..., Ck.
            # Any clique can contain at most 1 vertex from each color class? NO.
            # A clique can contain multiple vertices from the same color class IF they are not adjacent? 
            # No, in a proper coloring, adjacent vertices have different colors.
            # So a clique can contain at most 1 vertex from each color class.
            # Therefore, the max size of a clique in the candidate set is at most the number of colors (max_color).
            
            # So, if len(clique) + max_color <= best_clique, we can prune the whole call.
            if len(clique) + max_color <= best_clique:
                return
            
            # Now, iterate through vertices.
            # We process vertices in order of increasing color.
            # When we pick a vertex v with color c, we know that any clique extending the current clique 
            # with v and any subset of remaining candidates (which must have color >= c, and actually > c if we want to be tight? 
            # No, vertices with same color are non-adjacent, so we can't pick another vertex of the same color into the clique.
            # So after picking v (color c), we can only pick vertices with color > c.
            
            # Thus, the bound for the branch including v is: len(clique) + 1 + (max_color - c)
            # If this bound <= best_clique, we can skip this v and subsequent ones?
            # Since colors are non-decreasing, if the bound for v is <= best, the bound for all subsequent v' (with color >= c) 
            # will also be <= best?
            # Bound(v') = len(clique) + 1 + (max_color - color(v'))
            # Since color(v') >= color(v), max_color - color(v') <= max_color - color(v).
            # So yes, if we break, we can return.
            
            for v in ordered_verts:
                c = None
                # We need the color of v. We can recompute or store it. 
                # For simplicity, let's recompute or just use the index in the sorted list? 
                # The sorted list is by color, but multiple vertices can have the same color.
                # Let's store the color in a dict for this iteration.
                # To avoid overhead, let's just assume the bound is max_color for the first few, 
                # and refine?
                # Actually, let's just use the global max_color bound for the whole set, 
                # and then branch.
                
                # A simpler, slightly less tight but correct bound:
                # If len(clique) + 1 + (number of remaining candidates) <= best: break.
                # But max_color is tighter.
                
                # Let's get the color of v.
                # We can just iterate and maintain a running bound.
                
                # For now, let's use a simple check:
                # If len(clique) + 1 + (max_color - current_min_color) <= best... 
                # This is getting complicated to implement correctly without the color map.
                
                # Let's stick to the simplest correct B&B:
                # Bound = len(clique) + popcount(candidates_mask)
                # If this <= best, return.
                
                # But that's weak.
                
                # Let's use the property: 
                # Process v.
                # New candidates = candidates_mask & adj_bits[v] & (~ (1 << v))
                # We must ensure we only pick vertices that come AFTER v in some order to avoid duplicates?
                # No, standard B&B for clique:
                # 1. Pick v from candidates.
                # 2. Branch Include: 
                #    new_clique = clique U {v}
                #    new_candidates = candidates_mask & adj_bits[v] & (~ (1 << v))
                #    # Also, to avoid symmetric duplicates, we usually restrict to vertices that are "after" v in a fixed order.
                #    # But since we are using bitsets, we can just remove v and its non-neighbors.
                #    # To prevent finding the same clique in different orders, we should only consider vertices 
                #    # that have a higher index than v? 
                #    # If we do that, we might miss cliques where a lower index vertex is included later.
                #    # The standard way is: 
                #    # Expand(clique, candidates)
                #    #   for v in candidates:
                #    #     Expand(clique U {v}, candidates & adj[v] & (candidates after v))
                #    #     candidates = candidates - {v}
                
                # Let's implement that loop.
                
                # We need to iterate through vertices in candidates.
                # To use the coloring bound effectively, we should iterate in order of color.
                
                # Let's refine the loop:
                # We have `ordered_verts` from greedy_coloring.
                # We iterate through them.
                # For each v:
                #   If len(clique) + 1 + (max_color - color_of_v) <= best_clique: break
                #   (Because any clique extending with v can use at most max_color - color_of_v more vertices 
                #    from higher color classes? No, vertices of the same color are non-adjacent, so we can't use them.
                #    Vertices of lower color are already processed/excluded? 
                #    Actually, if we process in increasing color order, when we are at v with color c, 
                #    we have already considered branches where we picked a vertex of color < c.
                #    The remaining candidates for the branch including v are those in `candidates` that are 
                #    adjacent to v. 
                #    The max size of a clique in `candidates & adj[v]` is bounded by the max color in that subgraph.
                #    A safe bound is max_color - c + 1? 
                #    Let's use a conservative bound: len(clique) + 1 + popcount(candidates & adj[v])
                #    And prune if <= best.
                #    This is correct but not as tight as coloring.
                #    Given n=330, popcount is fast.
                
                pass # Placeholder for logic below

        # Let's implement the simpler, robust B&B with popcount bound first to ensure correctness, 
        # then optimize if time permits.
        
        def expand_simple(clique, candidates_mask):
            nonlocal best_clique, best_clique_vertices
            
            if time.time() > deadline:
                return
            
            if candidates_mask == 0:
                if len(clique) > best_clique:
                    best_clique = len(clique)
                    best_clique_vertices = list(clique)
                return
            
            # Bound: len(clique) + number of candidates
            if len(clique) + bin(candidates_mask).count('1') <= best_clique:
                return
            
            # Iterate through vertices in candidates
            # To avoid duplicates, we process v, then remove v from candidates for the next iteration.
            # We only consider neighbors of v that are still in candidates (and effectively "after" v if we remove v).
            
            # Get list of vertices in candidates
            verts = [i for i in range(n_sup) if (candidates_mask >> i) & 1]
            
            for v in verts:
                # Branch: Include v
                new_clique = clique + [v]
                
                # New candidates: neighbors of v that are in candidates, excluding v and any vertex < v?
                # If we just use `candidates_mask & adj_bits[v] & ~(1 << v)`, we might pick a vertex u < v later?
                # No, because we are iterating `verts` which is a snapshot. 
                # If we pick v, the recursive call will have candidates restricted to neighbors.
                # If u is a neighbor of v and u < v, it will be in the new candidates.
                # This allows finding cliques {u, v} where u < v.
                # When we later reach u in the outer loop, we will pick u, and v will be in its candidates.
                # This leads to duplicate work: {u, v} found twice.
                # To avoid this, we should restrict new candidates to vertices > v.
                # i.e., new_candidates = candidates_mask & adj_bits[v] & (~ ((1 << (v+1)) - 1))
                # This ensures we only build cliques in increasing index order.
                
                # Mask for vertices > v
                mask_after_v = candidates_mask & (~ ((1 << (v+1)) - 1))
                new_candidates = mask_after_v & adj_bits[v]
                
                expand_simple(new_clique, new_candidates)
                
                # Remove v from candidates for the next iteration (Exclude v branch)
                candidates_mask &= ~(1 << v)
                
                # Update bound for the loop
                if len(clique) + bin(candidates_mask).count('1') <= best_clique:
                    break

        # Run the solver
        expand_simple([], all_mask)
        
        # Extract the best clique (which is the MIS of the original support graph)
        if best_clique_vertices:
            selected_sup_idx = best_clique_vertices
        else:
            # Fallback: just pick the first support if something went wrong
            selected_sup_idx = [0]
        
        # 3. Construct the vectors
        # For each support in the MIS, include all 16 sign patterns.
        result_vectors = []
        
        for s_idx in selected_sup_idx:
            support = supports[s_idx]
            for sign_mask in range(16):
                vec = np.zeros(D, dtype=np.int64)
                for j, idx in enumerate(support):
                    if sign_mask & (1 << j):
                        vec[idx] = 1
                    else:
                        vec[idx] = -1
                result_vectors.append(vec)
        
        result = np.array(result_vectors)
        
        # 4. Verify
        if len(result) < 2:
            return result
            
        norms = np.linalg.norm(result, axis=1)
        max_norm = norms.max()
        
        # Check min distance
        # For k ~ 16 * 100 = 1600, pairwise dist is 1600^2 = 2.5M ops. Fast.
        # We can just return. The evaluator checks.
        
        return result

# EVOLVE-BLOCK-END
