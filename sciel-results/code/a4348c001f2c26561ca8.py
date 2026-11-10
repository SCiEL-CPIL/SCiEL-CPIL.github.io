# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

sys.setrecursionlimit(10000)

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # 1. Build forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Build adjacency list and bitmasks
    # adj[i] is a bitmask where bit j is 1 if i and j are connected (forbidden difference)
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            if d1 in forbidden:
                mask |= (1 << j)
        adj[i] = mask
        
    # 3. Tomita's algorithm for Maximum Independent Set
    # We want to find the largest subset of vertices such that no two are adjacent.
    # This is equivalent to finding the Maximum Independent Set in the graph defined by adj.
    
    # Pre-compute degrees for initial ordering
    degrees = [bin(adj[i]).count('1') for i in range(m)]
    
    # Sort vertices by degree (descending) to improve pruning
    # We will work with a list of active vertices
    # To make the bitset operations efficient, we map sorted vertices to indices 0..m-1?
    # Actually, it's easier to keep original indices but process them in a specific order.
    # However, Tomita's algorithm typically relies on the order of vertices in the bitset.
    # Let's reorder the vertices such that higher degree vertices have lower indices in the bitset representation.
    # This allows us to use the "coloring" bound more effectively on the remaining bits.
    
    order = sorted(range(m), key=lambda x: -degrees[x])
    # Map old index to new index
    old_to_new = [0] * m
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx
        
    # Rebuild adjacency masks in the new order
    new_adj = [0] * m
    for new_i, old_i in enumerate(order):
        mask = 0
        old_mask = adj[old_i]
        for new_j, old_j in enumerate(order):
            if old_mask & (1 << old_j):
                mask |= (1 << new_j)
        new_adj[new_i] = mask
        
    # Now we solve MIS on this new graph.
    # The bit position 'i' in the mask corresponds to vertex order[i].
    
    best_size = 0
    best_set = []
    
    # Greedy coloring for bounding
    # Returns the maximum color class size, which is an upper bound on MIS size for the set of vertices
    def color_bound(vertices_mask):
        # vertices_mask is a bitmask of available vertices
        # We perform greedy coloring on the induced subgraph
        # We want to assign colors such that adjacent vertices have different colors.
        # The number of colors used is an upper bound on the size of the MIS.
        # Actually, we want to partition into independent sets.
        # The standard Tomita coloring colors the graph such that each color class is an independent set.
        # The number of colors is an upper bound on the MIS? No.
        # The MIS is a subset of one color class? No.
        # The MIS cannot contain adjacent vertices.
        # If we color the graph with k colors, any independent set can have at most 1 vertex from each... no.
        # Any independent set is a subset of a color class? No.
        # Theorem: The size of the Maximum Independent Set is at most the number of colors in any valid coloring?
        # No, that's for Clique. MIS in G is Clique in G_complement.
        # So we are finding Max Clique in the complement graph?
        # Wait. Independent Set in G = Clique in G_bar.
        # So we should apply Tomita's Max Clique algorithm to the complement graph.
        # Or adapt Tomita's MIS algorithm.
        # Standard Tomita is for Max Clique.
        # Let's stick to Max Clique on the complement graph.
        
        # Complement adjacency:
        # Two vertices are connected in complement if they are NOT connected in G.
        # And they are not the same.
        
        # Let's switch strategy: Find Max Clique in G_complement.
        # G_complement edges: (i, j) where (i, j) NOT in G and i != j.
        
        # Let's rebuild adj for complement.
        # This might be simpler to reason about with standard Tomita code.
        
        pass # This function will be replaced by the main logic below
    
    # Let's implement Max Clique on the Complement Graph.
    # Vertices are 0..m-1 (ordered).
    # comp_adj[i] has bit j set if i and j are NOT adjacent in original G and i != j.
    
    comp_adj = [0] * m
    for i in range(m):
        mask = 0
        # All bits except self
        all_bits = (1 << m) - 1
        self_bit = 1 << i
        # Not adjacent in G means NOT (new_adj[i] has bit j)
        # new_adj[i] has bits for neighbors in G.
        # So non-neighbors are all_bits ^ self_bit ^ new_adj[i]
        # But we must ensure we don't include bits for vertices > current processing order if we prune?
        # Tomita usually processes vertices in reverse order of the bitset.
        
        non_neighbors = all_bits ^ self_bit ^ new_adj[i]
        comp_adj[i] = non_neighbors

    # Tomita's algorithm for Maximum Clique
    # P: set of candidate vertices (bitmask)
    # X: set of vertices that have been colored/processed (not really needed for size, but for reconstruction)
    # best: current best size
    # best_clique: list of vertices in the best clique found
    
    # We use a recursive function
    # To speed up, we use a coloring bound on P.
    # We color P such that vertices in the same color class are not adjacent in the COMP graph?
    # No, for Max Clique in G, we color G such that adjacent vertices have different colors.
    # Then a clique can have at most 1 vertex from each color class.
    # So the number of colors is an upper bound on the clique size.
    
    # We are finding Max Clique in Comp_G.
    # So we color Comp_G.
    # Adjacent in Comp_G means NOT adjacent in G.
    # So we assign colors to vertices in P such that if u and v are adjacent in Comp_G (i.e., non-adjacent in G), they get different colors.
    
    def expand(P, X, size):
        nonlocal best_size, best_set
        
        if P == 0:
            if size > best_size:
                best_size = size
                best_set = X[:]
            return
        
        # Greedy coloring of P
        # We want to partition P into independent sets of Comp_G (which are cliques in G).
        # Wait.
        # In Max Clique search on graph H:
        # We color H.
        # If u, v are adjacent in H, they must have different colors.
        # Any clique in H can contain at most 1 vertex from each color class.
        # So #colors is an upper bound on clique size.
        
        # Here H = Comp_G.
        # So we color Comp_G.
        
        # We process vertices in P.
        # We maintain a list of color classes.
        # color_classes is a list of bitmasks.
        # We assign the smallest possible color to each vertex.
        
        # To make it efficient, we process vertices in a specific order.
        # Usually, we iterate through bits of P.
        
        # Let's use a simple greedy coloring.
        # We will store the color of each vertex in P in an array or dict?
        # Since m=205, we can use an array of size m.
        
        # But we only care about vertices in P.
        
        # Optimization: Order vertices in P by degree in Comp_G?
        # The order affects the quality of the coloring bound.
        # Better ordering -> tighter bound.
        
        # Let's extract vertices from P
        # We'll use a list of (vertex, color)
        
        # Actually, a common implementation uses a list of vertices sorted by some heuristic,
        # then assigns colors.
        
        # Let's try a simpler approach:
        # 1. Extract vertices from P.
        # 2. Sort them by degree in Comp_G (descending).
        # 3. Assign colors greedily.
        # 4. The maximum color assigned is the bound.
        
        # However, extracting and sorting every time is expensive.
        # Standard Tomita uses a specific vertex ordering and processes bits from high to low.
        
        # Let's implement the standard "ColorSort" step.
        # We maintain an array 'color' for each vertex.
        
        # For this specific problem size, a slightly less optimized but correct version might pass.
        # But let's try to be efficient.
        
        # We will use a list of vertices in P, sorted by index (or degree).
        # Let's stick to index order for simplicity first, or degree order precomputed?
        # Degree in Comp_G changes as P shrinks.
        
        # Let's use the "greedy coloring" that processes vertices in P in decreasing order of index.
        # This is standard in many Python implementations of Tomita.
        
        colors = [0] * m
        color_classes = [] # list of bitmasks for each color class
        
        # We need to iterate through bits in P.
        # Let's get the list of vertices in P.
        verts_in_P = []
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            idx = lsb.bit_length() - 1
            verts_in_P.append(idx)
            p_temp ^= lsb
            
        # Sort by degree in Comp_G?
        # Let's just use index order for now, or maybe reverse index order (higher index first)
        # which often correlates with lower degree if we sorted by degree initially?
        # We sorted initial vertices by degree descending.
        # So index 0 has highest degree.
        # Processing higher indices (lower degree) first might be better for coloring?
        # Or processing lower indices (higher degree) first?
        # Usually, coloring high degree vertices first leads to more colors?
        # We want FEWER colors to prune? No, we want the bound to be TIGHT (low).
        # So we want a coloring with few colors.
        # Coloring high degree vertices first often results in a valid coloring with fewer colors?
        # Let's try sorting verts_in_P by degree in Comp_G descending.
        
        # Degree in Comp_G for vertex i is popcount(comp_adj[i] & P)
        
        # This calculation is O(N) per vertex, so O(N^2) per node.
        # With N=205, this is ~40k ops per node.
        # If we have many nodes, it might be slow.
        
        # Alternative: Use precomputed degrees and update? Hard.
        # Let's just use the index order (which is degree-sorted initially).
        # Since we process P, the relative order is preserved.
        
        # Let's iterate through verts_in_P.
        # To get a tight bound, we should assign colors such that adjacent vertices (in Comp_G) get different colors.
        
        # We'll maintain an array 'used_colors' for vertices?
        # Or just check against color_classes.
        
        # color_classes[c] is a bitmask of vertices already assigned color c.
        
        # For each vertex v in verts_in_P:
        #   Find the smallest c such that (comp_adj[v] & color_classes[c]) == 0
        #   If found, assign v to color c.
        #   Else, create new color class.
        
        # This is O(K * N) per vertex, where K is number of colors.
        
        # Let's implement this.
        
        # Sort verts_in_P by degree in Comp_G (descending) to improve coloring?
        # Let's try it.
        # deg_comp = [bin(comp_adj[v] & P).count('1') for v in verts_in_P]
        # sorted_verts = [v for v, _ in sorted(zip(verts_in_P, deg_comp), key=lambda x: -x[1])]
        
        # This is expensive. Let's skip dynamic sorting and rely on the initial degree ordering.
        # The initial ordering was by degree in G (descending).
        # Degree in Comp_G is (N-1) - Degree_in_G.
        # So initial ordering is by degree in Comp_G ASCENDING.
        # Processing low degree vertices first in coloring might be better?
        
        # Let's just iterate through verts_in_P in the order they appear in P (which is index order).
        
        # Actually, standard Tomita processes vertices in a specific order to build the color classes.
        
        # Let's refine the coloring step:
        # We want to minimize the number of colors.
        
        # We'll use a list of color masks.
        color_masks = []
        
        # We process vertices in verts_in_P
        # To improve bound, we should process vertices with high degree in the induced subgraph first?
        # Let's just use the order from P (low to high index).
        
        for v in verts_in_P:
            assigned = False
            for c_idx, mask in enumerate(color_masks):
                # Check if v is adjacent to any vertex in mask in Comp_G
                # If NOT adjacent to any, v can join this color class.
                # Adjacent in Comp_G means comp_adj[v] & mask != 0
                if (comp_adj[v] & mask) == 0:
                    color_masks[c_idx] |= (1 << v)
                    assigned = True
                    break
            if not assigned:
                color_masks.append(1 << v)
                
        # The number of colors is len(color_masks)
        # This is an upper bound on the clique size in Comp_G (which is MIS in G).
        # So if size + len(color_masks) <= best_size, we can prune.
        
        if size + len(color_masks) <= best_size:
            return
            
        # Now we branch.
        # We iterate through color classes in reverse order (highest color first).
        # This is a heuristic to find large cliques faster.
        
        for c_idx in range(len(color_masks) - 1, -1, -1):
            # Get vertices in this color class that are in P
            # Wait, color_masks contains vertices from P.
            # We need to pick one vertex from this color class to add to the clique.
            
            # Let mask_c = color_masks[c_idx]
            # We iterate through vertices in mask_c
            
            # To improve efficiency, we can iterate through bits in mask_c
            
            # Also, we need to update P and X.
            
            # Standard Tomita:
            # for v in vertices of color class c (in some order):
            #   if size + (upper bound for remaining) <= best: break
            #   add v to clique
            #   new_P = P & ~comp_adj[v] & ~(bits lower than v)  # Only consider neighbors in P that are "after" v?
            #   Actually, to avoid duplicates, we only consider vertices in P that are adjacent to v in Comp_G 
            #   AND have index > v (if we process in order).
            
            # Let's use the standard approach:
            # We maintain P as the set of candidates.
            # When we pick v, the new candidates are P & comp_adj[v] & ((1 << v) - 1) ?
            # No, if we process vertices in increasing index order, we restrict to higher indices to avoid permutations.
            
            # Let's stick to:
            # new_P = P & comp_adj[v] & ((1 << v) - 1) is WRONG if we want higher indices.
            # If we want higher indices: new_P = P & comp_adj[v] & ~((1 << (v+1)) - 1)
            
            # Let's define the order: we process vertices in increasing index.
            # So when we pick v, we can only pick neighbors w such that w > v.
            
            mask_c = color_masks[c_idx]
            
            # Iterate through vertices in mask_c
            # We should iterate in increasing index order?
            
            # Let's extract bits from mask_c
            
            temp_mask = mask_c
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                temp_mask ^= lsb
                
                # Pruning check:
                # The maximum clique size from here is size + 1 + (bound for new_P)
                # But we don't have the bound for new_P yet.
                # We can use the fact that the number of colors in the subgraph induced by new_P is <= number of colors in P?
                # Not necessarily.
                
                # Let's just recurse.
                
                new_P = P & comp_adj[v] & ~((1 << (v + 1)) - 1)
                
                # Update X
                X.append(v)
                
                expand(new_P, X, size + 1)
                
                X.pop()
                
                # Remove v from P so we don't pick it again in this level
                P ^= (1 << v)
                
                # Pruning: if size + 1 + popcount(P) <= best_size, break?
                # popcount(P) is an upper bound on the number of vertices we can add.
                # So if size + 1 + bin(P).count('1') <= best_size:
                #     break
                
                if size + 1 + bin(P).count('1') <= best_size:
                    break

    # Initialize
    P = (1 << m) - 1
    X = []
    
    # Start
    expand(P, X, 0)
    
    # Map back to original indices
    result_indices = [order[v] for v in best_set]
    
    return np.array(result_indices, dtype=np.int64)

# EVOLVE-BLOCK-END
