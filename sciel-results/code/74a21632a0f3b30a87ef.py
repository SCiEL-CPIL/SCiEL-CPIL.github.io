import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    n = limit
    # Set a deadline slightly below the 90s hard limit to be safe
    deadline = start_time + 85.0
    
    # 1. Precompute neighbors as bitmasks
    # neighbors[i] is an integer where bit j is 1 if lcm(i+1, j+1) <= 512
    neighbors = [0] * n
    
    for i in range(n):
        num_i = i + 1
        mask = 0
        for j in range(n):
            if i == j:
                continue
            num_j = j + 1
            g = gcd(num_i, num_j)
            lcm_val = (num_i * num_j) // g
            if lcm_val <= limit:
                mask |= (1 << j)
        neighbors[i] = mask

    best_size = 0
    best_clique = [] # List of vertices (0-indexed)
    
    # Tomita's Algorithm for Maximum Clique
    # Uses greedy coloring to provide upper bounds for pruning
    
    def color_sort(vertices_mask):
        """
        Greedily colors the vertices in vertices_mask.
        Returns a list of vertices sorted by color (ascending) and the max color used.
        vertices_mask: bitmask of vertices to color.
        """
        if vertices_mask == 0:
            return [], 0
            
        # We process vertices one by one.
        # To make it fast, we just iterate through bits.
        # However, standard Tomita sorts by color.
        # Greedy coloring: assign smallest possible color.
        
        # For performance, we can store colors in a dict or list.
        # Since N=512, a list of size 512 is fine.
        colors = [0] * n
        max_color = 0
        uncolored = vertices_mask
        colored_vertices = []
        
        # To speed up finding available colors, we can just check 1..max_color
        # But for N=512, max_color is small.
        
        # We need to iterate through vertices in a specific order?
        # Tomita suggests ordering by degree or similar.
        # For simplicity and speed, we iterate through bits in increasing order.
        # A better heuristic: process vertices with higher degree in the subgraph first?
        # Let's stick to simple bit iteration for now, it's often sufficient.
        
        m = uncolored
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # Find min color not used by neighbors of v in the colored set
            # Neighbors of v in the colored set:
            # We need to know colors of colored neighbors.
            # This is tricky with bitmasks.
            
            # Alternative: Keep a list of colored vertices.
            # For each new v, check colors of all colored neighbors.
            # This is O(N^2) per coloring step.
            # With N=512, and potentially many coloring steps, this might be slow in Python.
            
            # Optimization: Precompute neighbor lists?
            # Or use a different coloring bound?
            
            # Let's use a simpler bound if coloring is too slow:
            # Bound by degree? No.
            
            # Let's try to optimize the coloring.
            # We maintain `colored_list` and `color_of` array.
            # For v, we look at neighbors in colored_list.
            
            # To avoid O(N^2) inside the loop, we can precompute 
            # intersection of neighbors[v] with the set of already colored vertices.
            # But the set of colored vertices changes.
            
            # Given N=512, O(N^2) is ~260k ops.
            # If we do this 512 times, that's 134M ops. In Python, this is ~10-20s.
            # Too slow if called many times.
            
            # We need a faster coloring or a different bound.
            # Maybe use a simple greedy bound: 
            # The size of the maximum clique in P is at most the size of a maximum matching? No.
            # At most the number of vertices? Yes, trivial.
            
            # Let's stick to Tomita but optimize the inner loop.
            # Actually, we can just use a simple greedy coloring on the induced subgraph.
            
            # Revised Color Sort:
            # 1. Extract list of vertices from mask.
            # 2. Order them (e.g., by degree in the subgraph, or just index).
            # 3. Assign colors greedily.
            
            # Let's implement a fast greedy coloring.
            # We will store the list of vertices to return.
            
            # To speed up, we can use the fact that we only need the *bound*.
            # The bound is the max color assigned.
            
            # Let's try a different approach for the bound:
            # Use a simple greedy independent set on the complement? No.
            
            # Let's try to implement the standard Tomita coloring efficiently.
            # Key insight: We can precompute the neighbors of each node as a bitmask.
            # For a node v, the set of colored neighbors is `neighbors[v] & colored_mask`.
            # We need to know the colors of these neighbors.
            # We can maintain an array `node_color` of size n.
            # For v, we iterate through `neighbors[v] & colored_mask` to find used colors.
            
            # This is still potentially slow if done naively.
            
            # However, N=512 is small enough that a well-optimized Python loop might work 
            # if we minimize overhead.
            
            # Let's try a different strategy for the bound if coloring is too slow:
            # Use a recursive depth limit or a simpler heuristic.
            
            # Actually, let's just implement the coloring carefully.
            
            # We'll do the coloring in the caller or here?
            # Here is a fast version:
            
            # 1. Get list of vertices in mask.
            verts = []
            mm = vertices_mask
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                verts.append(v)
                mm ^= lsb
            
            # Sort by degree in the subgraph? 
            # For now, just use the order they appear (increasing index).
            # A better order: sort by degree in the original graph descending.
            # This often leads to tighter colorings.
            # Precompute degrees.
            
            # Let's assume the simple order is fast enough for the first pass.
            
            # Greedy Coloring
            # colors[v] is the color assigned to v
            # We want to assign min color not used by neighbors.
            
            # To speed up "find min color not used", we can use a small array of booleans.
            # Max color is at most len(verts).
            
            # We iterate through verts.
            # For each v, we check neighbors in `colored_set`.
            
            # This is getting complex. Let's use a simpler bound first:
            # Bound = len(vertices_mask). 
            # This is very loose.
            
            # Let's try to implement a simple greedy coloring that is reasonably fast.
            
            # We'll use a dictionary for colors of colored nodes.
            # And a bitmask for colored nodes.
            
            # Actually, there is a known fast implementation of Tomita's coloring in Python 
            # for N<1000.
            
            # Let's try this:
            # 1. Order vertices by degree (precomputed global degree, descending).
            # 2. Assign colors.
            
            # Precomputed degrees:
            # degs = [bin(neighbors[i]).count('1') for i in range(n)]
            
            # Let's pass the vertices list and sort it.
            
            # To avoid sorting every time (expensive), we can just use the natural order 
            # or a precomputed static order.
            # Static order by degree is good.
            
            # Let's generate the static order once outside.
            # But this function is local. We can access it from closure if defined outside.
            
            # For now, let's just return a trivial bound if coloring is too complex to code quickly.
            # No, the score depends on tight bounds.
            
            # Let's implement a simple greedy coloring:
            
            # 1. List of vertices in mask, sorted by global degree descending.
            # 2. For each v, check colors of neighbors that are already colored.
            
            # To make this fast, we precompute the global degree order.
            # But we are inside the function.
            
            # Let's just use the index order for now and see if it's fast enough.
            # If not, we can optimize.
            
            # Actually, for N=512, even O(N^2) coloring might be okay if the tree is pruned well.
            
            # Let's implement the coloring properly.
            
            # We need a global list of degrees.
            # Let's assume `degs` is available in the closure.
            
            # Sort verts by degs descending
            # verts.sort(key=lambda x: -degs[x])
            
            # This sort is O(N log N).
            
            # Now color.
            # color_of = {}
            # colored_mask = 0
            # max_c = 0
            # sorted_verts = []
            
            # for v in verts:
            #    # find min color
            #    used = set()
            #    # neighbors of v that are colored
            #    neigh_mask = neighbors[v] & colored_mask
            #    # iterate bits in neigh_mask
            #    nm = neigh_mask
            #    while nm:
            #        lsb = nm & -nm
            #        u = lsb.bit_length() - 1
            #        used.add(color_of[u])
            #        nm ^= lsb
            #    
            #    c = 1
            #    while c in used:
            #        c += 1
            #    
            #    color_of[v] = c
            #    if c > max_c:
            #        max_c = c
            #    sorted_verts.append((c, v))
            #    colored_mask |= (1 << v)
            
            # # Sort sorted_verts by color
            # sorted_verts.sort(key=lambda x: x[0])
            # final_verts = [v for c, v in sorted_verts]
            
            # return final_verts, max_c

            # This is O(N * (N + MaxColor)).
            
            # Let's code this inside.
            
            # We need `degs`.
            # Let's define `degs` in the outer scope.
            
            # For now, let's just return a trivial bound to get a working skeleton, 
            # then optimize.
            
            # No, I must provide the full solution.
            
            # I will implement the coloring as described.
            
            # To access `degs`, I will define it in `construct` before this function.
            
            # Let's re-structure:
            # 1. Precompute `degs`.
            # 2. Define `color_sort` using `degs`.
            
            # I will modify the code structure in the final output.
            
            # For this inner block, I will assume `degs` is available.
            
            # Since I'm writing the whole file, I will put `degs` in `construct`.
            
            # Let's refine the `color_sort` implementation to be as fast as possible.
            
            # Optimization: Use lists instead of dicts for colors if possible?
            # `color_of` array of size n, initialized to 0.
            # But we need to reset it? No, we only look at colored nodes.
            
            # Let's use a list `color_of` of size n.
            
            # Wait, `color_sort` is called recursively.
            # We can just use a local dict.
            
            # Let's try.
            
            pass # Placeholder for actual implementation in final code

    # To ensure I don't get stuck in this thought loop, I will write the final code 
    # with the full Tomita implementation.
    
    # Precompute degrees for sorting in color_sort
    degs = [bin(neighbors[i]).count('1') for i in range(n)]
    
    # Recursive function
    # R: bitmask of current clique
    # P: bitmask of candidates
    def expand(R_mask, P_mask):
        nonlocal best_size, best_clique, deadline
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            # Update best
            if bin(R_mask).count('1') > best_size:
                best_size = bin(R_mask).count('1')
                # Extract clique
                clique = []
                m = R_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    clique.append(v)
                    m ^= lsb
                best_clique = clique
            return
            
        # Color sort P
        # 1. Get vertices in P
        verts = []
        m = P_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
            
        # Sort by degree descending
        verts.sort(key=lambda x: -degs[x])
        
        # Greedy Coloring
        color_of = {}
        colored_mask = 0
        max_color = 0
        colored_list = [] # List of (color, vertex)
        
        for v in verts:
            # Find min color not used by colored neighbors
            used_colors = set()
            neigh_mask = neighbors[v] & colored_mask
            nm = neigh_mask
            while nm:
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                used_colors.add(color_of[u])
                nm ^= lsb
                
            c = 1
            while c in used_colors:
                c += 1
                
            color_of[v] = c
            if c > max_color:
                max_color = c
            colored_list.append((c, v))
            colored_mask |= (1 << v)
            
        # Sort by color
        colored_list.sort(key=lambda x: x[0])
        # Extract vertices in sorted order
        sorted_verts = [v for c, v in colored_list]
        
        # Tomita's Loop
        # Iterate from the end (highest color) to the start
        # P_remaining is the set of vertices in sorted_verts[0..i]
        # We need to update P_mask as we go?
        # Standard Tomita:
        # P is sorted.
        # for i from len(P)-1 down to 0:
        #   v = P[i]
        #   if |R| + color[i] <= best_size: break
        #   R.add(v)
        #   P_new = P[0..i-1] intersect N(v)
        #   expand(R, P_new)
        #   R.remove(v)
        
        # We need to construct P_new efficiently.
        # P_new_mask = 0
        # for j in range(i):
        #    u = sorted_verts[j]
        #    if neighbors[v] & (1 << u):
        #       P_new_mask |= (1 << u)
        
        # This is O(i) per step. Total O(N^2) per node.
        # With N=512, this is acceptable.
        
        current_size = bin(R_mask).count('1')
        
        for i in range(len(sorted_verts) - 1, -1, -1):
            v = sorted_verts[i]
            c = color_of[v]
            
            if current_size + c <= best_size:
                break
                
            # Add v to R
            new_R = R_mask | (1 << v)
            
            # Calculate P_new
            # P_new consists of vertices in sorted_verts[0..i-1] that are neighbors of v
            P_new = 0
            # Iterate j from 0 to i-1
            # Optimization: We can just iterate through all j < i
            # Since i can be up to 512, this loop is small.
            for j in range(i):
                u = sorted_verts[j]
                if neighbors[v] & (1 << u):
                    P_new |= (1 << u)
                    
            expand(new_R, P_new)
            
            if time.time() > deadline:
                return

    # Initial call
    # Start with all vertices in P
    initial_P = (1 << n) - 1
    expand(0, initial_P)
    
    # Convert best_clique to list of 1-based indices
    result = sorted([v + 1 for v in best_clique])
    
    # Ensure nonempty
    if not result:
        return [1]
        
    return result

construct()
