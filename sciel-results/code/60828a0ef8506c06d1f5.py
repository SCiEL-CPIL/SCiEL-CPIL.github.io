import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Finds a maximum cardinality subset of [1, 256] such that for all distinct a, b in the subset,
    (a * b) % (a + b) != 0.
    Uses Tomita Max Clique algorithm on the complement graph of the divisibility graph.
    """
    N = 256
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # 1. Build the graph of "good" pairs
    # good_adj[u] is a bitmask of v such that (u*v) % (u+v) != 0
    good_adj = [0] * (N + 1)
    
    for u in range(1, N + 1):
        mask = 0
        for v in range(1, N + 1):
            if u == v:
                continue
            if (u * v) % (u + v) != 0:
                mask |= (1 << (v - 1))
        good_adj[u] = mask

    best_clique_mask = 0
    best_size = 0
    # Initial lower bound: Use a greedy heuristic to get a good starting point
    # Greedy: pick vertex with max degree, add to clique, remove it and its non-neighbors?
    # For Max Clique, greedy: pick vertex, intersect neighbors.
    # Let's just start with 0. The algorithm will find something quickly.
    
    # To speed up, we can find a large clique greedily first.
    # Greedy Max Clique:
    P = (1 << N) - 1
    R = 0
    while P:
        # Pick vertex with max degree in P
        max_deg = -1
        best_v = -1
        # Iterate through bits in P
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            v = lsb.bit_length() - 1
            # degree in P is popcount(good_adj[v+1] & P)
            deg = (good_adj[v + 1] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            p_temp ^= lsb
        if best_v == -1:
            break
        R |= (1 << best_v)
        P &= good_adj[best_v + 1]
        P &= ~(1 << best_v) # Remove v itself
        
    if R.bit_count() > best_size:
        best_size = R.bit_count()
        best_clique_mask = R

    def expand(P, R):
        nonlocal best_size, best_clique_mask
        
        if time.time() > deadline:
            return

        if not P:
            if R.bit_count() > best_size:
                best_size = R.bit_count()
                best_clique_mask = R
            return

        # Greedy coloring of P to get an upper bound and vertex ordering
        # Returns (num_colors, list_of_vertices_sorted_by_color_desc)
        def color_bound(P):
            colors = []
            current_P = P
            color_idx = 0
            while current_P:
                # Find a maximal independent set in current_P
                # This is equivalent to finding a clique in the complement, 
                # but we are coloring the graph P (edges are good pairs).
                # A color class is an independent set in P (no edges between them).
                # Wait. We are finding a Max Clique in the graph H (good pairs).
                # The Tomita algorithm uses coloring of the *candidate set P* in the graph H.
                # The bound is the chromatic number of the subgraph induced by P.
                # The chromatic number is an upper bound on the clique size.
                # To color:
                # While P is not empty:
                #   Increment color
                #   Find Max Independent Set in P? No, just a maximal one is enough for a valid bound.
                #   Actually, for the bound to be tight, we want to assign colors such that
                #   each color class is an independent set.
                #   So, we pick a vertex v, assign it color c, remove v and its neighbors from P?
                #   No, if v has color c, its neighbors cannot have color c.
                #   So, we remove v and all its neighbors in P from the current color class candidates.
                
                color_class = []
                avail = current_P
                while avail:
                    # Pick a vertex from avail
                    lsb = avail & -avail
                    v = lsb.bit_length() - 1
                    color_class.append(v)
                    # Remove v and its neighbors in current_P from avail
                    # Neighbors of v in current_P are good_adj[v+1] & current_P
                    avail &= ~(good_adj[v + 1] | lsb)
                colors.append(color_class)
                color_idx += 1
                # Remove the color class from current_P
                mask_to_remove = 0
                for v in color_class:
                    mask_to_remove |= (1 << v)
                current_P &= ~mask_to_remove
            
            num_colors = len(colors)
            # Flatten colors in reverse order (highest color first) for branching
            # Vertices with higher colors are more likely to be in a large clique?
            # Standard Tomita: branch on vertices in reverse order of coloring.
            ordered_vertices = []
            for i in range(num_colors - 1, -1, -1):
                ordered_vertices.extend(colors[i])
            return num_colors, ordered_vertices

        k, ordered_vertices = color_bound(P)
        
        # Prune if upper bound is not better
        if R.bit_count() + k <= best_size:
            return

        for v in ordered_vertices:
            if time.time() > deadline:
                return
            
            # Check if v is still in P (it might have been removed if we modified P? 
            # In this implementation, we pass P by value (int is immutable), 
            # but we update P in the loop. 
            # However, the ordering was computed on the initial P.
            # If v is no longer in P, skip it.
            if not (P & (1 << v)):
                continue
                
            # Branch
            new_R = R | (1 << v)
            new_P = P & good_adj[v + 1]
            
            expand(new_P, new_R)
            
            # Remove v from P for subsequent iterations
            P &= ~(1 << v)
            
            # Update bound: k decreases by 1 as we remove a vertex from the top color class
            # This is a valid (though loose) update
            k -= 1
            
            if R.bit_count() + k <= best_size:
                break

    # Start search
    # P is all vertices
    P = (1 << N) - 1
    R = 0
    expand(P, R)

    # Convert best_clique_mask to list of integers
    result = []
    for i in range(N):
        if best_clique_mask & (1 << i):
            result.append(i + 1)
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
