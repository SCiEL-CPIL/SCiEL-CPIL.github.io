import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    n = limit
    # Set a deadline slightly below the 90s hard limit to be safe
    deadline = start_time + 80.0
    
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
        
    # Precompute degrees for sorting vertices in coloring
    degs = [bin(neighbors[i]).count('1') for i in range(n)]
    # Sort indices by degree descending to use as a static order for coloring
    # This helps in finding a good coloring quickly
    static_order = sorted(range(n), key=lambda x: -degs[x])
    
    best_size = 0
    best_clique = [] # List of vertices (0-indexed)
    
    def expand(R_mask, P_mask):
        nonlocal best_size, best_clique, deadline
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            # Update best
            r_size = bin(R_mask).count('1')
            if r_size > best_size:
                best_size = r_size
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
            
        # 1. Get vertices in P, ordered by static_order (degree desc)
        # We need to filter static_order to only include vertices in P_mask
        # And sort them by their position in static_order? 
        # Actually, for Tomita, we just need a consistent order. 
        # Using static_order restricted to P is a good heuristic.
        
        verts = []
        # Instead of iterating all n and checking mask, we can iterate bits of P_mask
        # But we want them sorted by degree.
        # Since n=512, iterating all n is fast (512 ops).
        # Or we can iterate bits of P_mask and then sort.
        # Let's iterate bits of P_mask and then sort by -degs[v].
        # This is O(|P| log |P|).
        
        m = P_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
            
        # Sort by degree descending
        # Using static_order as a key? Or just degs?
        # degs is a proxy.
        verts.sort(key=lambda x: -degs[x])
        
        # 2. Greedy Coloring
        # We assign colors to vertices in `verts`
        # color_of[v] stores the color assigned to v
        # We maintain a list `colored_vertices` to know which are colored
        # And `colored_mask` for fast neighbor checks
        
        color_of = {}
        colored_mask = 0
        max_color = 0
        # We need to store the color for each vertex in verts to sort them later
        # Let's create a list of tuples (color, vertex)
        colored_list = []
        
        for v in verts:
            # Find min color not used by neighbors of v in colored_mask
            # Neighbors of v in colored set: neighbors[v] & colored_mask
            neigh_mask = neighbors[v] & colored_mask
            
            # Get set of colors used by these neighbors
            used_colors = set()
            nm = neigh_mask
            while nm:
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                used_colors.add(color_of[u])
                nm ^= lsb
                
            # Find smallest positive integer not in used_colors
            c = 1
            while c in used_colors:
                c += 1
                
            color_of[v] = c
            if c > max_color:
                max_color = c
            colored_list.append((c, v))
            colored_mask |= (1 << v)
            
        # 3. Sort colored_list by color ascending
        # This gives us the order for the Tomita loop
        # Vertices with lower colors are processed later in the loop?
        # No, Tomita processes from highest color to lowest?
        # Let's check the standard Tomita algorithm.
        # MCQ/R1:
        # P is sorted by color.
        # for i from |P|-1 down to 0:
        #   v = P[i]
        #   if |R| + color(v) <= best: break
        #   ...
        # So we iterate from the end of the list (highest color) to the beginning.
        # The list `colored_list` is currently unsorted by color? 
        # We appended in `verts` order (degree desc).
        # We need to sort `colored_list` by color.
        
        colored_list.sort(key=lambda x: x[0])
        
        # Now iterate in reverse
        current_size = bin(R_mask).count('1')
        
        for i in range(len(colored_list) - 1, -1, -1):
            c, v = colored_list[i]
            
            if current_size + c <= best_size:
                break
                
            # Add v to R
            new_R = R_mask | (1 << v)
            
            # Calculate P_new
            # P_new consists of vertices in colored_list[0..i-1] that are neighbors of v
            # Note: The vertices in colored_list[0..i-1] are exactly the ones processed "before" v in the sorted order?
            # Wait, the loop goes backwards. 
            # The set of candidates for the next step should be the intersection of 
            # (vertices in P with index < i in the sorted list) and N(v).
            # Since we sorted by color, the "index" in the sorted list is what matters for the bound.
            # The vertices available are those with color < c (or index < i in the sorted list).
            
            P_new = 0
            # Iterate j from 0 to i-1
            for j in range(i):
                _, u = colored_list[j]
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
