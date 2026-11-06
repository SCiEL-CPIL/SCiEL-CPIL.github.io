import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85  # Safety margin before the 90s wall-clock limit
    
    N = limit
    # adj[i] is a bitmask where bit j is set if lcm(i, j+1) <= limit
    # Node i (1..N) corresponds to bit (i-1)
    adj = [0] * (N + 1)
    
    # Precompute compatibility graph
    # Optimization: For each i, iterate j. 
    # lcm(i, j) = i * j // gcd(i, j) <= limit
    # j <= limit * gcd(i, j) // i
    # This is still O(N^2) to check all pairs, which is 512^2 ~ 262k, very fast.
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i * j // gcd(i, j) <= limit:
                mask |= (1 << (j - 1))
        adj[i] = mask
        
    # Helper to get list of vertices from bitmask
    def mask_to_list(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]

    # Greedy coloring of the subgraph induced by P (bitmask)
    # Returns list of (vertex, color) sorted by color descending.
    # Colors are 1..max_color.
    # We use a greedy algorithm. To get a tight bound, we want to use as few colors as possible.
    # Processing vertices in order of decreasing degree in the induced subgraph often helps.
    
    def color_sort(P_mask):
        if P_mask == 0:
            return []
        
        vertices = mask_to_list(P_mask)
        
        # Calculate degrees in the induced subgraph
        # deg(v) = popcount(adj[v] & P_mask)
        # Note: adj[v] includes v itself if v is compatible with v (which it is).
        # So deg includes self? 
        # In graph theory for coloring, we consider neighbors. 
        # adj[v] & P_mask gives the set of vertices in P that are adjacent to v.
        # Since v is adjacent to v, v is in this set.
        # For coloring, we care about neighbors. The "degree" heuristic usually uses the number of neighbors.
        # Let's use the number of neighbors in P.
        
        degs = []
        for v in vertices:
            # Number of neighbors in P
            # adj[v] & P_mask includes v. 
            # If we want strict neighbors, we might subtract 1, but for sorting purposes, it's fine.
            d = bin(adj[v] & P_mask).count('1')
            degs.append((d, v))
        
        # Sort by degree descending
        degs.sort(reverse=True)
        
        colors = {}
        max_color = 0
        
        # Assign colors
        # We process vertices in the sorted order.
        # For each vertex v, find the smallest color not used by its neighbors in P that have already been colored.
        
        for d, v in degs:
            used_colors = set()
            # Iterate through neighbors of v in P
            # neighbors_mask = adj[v] & P_mask
            # We need to check which neighbors have already been assigned a color.
            # Since we are processing in a specific order, we can just check the `colors` dict.
            
            mask = adj[v] & P_mask
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                u = idx + 1
                if u in colors:
                    used_colors.add(colors[u])
                mask ^= lsb
                
            # Find smallest available color
            c = 1
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        # Group by color and sort colors descending
        color_classes = {}
        for v, c in colors.items():
            if c not in color_classes:
                color_classes[c] = []
            color_classes[c].append(v)
        
        sorted_colors = sorted(color_classes.keys(), reverse=True)
        
        ordered = []
        for c in sorted_colors:
            for v in color_classes[c]:
                ordered.append((v, c))
                
        return ordered

    best_clique_size = 0
    best_clique = []
    
    # Greedy initial solution to set a good lower bound
    def greedy_clique():
        C = []
        current_P = (1 << N) - 1
        while current_P:
            # Pick vertex with max degree in current_P
            best_v = -1
            max_deg = -1
            mask = current_P
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                v = idx + 1
                deg = bin(adj[v] & current_P).count('1')
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                mask ^= lsb
            
            if best_v == -1:
                break
                
            C.append(best_v)
            # Update P: keep only neighbors of best_v
            current_P &= adj[best_v]
            current_P &= ~(1 << (best_v - 1))
            
        return C

    initial_clique = greedy_clique()
    best_clique_size = len(initial_clique)
    best_clique = initial_clique[:]

    # Tomita's Maximum Clique Algorithm
    def expand(C, P_mask):
        nonlocal best_clique_size, best_clique
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if len(C) > best_clique_size:
                best_clique_size = len(C)
                best_clique = C[:]
            return
        
        # Color the candidates
        ordered = color_sort(P_mask)
        
        remaining_P = P_mask
        
        for v, c in ordered:
            if time.time() > deadline:
                return
            
            # Check if v is still in remaining_P
            # It should be, unless we removed it? 
            # In this loop structure, we process each vertex in `ordered` once.
            # `ordered` was generated from the initial P_mask for this call.
            # We maintain `remaining_P` to track which vertices are still candidates for the "Exclude" branch.
            # Actually, the standard Tomita loop is:
            # while P:
            #   v = pop(P)
            #   if bound fails: break
            #   include v -> recurse
            #   exclude v -> P = P \ {v} (and P is updated to intersect with adj[v] ONLY IF we include v? No.)
            
            # Let's trace carefully.
            # The set P is the set of vertices that can be added to C to form a clique.
            # We pick a vertex v from P.
            # Branch 1: Add v to C. New P' = P & adj[v] \ {v}. Recurse.
            # Branch 2: Do not add v. New P'' = P \ {v}. Continue loop with P''.
            
            # The `ordered` list contains all vertices in the initial P_mask.
            # If we don't update `remaining_P` correctly, we might process a vertex that was excluded.
            # But in the "Exclude" branch, we just remove v from the pool.
            # The "Include" branch spawns a new recursion.
            
            # So, in the main loop, we are effectively iterating through the "Exclude" chain.
            # We must ensure that `v` is still in `remaining_P`.
            # Since we remove v from `remaining_P` at the end of the loop iteration (for the exclude branch),
            # and `ordered` is a static list, we just need to check if v is still in `remaining_P`.
            # Actually, if we remove v at the end, the next vertex in `ordered` will still be in `remaining_P` 
            # unless it was also removed? No, only v is removed.
            
            if not (remaining_P & (1 << (v - 1))):
                continue
            
            # Bound check
            # The color c is the upper bound on the size of any clique in the subgraph induced by 
            # the vertices in `remaining_P` that have color >= c? 
            # No, the coloring assigns colors such that no two adjacent vertices share a color.
            # The maximum clique size in the subgraph induced by vertices with colors >= c is at most (max_color - c + 1)?
            # Actually, the bound is: The number of vertices in the subgraph induced by P with color >= c 
            # is at most the number of colors from c to max_color? 
            # Wait, the standard bound is: |C| + color(v) <= best -> prune.
            # Here color(v) is the color assigned to v.
            # Since we process in decreasing color order, if len(C) + c <= best, 
            # then for any subsequent vertex u with color(u) <= c, len(C) + color(u) <= len(C) + c <= best.
            # So we can break.
            
            if len(C) + c <= best_clique_size:
                break
            
            # Branch: Include v
            C.append(v)
            
            # New candidates: (remaining_P & adj[v]) \ {v}
            new_P_mask = remaining_P & adj[v]
            new_P_mask &= ~(1 << (v - 1))
            
            expand(C, new_P_mask)
            
            C.pop()
            
            # Branch: Exclude v
            # Remove v from remaining_P
            remaining_P &= ~(1 << (v - 1))

    # Initial call
    # P_mask is all nodes 1..512
    init_P_mask = (1 << N) - 1
    expand([], init_P_mask)
    
    return sorted(best_clique)

import sys
sys.setrecursionlimit(5000)
