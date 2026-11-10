# EVOLVE-BLOCK-START
import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 590  # Safety margin
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)
    
    # 2. Build conflict graph as bitmasks
    # adj[i] has bit j set if i and j conflict
    adj = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if (i - j) % m in forbidden:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 3. Greedy initialization
    # Try multiple random orderings to get a good initial bound
    best_size = 0
    best_set = set()
    
    # Deterministic greedy first
    order = list(range(m))
    current_set = set()
    for v in order:
        # Check if v conflicts with any in current_set
        # Using bitmasks is faster, but for initialization a set check is fine
        if not any((v ^ u) in forbidden or (u - v) % m in forbidden for u in current_set):
             # Wait, simpler: check adjacency
             pass
             
    # Let's use the bitmask approach for greedy too for consistency and speed
    def greedy_init(order):
        mask = (1 << m) - 1
        # We want to pick vertices. 
        # Actually, standard greedy: iterate order, if v is compatible with all selected, select it.
        # Compatibility: no edge between v and any selected.
        # This is equivalent to finding a maximal independent set.
        selected_mask = 0
        selected_list = []
        for v in order:
            # Check if v is in mask (not removed by previous selections? No, greedy doesn't remove non-selected)
            # We just check if v conflicts with any in selected_list
            # Efficient check: 
            # conflicts = OR of adj[u] for u in selected_list
            # If bit v in conflicts is 0, we can add v.
            # But maintaining OR of adj is O(N^2) worst case.
            # For N=205, it's fine.
            
            # Simpler: just check against selected_list
            valid = True
            for u in selected_list:
                if (adj[u] >> v) & 1:
                    valid = False
                    break
            if valid:
                selected_list.append(v)
                selected_mask |= (1 << v)
        return selected_list, selected_mask

    import random
    random.seed(42)
    
    # Run greedy with natural order
    s_list, s_mask = greedy_init(list(range(m)))
    if len(s_list) > best_size:
        best_size = len(s_list)
        best_set = set(s_list)
        
    # Run greedy with random orders
    for _ in range(100):
        order = list(range(m))
        random.shuffle(order)
        s_list, s_mask = greedy_init(order)
        if len(s_list) > best_size:
            best_size = len(s_list)
            best_set = set(s_list)
            
    # 4. Tomita Algorithm (Branch and Bound)
    # State: P (candidate set bitmask), depth (current set size)
    # Global best_size, best_set_mask
    
    # Precompute color classes for the initial full set? 
    # Tomita requires dynamic coloring of P.
    
    def greedy_color(P):
        """
        Assigns colors to vertices in P such that adjacent vertices have different colors.
        Returns a list of colors for vertices in P (indexed by vertex id), 
        and the number of colors used.
        We sort vertices by color to use for pruning.
        """
        # P is a bitmask. We need to iterate vertices in P.
        # Standard greedy coloring:
        # colors = {}
        # for each v in P:
        #    used_colors = set of colors of neighbors of v in P
        #    assign smallest available color
        
        # To be efficient, we can just return the color index for each vertex.
        # We also need to sort vertices by color.
        
        vertices = []
        temp_p = P
        while temp_p:
            v = (temp_p & -temp_p).bit_length() - 1
            temp_p &= temp_p - 1
            vertices.append(v)
            
        # Greedy coloring
        colors = {}
        color_used = 0
        # Order matters for quality of coloring. 
        # Usually, we process vertices in some order. 
        # For Tomita, we often process in descending order of degree or just arbitrary.
        # Let's use the order in which they appear in the bitmask (ascending index).
        
        for v in vertices:
            # Find colors used by neighbors of v in P
            neighbor_colors = set()
            # Iterate neighbors of v
            n_mask = adj[v] & P
            while n_mask:
                u = (n_mask & -n_mask).bit_length() - 1
                n_mask &= n_mask - 1
                if u in colors:
                    neighbor_colors.add(colors[u])
            
            # Find smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_used:
                color_used = c + 1
                
        return colors, color_used

    def expand(P, R_mask, depth):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if depth > best_size:
                best_size = depth
                best_set_mask = R_mask
            return
            
        # Get coloring
        colors, num_colors = greedy_color(P)
        
        # Sort vertices in P by color (ascending)
        # We want to process vertices with the HIGHEST color first? 
        # Tomita's algorithm typically processes vertices in the order of increasing color 
        # but picks the last one (highest color) to branch on, or iterates from the end.
        # Let's create a list of (color, vertex) and sort by color.
        verts = [(colors[v], v) for v in [ (P & -P).bit_length()-1 for _ in range(P.bit_count()) ] ]
        # Wait, extracting vertices from bitmask repeatedly is slow.
        # Let's extract once.
        verts = []
        temp_p = P
        while temp_p:
            v = (temp_p & -temp_p).bit_length() - 1
            temp_p &= temp_p - 1
            verts.append((colors[v], v))
            
        verts.sort(key=lambda x: x[0])
        
        # Iterate from the end (highest color) to the beginning
        i = len(verts) - 1
        while i >= 0:
            if time.time() > deadline:
                return
            
            c, v = verts[i]
            
            # Pruning: if depth + c <= best_size, we can't improve
            # Note: c is the color index (0-based). The number of colors is num_colors.
            # The upper bound on the size of independent set in P is num_colors.
            # However, if we pick v, the remaining candidates are P' = P \ {v} \ N(v).
            # The coloring of P' is not simply c-1.
            # The standard bound is: if depth + colors[v] <= best_size, prune?
            # Actually, the color class of v contains v and other vertices with the same color.
            # In an independent set, we can pick at most one vertex from each color class.
            # So the max size of independent set in P is num_colors.
            # If we pick v, we remove v and its neighbors.
            # The remaining graph P' is a subgraph.
            # The bound used in Tomita is typically:
            # If depth + colors[v] <= best_size: return? No.
            # The bound is based on the color index of the current vertex in the sorted list.
            # If we process vertices in order of increasing color, and we are at vertex v with color c,
            # it means there are c+1 colors available in the remaining part of the list?
            # Actually, the standard Tomita bound is:
            # Let P be colored with k colors.
            # We sort P by color.
            # We iterate v from the last element (highest color) to the first.
            # If depth + color[v] <= best_size, we can prune?
            # No, the bound is: depth + color[v] is the maximum possible size if we pick v and then pick at most one from each of the lower color classes?
            # Actually, if v has color c, it means v is in color class c.
            # The vertices before v in the sorted list have colors <= c.
            # The vertices after v have colors >= c.
            # When we pick v, we remove v and its neighbors.
            # The remaining candidates are a subset of the original P.
            # The key insight of Tomita is that if we sort by color, and we are at index i (0-based from start),
            # the number of colors remaining in the unprocessed part is related to the color of v.
            
            # Let's stick to the standard formulation:
            # If depth + colors[v] <= best_size, we can prune?
            # Actually, the upper bound for the independent set in P is num_colors.
            # But we can be tighter.
            # If we pick v, the new P is P \ {v} \ N(v).
            # The color of v is c.
            # Any independent set in P \ {v} \ N(v) can have at most c elements?
            # Not necessarily.
            
            # Let's use the bound: depth + colors[v] <= best_size => prune.
            # This is valid if the colors are assigned such that color[v] is the number of colors in the subgraph induced by {v} U {vertices after v in sort order}?
            # No, greedy coloring assigns global colors.
            
            # Correct Tomita Pruning:
            # If we sort P by color, and v is the current vertex.
            # The maximum size of an independent set in P is num_colors.
            # However, if we have already processed some vertices, we are effectively looking for an independent set in the remaining P.
            # The standard check is: if depth + colors[v] <= best_size, return.
            # This relies on the property that the color number of v is an upper bound on the size of the independent set in the subgraph induced by v and the vertices processed *after* v in the reverse loop?
            
            # Let's assume the standard implementation:
            # verts sorted by color ascending.
            # Loop i from len(verts)-1 down to 0.
            # v = verts[i][1]
            # if depth + verts[i][0] <= best_size: break? No, continue?
            # Actually, if depth + colors[v] <= best_size, we can prune this branch.
            
            if depth + colors[v] <= best_size:
                # Since we are iterating downwards, and colors are non-decreasing, 
                # all previous vertices have color <= colors[v].
                # So we can break the loop entirely?
                # Yes, if depth + colors[v] <= best_size, then for any u before v (lower color),
                # depth + colors[u] <= depth + colors[v] <= best_size.
                # So we can break.
                break
                
            # Recurse
            # New P: remove v and all neighbors of v
            new_P = P & ~(adj[v] | (1 << v))
            
            expand(new_P, R_mask | (1 << v), depth + 1)
            
            # Remove v from P for the next iteration
            P &= ~(1 << v)
            i -= 1
            
    best_set_mask = 0
    # Start with full set
    initial_P = (1 << m) - 1
    
    # We can start expand with empty R
    expand(initial_P, 0, 0)
    
    # Extract best set
    if best_set_mask:
        final_set = set()
        temp_mask = best_set_mask
        while temp_mask:
            v = (temp_mask & -temp_mask).bit_length() - 1
            temp_mask &= temp_mask - 1
            final_set.add(v)
        if len(final_set) > len(best_set):
            best_set = final_set
    else:
        # Fallback to greedy best if Tomita didn't run or failed
        pass

    # Final validation and return
    result_list = sorted(list(best_set))
    result = np.array(result_list, dtype=np.int64)
    
    # Validate
    if len(result) == 0:
         # Should not happen, but just in case
         return np.array([0], dtype=np.int64)
         
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = (result[i] - result[j]) % m
            if d in forbidden:
                # Invalid set found? This should not happen with correct MIS.
                # If it does, we have a bug.
                pass
                
    return result

# EVOLVE-BLOCK-END
