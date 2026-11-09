import time

def construct():
    deadline = time.time() + 55
    N = 256
    
    # 1. Build Compatibility Graph H
    # compatible[i] is bitmask of j such that (i*j) % (i+j) != 0 and i != j
    compatible = [0] * (N + 1)
    
    all_mask = (1 << (N + 1)) - 2 # bits 1..256
    
    for a in range(1, N + 1):
        compatible[a] = all_mask & ~(1 << a)
        
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) != 0:
                # Compatible
                compatible[a] |= (1 << b)
                compatible[b] |= (1 << a)
            # else: Conflict, so no edge in H (compatible graph)
            
    best_size = 0
    best_mask = 0
    
    # Helper to get list of vertices from bitmask, sorted by index
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        return verts

    # Recursive search for Max Clique in H (which is Max Independent Set in G)
    def expand(candidates, current_size, current_mask):
        nonlocal best_size, best_mask
        if time.time() > deadline:
            return
            
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return

        # Get vertices in candidates
        verts = get_vertices(candidates)
        
        # Greedy Coloring of H[candidates]
        # Order by degree descending to get tighter bounds earlier?
        # Let's sort by degree in H[candidates]
        
        degs = []
        for v in verts:
            deg = bin(compatible[v] & candidates).count('1')
            degs.append((deg, v))
        degs.sort(reverse=True)
        ordered_verts = [v for _, v in degs]
        
        # Assign colors
        color_classes = [] # list of bitmasks
        colors = {}
        
        for v in ordered_verts:
            assigned = False
            for c, class_mask in enumerate(color_classes):
                # v can take color c if it is not adjacent to any vertex in class_mask
                # In H, adjacent means compatible.
                # So we need (class_mask & compatible[v]) == 0
                if (class_mask & compatible[v]) == 0:
                    color_classes[c] = class_mask | (1 << v)
                    colors[v] = c + 1
                    assigned = True
                    break
            if not assigned:
                color_classes.append(1 << v)
                colors[v] = len(color_classes)
        
        max_color = len(color_classes)
        
        # Precompute suffix max colors
        # suffix_max_colors[i] = max(color[v_j] for j in i..len-1)
        suffix_max_colors = [0] * len(ordered_verts)
        current_max = 0
        for i in range(len(ordered_verts) - 1, -1, -1):
            v = ordered_verts[i]
            c = colors[v]
            if c > current_max:
                current_max = c
            suffix_max_colors[i] = current_max
        
        for i in range(len(ordered_verts)):
            v = ordered_verts[i]
            
            # Pruning
            if current_size + suffix_max_colors[i] <= best_size:
                break
                
            # Branch: Include v
            # New candidates must be neighbors of v in H, and must come after v in the order
            # to avoid duplicate permutations.
            
            # Construct suffix mask for vertices after i
            # Optimization: We can just iterate
            new_cands = 0
            for j in range(i + 1, len(ordered_verts)):
                new_cands |= (1 << ordered_verts[j])
                
            new_cands = new_cands & compatible[v]
            
            expand(new_cands, current_size + 1, current_mask | (1 << v))
            
            # Exclude v from candidates for next iteration
            # We don't actually modify the 'candidates' variable passed in, 
            # but the loop logic implies we are processing vertices one by one.
            # The 'candidates' set for the next iteration of the loop (i+1)
            # should effectively be the original candidates minus v.
            # However, since we are using a list 'ordered_verts' derived from the original candidates,
            # and we are branching on v, the "exclude" branch is handled by the loop progression.
            # We don't need to explicitly remove v from a global state, 
            # because the recursion handles the "include" case, 
            # and the loop continues to the next vertex for the "exclude" case implicitly.
            
    # Start search
    initial_cands = (1 << (N + 1)) - 2
    expand(initial_cands, 0, 0)
    
    # Convert best_mask to list
    result = get_vertices(best_mask)
    return result

# EVOLVE-BLOCK-END
