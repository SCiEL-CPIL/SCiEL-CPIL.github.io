"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 256
    nums = list(range(1, n + 1))
    
    # Build compatibility masks (complement of conflict)
    # compat_masks[i] is a bitset of j such that (i*j) % (i+j) != 0
    compat_masks = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) != 0:
                compat_masks[i] |= (1 << (j - 1))
                compat_masks[j] |= (1 << (i - 1))
                
    # Greedy Initial Solution (Maximal Independent Set)
    # Sort by degree in conflict graph (ascending) -> High degree in compat graph
    # Actually, for MIS, we usually pick low degree in conflict graph.
    # Let's just do a greedy pass on the compatibility graph to build a clique.
    # Or simpler: Greedy MIS on conflict graph.
    
    # Let's use the previous greedy logic to get a good initial lower bound
    conflict_masks = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                conflict_masks[i] |= (1 << (j - 1))
                conflict_masks[j] |= (1 << (i - 1))
                
    degrees = [conflict_masks[i].bit_count() for i in range(n + 1)]
    sorted_nums = sorted(nums, key=lambda x: degrees[x])
    
    S_mask = 0
    S_list = []
    for i in sorted_nums:
        if (conflict_masks[i] & S_mask) == 0:
            S_mask |= (1 << (i - 1))
            S_list.append(i)
            
    best_size = len(S_list)
    best_set = set(S_list)
    
    # Bron-Kerbosch with Coloring for Maximum Clique in Compatibility Graph
    # Vertices are 1..n. Bit index is v-1.
    
    # Helper: Extract vertices from mask
    def get_vertices(mask):
        verts = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            verts.append(idx + 1)
            mask ^= lsb
        return verts

    # Helper: Greedy Coloring of a set of vertices
    # Returns number of colors.
    # Uses degree ordering in P for better bound.
    def greedy_color_count(P_mask):
        if P_mask == 0:
            return 0
        
        # Get vertices in P
        verts = []
        pm = P_mask
        while pm:
            lsb = pm & -pm
            idx = lsb.bit_length() - 1
            verts.append(idx + 1)
            pm ^= lsb
            
        # Sort by degree in P (descending) for tighter bound
        verts.sort(key=lambda v: (compat_masks[v] & P_mask).bit_count(), reverse=True)
        
        color_classes = [] # list of bitsets
        
        for v in verts:
            placed = False
            for c_idx, class_mask in enumerate(color_classes):
                # v can join class c if it has no neighbors in class c
                if (compat_masks[v] & class_mask) == 0:
                    color_classes[c_idx] |= (1 << (v - 1))
                    placed = True
                    break
            
            if not placed:
                color_classes.append(1 << (v - 1))
                
        return len(color_classes)

    # Global state for BK
    import sys
    sys.setrecursionlimit(10000)
    
    def bron_kerbosch(R_mask, P_mask, X_mask, depth):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if P_mask == 0 and X_mask == 0:
            # Maximal clique found
            size = R_mask.bit_count()
            if size > best_size:
                best_size = size
                best_set = set(get_vertices(R_mask))
            return
        
        # Coloring Bound
        num_colors = greedy_color_count(P_mask)
        if R_mask.bit_count() + num_colors <= best_size:
            return
        
        # Pivot Selection
        # Choose u in P | X that maximizes |N(u) & P|
        PX_mask = P_mask | X_mask
        if PX_mask == 0:
            return
            
        # Iterate PX to find pivot
        # This can be slow if we do it naively.
        # For N=256, it's okay.
        
        pivot_val = 0
        pivot_mask = 0
        px_iter = PX_mask
        while px_iter:
            lsb = px_iter & -px_iter
            u_idx = lsb.bit_length() - 1
            u = u_idx + 1
            count = (compat_masks[u] & P_mask).bit_count()
            if count > pivot_val:
                pivot_val = count
                pivot_mask = lsb
            px_iter ^= lsb
            
        if pivot_mask == 0:
            return
            
        # Candidates are P \ N(pivot)
        # We need to iterate over candidates in P that are NOT neighbors of pivot
        # Let u be the pivot vertex
        u_idx = pivot_mask.bit_length() - 1
        u = u_idx + 1
        
        candidates_mask = P_mask & ~compat_masks[u]
        
        # Iterate candidates
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            cm ^= lsb
            
            # Recurse
            # New P: P & N(v)
            # New X: X & N(v)
            # New R: R | {v}
            new_P = P_mask & compat_masks[v]
            new_X = X_mask & compat_masks[v]
            new_R = R_mask | lsb
            
            bron_kerbosch(new_R, new_P, new_X, depth + 1)
            
            # Move v from P to X
            P_mask ^= lsb
            X_mask |= lsb
            
            if time.time() > deadline:
                return

    # Start BK
    # P is all vertices, X is empty, R is empty
    # But we have an initial solution. We can try to extend it?
    # Standard BK starts from scratch.
    # To benefit from the initial solution, we can just let BK find the max.
    # The initial solution helps prune if we pass it as a lower bound.
    
    # We need to be careful with recursion depth and time.
    # Let's run it.
    
    # Initial P is all vertices
    init_P = 0
    for i in range(1, n + 1):
        init_P |= (1 << (i - 1))
        
    bron_kerbosch(0, init_P, 0, 0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
