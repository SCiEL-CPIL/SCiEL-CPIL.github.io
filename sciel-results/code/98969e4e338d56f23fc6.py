"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin (90s limit, leave 35s buffer)
    
    n = 256
    nums = list(range(1, n + 1))
    
    # Build compatibility masks (complement of conflict)
    # compat_masks[i] is a bitset of j such that (i*j) % (i+j) != 0
    compat_masks = [0] * (n + 1)
    conflict_masks = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) != 0:
                compat_masks[i] |= (1 << (j - 1))
                compat_masks[j] |= (1 << (i - 1))
            else:
                conflict_masks[i] |= (1 << (j - 1))
                conflict_masks[j] |= (1 << (i - 1))
    
    # Precompute bit_count for speed (use int.bit_count if available)
    def pc(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
    
    # 1. Strong Heuristic Initialization
    # All odd numbers form a valid set because for distinct odd a,b,
    # a+b is even, ab is odd, so ab % (a+b) != 0.
    S_mask = 0
    for i in range(1, n + 1, 2):
        S_mask |= (1 << (i - 1))
        
    # Greedily add even numbers
    # Sort evens by degree in conflict graph (ascending) to pick "safe" ones
    evens = [i for i in range(2, n + 1, 2)]
    even_degrees = {i: pc(conflict_masks[i]) for i in evens}
    evens.sort(key=lambda x: even_degrees[x])
    
    for i in evens:
        # Check if i is compatible with all in S
        if (conflict_masks[i] & S_mask) == 0:
            S_mask |= (1 << (i - 1))
            
    # Also try randomized greedy from scratch to see if we get better
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        random.seed(trial * 12345 + 67890)
        rand_order = list(range(1, n + 1))
        random.shuffle(rand_order)
        # Sort by conflict degree (ascending) within random groups
        S_mask2 = 0
        for i in rand_order:
            if (conflict_masks[i] & S_mask2) == 0:
                S_mask2 |= (1 << (i - 1))
        if pc(S_mask2) > pc(S_mask):
            S_mask = S_mask2
    
    best_size = pc(S_mask)
    best_set = set()
    tmp = S_mask
    while tmp:
        lsb = tmp & -tmp
        idx = lsb.bit_length() - 1
        best_set.add(idx + 1)
        tmp ^= lsb

    # Helper: Extract vertices from mask
    def get_vertices(mask):
        verts = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            verts.append(idx + 1)
            mask ^= lsb
        return verts

    # Helper: Fast Greedy Coloring Count
    # Uses bitsets for color classes
    def greedy_color_count(P_mask):
        if P_mask == 0:
            return 0
        
        # Order vertices by degree in P (descending) for tighter bound
        verts = get_vertices(P_mask)
        # Sort by degree in P descending for better coloring
        verts.sort(key=lambda v: pc(compat_masks[v] & P_mask), reverse=True)
        
        color_classes = [] # list of bitsets
        
        for v in verts:
            placed = False
            v_bit = 1 << (v - 1)
            for c_idx, class_mask in enumerate(color_classes):
                # v can join class if no edge to any node in class
                if (compat_masks[v] & class_mask) == 0:
                    color_classes[c_idx] |= v_bit
                    placed = True
                    break
            
            if not placed:
                color_classes.append(v_bit)
                
        return len(color_classes)

    # Global state for BK
    sys.setrecursionlimit(10000)
    
    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if P_mask == 0 and X_mask == 0:
            # Maximal clique found
            size = pc(R_mask)
            if size > best_size:
                best_size = size
                best_set = set(get_vertices(R_mask))
            return
        
        # Weak bound: popcount
        if pc(R_mask) + pc(P_mask) <= best_size:
            return
            
        # Coloring Bound
        num_colors = greedy_color_count(P_mask)
        if pc(R_mask) + num_colors <= best_size:
            return
        
        # Pivot Selection: choose u in P that maximizes |N(u) & P|
        # Only need to consider P (not P∪X) for the pivot to get valid candidates
        if P_mask == 0:
            return
            
        # Choose u in P that maximizes |N(u) & P|
        pivot_val = -1
        pivot_bit = 0
        p_iter = P_mask
        while p_iter:
            lsb = p_iter & -p_iter
            u_idx = lsb.bit_length() - 1
            u = u_idx + 1
            count = pc(compat_masks[u] & P_mask)
            if count > pivot_val:
                pivot_val = count
                pivot_bit = lsb
            p_iter ^= lsb
            
        if pivot_bit == 0:
            return
            
        u_idx = pivot_bit.bit_length() - 1
        u = u_idx + 1
        
        # Candidates are P \ N(pivot)
        # Nodes in P that are NOT neighbors of u
        candidates_mask = P_mask & ~compat_masks[u]
        
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            cm ^= lsb
            
            new_P = P_mask & compat_masks[v]
            new_X = X_mask & compat_masks[v]
            new_R = R_mask | lsb
            
            bron_kerbosch(new_R, new_P, new_X)
            
            P_mask ^= lsb
            X_mask |= lsb
            
            if time.time() > deadline:
                return

    # Start BK
    init_P = 0
    for i in range(1, n + 1):
        init_P |= (1 << (i - 1))
        
    bron_kerbosch(0, init_P, 0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
