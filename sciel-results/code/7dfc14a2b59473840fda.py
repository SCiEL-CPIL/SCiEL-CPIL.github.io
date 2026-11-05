import time
import random

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    def generate_paley(n):
        squares = set((x*x) % n for x in range(n))
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = (j - i) % n
                colors.append(1 if d in squares else 0)
        return colors

    paley_29_colors = generate_paley(29)
    
    # Build adjacency bitmasks for Paley(29)
    n_base = 29
    adj1_base = [0] * n_base
    adj0_base = [0] * n_base
    idx = 0
    for i in range(n_base):
        for j in range(i+1, n_base):
            if paley_29_colors[idx] == 1:
                adj1_base[i] |= (1 << j)
                adj1_base[j] |= (1 << i)
            else:
                adj0_base[i] |= (1 << j)
                adj0_base[j] |= (1 << i)
            idx += 1

    # Precompute non-neighbor masks for base graphs
    all_mask_29 = (1 << 29) - 1
    non_neigh1_base = [all_mask_29 & ~(adj1_base[i] | (1 << i)) for i in range(29)]
    non_neigh0_base = [all_mask_29 & ~(adj0_base[i] | (1 << i)) for i in range(29)]

    # Fast checker for K2,2,2 in a graph defined by adjacency bitmasks
    # K2,2,2 is a complete tripartite graph with parts of size 2
    # We need to find 6 vertices partitioned into 3 pairs (a,b), (c,d), (e,f)
    # such that all cross-edges between different pairs are present,
    # and edges within each pair are absent (non-edges).
    def has_k222(adj, n, non_neigh=None):
        if non_neigh is None:
            all_mask = (1 << n) - 1
            non_neigh = [all_mask & ~(adj[i] | (1 << i)) for i in range(n)]
        
        # Helper: check if a set (bitmask) contains a non-edge pair
        def contains_non_edge(S, adj, non_neigh):
            if S.bit_count() < 2:
                return False
            temp = S
            while temp:
                lsb_z = temp & -temp
                z = lsb_z.bit_length() - 1
                temp ^= lsb_z
                if (adj[z] & S) != S:
                    return True
            return False
        
        for u in range(n):
            # Iterate over non-neighbors v of u with v > u
            v_mask = non_neigh[u] & ((1 << (n - u - 1)) - 1) << (u + 1)
            while v_mask:
                lsb_v = v_mask & -v_mask
                v = lsb_v.bit_length() - 1
                v_mask ^= lsb_v
                
                # Common neighbors of u and v
                common_uv = adj[u] & adj[v]
                
                # Iterate over non-edge pairs (x,y) in common_uv
                # x and y must both be in common_uv and (x,y) must be a non-edge
                x_mask = common_uv
                while x_mask:
                    lsb_x = x_mask & -x_mask
                    x = lsb_x.bit_length() - 1
                    x_mask ^= lsb_x
                    
                    # y must be in common_uv, y > x, and (x,y) is a non-edge
                    y_mask = common_uv & non_neigh[x] & ((1 << (n - x - 1)) - 1) << (x + 1)
                    while y_mask:
                        lsb_y = y_mask & -y_mask
                        y = lsb_y.bit_length() - 1
                        y_mask ^= lsb_y
                        
                        # Now we have non-edge pairs (u,v) and (x,y)
                        # with x,y in N(u) ∩ N(v)
                        # Check if N(x) ∩ N(y) ∩ N(u) ∩ N(v) contains a non-edge
                        common_xyz = adj[x] & adj[y] & common_uv
                        
                        # Check if common_xyz contains a non-edge pair
                        if contains_non_edge(common_xyz, adj, non_neigh):
                            return True
        
        return False

    def build_adj_30(mask):
        """Build adj1 and adj0 for n=30 given mask for vertex 29's edges."""
        adj1 = [0] * 30
        adj0 = [0] * 30
        
        # Copy base
        for i in range(29):
            adj1[i] = adj1_base[i]
            adj0[i] = adj0_base[i]
        
        # Set edges for vertex 29
        for i in range(29):
            if (mask >> i) & 1:
                adj1[29] |= (1 << i)
                adj1[i] |= (1 << 29)
            else:
                adj0[29] |= (1 << i)
                adj0[i] |= (1 << 29)
        
        return adj1, adj0

    def is_valid_30(mask):
        adj1, adj0 = build_adj_30(mask)
        # Build non_neigh for n=30
        all_mask_30 = (1 << 30) - 1
        non_neigh1_30 = [all_mask_30 & ~(adj1[i] | (1 << i)) for i in range(30)]
        non_neigh0_30 = [all_mask_30 & ~(adj0[i] | (1 << i)) for i in range(30)]
        if has_k222(adj1, 30, non_neigh1_30):
            return False
        if has_k222(adj0, 30, non_neigh0_30):
            return False
        return True

    # Strategy: Try to find a 30-vertex coloring
    # 1. Structured candidates first
    # 2. Then local search / hill climbing
    # 3. Then random with more samples if time permits
    
    found_30 = False
    valid_mask = None
    
    # Phase 1: Structured candidates
    candidates = set()
    candidates.add(0)
    candidates.add((1 << 29) - 1)
    
    # Neighborhoods and non-neighborhoods
    for i in range(29):
        candidates.add(adj1_base[i])
        candidates.add(non_neigh1_base[i])
        candidates.add(adj0_base[i])
        candidates.add(non_neigh0_base[i])
    
    # Symmetric differences, intersections, unions of neighborhoods
    for i in range(29):
        for j in range(i+1, 29):
            candidates.add(adj1_base[i] ^ adj1_base[j])
            candidates.add(adj1_base[i] & adj1_base[j])
            candidates.add(adj1_base[i] | adj1_base[j])
            candidates.add(adj0_base[i] ^ adj0_base[j])
            candidates.add(adj0_base[i] & adj0_base[j])
            candidates.add(adj0_base[i] | adj0_base[j])
            # Mixed
            candidates.add(adj1_base[i] ^ adj0_base[j])
            candidates.add(adj1_base[i] & adj0_base[j])
            candidates.add(adj1_base[i] | adj0_base[j])
    
    # Triple combinations
    for i in range(29):
        for j in range(i+1, 29):
            for k in range(j+1, 29):
                candidates.add(adj1_base[i] ^ adj1_base[j] ^ adj1_base[k])
                candidates.add(adj1_base[i] & adj1_base[j] & adj1_base[k])
                candidates.add(adj1_base[i] | adj1_base[j] | adj1_base[k])
                candidates.add(adj0_base[i] ^ adj0_base[j] ^ adj0_base[k])
                candidates.add(adj0_base[i] & adj0_base[j] & adj0_base[k])
                candidates.add(adj0_base[i] | adj0_base[j] | adj0_base[k])
    
    # Convert to list for iteration
    candidates = list(candidates)
    
    for mask in candidates:
        if time.time() > deadline:
            break
        if is_valid_30(mask):
            found_30 = True
            valid_mask = mask
            break
    
    # Phase 2: Local search - start from random and flip bits
    if not found_30 and time.time() < deadline - 5:
        random.seed(42)
        
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            
            mask = random.getrandbits(29)
            
            # Evaluate and try to improve by flipping single bits
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for bit in range(29):
                    new_mask = mask ^ (1 << bit)
                    if is_valid_30(new_mask):
                        mask = new_mask
                        improved = True
                        break
                # Also try double flips
                if not improved:
                    for bit1 in range(29):
                        for bit2 in range(bit1+1, 29):
                            new_mask = mask ^ (1 << bit1) ^ (1 << bit2)
                            if is_valid_30(new_mask):
                                mask = new_mask
                                improved = True
                                break
                        if improved:
                            break
                # Try triple flips if still stuck
                if not improved:
                    for bit1 in range(29):
                        for bit2 in range(bit1+1, 29):
                            for bit3 in range(bit2+1, 29):
                                new_mask = mask ^ (1 << bit1) ^ (1 << bit2) ^ (1 << bit3)
                                if is_valid_30(new_mask):
                                    mask = new_mask
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
            
            if is_valid_30(mask):
                found_30 = True
                valid_mask = mask
                break
    
    # Phase 3: More random if time permits
    if not found_30 and time.time() < deadline - 2:
        random.seed(123)
        for _ in range(10000):
            if time.time() > deadline - 2:
                break
            mask = random.getrandbits(29)
            if is_valid_30(mask):
                found_30 = True
                valid_mask = mask
                break
    
    if found_30:
        # Construct the full color list for n=30
        colors = []
        for i in range(30):
            for j in range(i + 1, 30):
                if i == 29:
                    colors.append((valid_mask >> j) & 1)
                elif j == 29:
                    colors.append((valid_mask >> i) & 1)
                else:
                    if (adj1_base[i] >> j) & 1:
                        colors.append(1)
                    else:
                        colors.append(0)
        return {'n': 30, 'colors': colors}
        
    # If 30 not found, return 29
    return {'n': 29, 'colors': paley_29_colors}
