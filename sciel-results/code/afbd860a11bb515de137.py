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

    # Fast checker for K2,2,2 in a graph defined by adjacency bitmasks
    # K2,2,2 is a complete tripartite graph with parts of size 2
    # We need to find 6 vertices partitioned into 3 pairs (a,b), (c,d), (e,f)
    # such that all cross-edges between different pairs are present,
    # and edges within each pair are absent (non-edges).
    def has_k222(adj, n):
        all_mask = (1 << n) - 1
        
        # Precompute non-neighbor masks
        non_neigh = [all_mask & ~(adj[i] | (1 << i)) for i in range(n)]
        
        # For each pair (u,v) that is a non-edge, compute common neighbors
        # Then for each pair (x,y) in common neighbors that is a non-edge,
        # check if common neighbors of (x,y) intersected with common neighbors of (u,v)
        # contains a non-edge pair.
        
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
                        # A set S contains a non-edge iff it's not a clique
                        # Check: for some z in S, adj[z] & S != S
                        temp = common_xyz
                        is_clique = True
                        while temp:
                            lsb_z = temp & -temp
                            z = lsb_z.bit_length() - 1
                            temp ^= lsb_z
                            if (adj[z] & common_xyz) != common_xyz:
                                is_clique = False
                                break
                        
                        if not is_clique:
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
        if has_k222(adj1, 30):
            return False
        if has_k222(adj0, 30):
            return False
        return True

    # Strategy: Try to find a 30-vertex coloring
    # 1. Structured candidates first
    # 2. Then local search / hill climbing
    # 3. Then random with more samples if time permits
    
    found_30 = False
    valid_mask = None
    
    # Phase 1: Structured candidates
    candidates = []
    candidates.append(0)
    candidates.append((1 << 29) - 1)
    
    # Neighborhoods and non-neighborhoods
    for i in range(29):
        candidates.append(adj1_base[i])
        non_neigh_i = ((1 << 29) - 1) & ~(adj1_base[i] | (1 << i))
        candidates.append(non_neigh_i)
        # Also try adj0 neighborhoods
        candidates.append(adj0_base[i])
        non_neigh0_i = ((1 << 29) - 1) & ~(adj0_base[i] | (1 << i))
        candidates.append(non_neigh0_i)
    
    # Symmetric differences of neighborhoods
    for i in range(29):
        for j in range(i+1, 29):
            candidates.append(adj1_base[i] ^ adj1_base[j])
            candidates.append(adj1_base[i] & adj1_base[j])
            candidates.append(adj1_base[i] | adj1_base[j])
    
    # Deduplicate
    candidates = list(set(candidates))
    
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
        best_mask = None
        best_score = -1
        
        for restart in range(50):
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
            
            if is_valid_30(mask):
                found_30 = True
                valid_mask = mask
                break
    
    # Phase 3: More random if time permits
    if not found_30 and time.time() < deadline - 2:
        random.seed(123)
        for _ in range(5000):
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
