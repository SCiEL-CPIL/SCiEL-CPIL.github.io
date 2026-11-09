import time
import random

def construct():
    """
    Constructs a graph with no K5 and no I5, maximizing the number of vertices.
    Returns (n, edges).
    """
    start_time = time.time()
    DEADLINE = 280  # Safety margin before 300s limit
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency as list of bitmasks (Python ints)
    # adj[i] is a bitmask where bit j is 1 if (i, j) is an edge
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Helper to get current edges list
    def get_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only count edges where i < j
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                mask ^= lsb
        return edges

    # Precompute 4-cliques and 4-independent sets
    # A 4-clique is a set of 4 vertices all connected to each other.
    # An 4-independent set is a set of 4 vertices with no edges between them.
    
    def find_4_cliques():
        cliques = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)): continue
                for k in range(j+1, n):
                    if not (adj[i] & (1 << k)): continue
                    if not (adj[j] & (1 << k)): continue
                    for l in range(k+1, n):
                        if not (adj[i] & (1 << l)): continue
                        if not (adj[j] & (1 << l)): continue
                        if not (adj[k] & (1 << l)): continue
                        # Found a K4: {i, j, k, l}
                        # Represent as a bitmask
                        mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                        cliques.append(mask)
        return cliques

    def find_4_independent_sets():
        indep_sets = []
        # Independent set: no edges between vertices.
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j): continue
                for k in range(j+1, n):
                    if adj[i] & (1 << k): continue
                    if adj[j] & (1 << k): continue
                    for l in range(k+1, n):
                        if adj[i] & (1 << l): continue
                        if adj[j] & (1 << l): continue
                        if adj[k] & (1 << l): continue
                        # Found an I4: {i, j, k, l}
                        mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                        indep_sets.append(mask)
        return indep_sets

    k4_list = find_4_cliques()
    i4_list = find_4_independent_sets()
    
    # Index K4s and I4s by vertex for fast incremental updates
    # k4_index[v] = list of indices in k4_list that contain v
    # i4_index[v] = list of indices in i4_list that contain v
    def build_indices(k4s, i4s, current_n):
        k4_idx = [[] for _ in range(current_n)]
        i4_idx = [[] for _ in range(current_n)]
        for idx, mask in enumerate(k4s):
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                k4_idx[v].append(idx)
                m ^= lsb
        for idx, mask in enumerate(i4s):
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                i4_idx[v].append(idx)
                m ^= lsb
        return k4_idx, i4_idx

    k4_index, i4_index = build_indices(k4_list, i4_list, n)
    
    best_n = n
    best_edges = get_edges()
    
    # Local Search function to find a valid mask for a new vertex
    def find_valid_mask(current_n, k4s, i4s, k4_idx, i4_idx):
        # k4s, i4s are lists of bitmasks
        # k4_idx, i4_idx are lists of lists of indices
        # We want to find a mask M of size current_n such that:
        # 1. No K4 is subset of M
        # 2. No I4 is subset of ~M
        
        # Initial random mask
        # Heuristic: Start with a mask that mimics a random existing vertex?
        # Or just random.
        # Let's try a few different starting points.
        
        max_iters = 5000
        best_cost = float('inf')
        best_mask = 0
        
        for restart in range(20): # 20 restarts
            # Initialize mask
            # Strategy: Random, but biased towards 0.5 density
            mask = 0
            for i in range(current_n):
                if random.random() < 0.5:
                    mask |= (1 << i)
            
            # Calculate initial cost
            cost = 0
            full_mask = (1 << current_n) - 1
            
            # Count violations
            # K4 violation: k4 & mask == k4
            for k4 in k4s:
                if (k4 & mask) == k4:
                    cost += 1
            # I4 violation: i4 & (full_mask ^ mask) == i4
            # Which is: (i4 & mask) == 0
            for i4 in i4s:
                if (i4 & mask) == 0:
                    cost += 1
            
            if cost == 0:
                return mask
            
            current_cost = cost
            current_mask = mask
            
            for _ in range(max_iters):
                if time.time() - start_time > DEADLINE:
                    return None # Timeout
                
                # Pick a random vertex to flip
                v = random.randint(0, current_n - 1)
                bit = 1 << v
                
                # Calculate delta cost
                # If we flip v, we need to check K4s and I4s containing v.
                delta = 0
                
                # Check K4s containing v
                # Before flip:
                # If bit v was 0 in mask:
                #   K4 is satisfied if (k4 & mask) == k4.
                #   Since v in k4, (k4 & mask) has bit v as 0, so it can never be == k4.
                #   So flipping 0->1 might satisfy a K4 if the other 3 bits are already set.
                # If bit v was 1 in mask:
                #   K4 might be satisfied. Flipping to 0 will break it.
                
                # Let's just recalculate the status of K4s containing v.
                # This is safer and still fast enough for n~40.
                
                old_v_bit = 1 if (current_mask & bit) else 0
                new_v_bit = 1 - old_v_bit
                
                # For each K4 containing v:
                for idx in k4_idx[v]:
                    k4 = k4s[idx]
                    # Check if K4 was satisfied before
                    # k4 & current_mask == k4 ?
                    # Since v is in k4, we can check if (k4 & (current_mask ^ bit)) == (k4 ^ bit)
                    # i.e., are the other 3 bits set?
                    others = k4 ^ bit
                    if (others & current_mask) == others:
                        # The other 3 bits are set.
                        if old_v_bit == 0:
                            # Flipping to 1 makes it satisfied. Cost increases by 1.
                            delta += 1
                        else:
                            # Flipping to 0 makes it unsatisfied. Cost decreases by 1.
                            delta -= 1
                    # If others not set, status doesn't change (it was unsatisfied, stays unsatisfied)
                
                # For each I4 containing v:
                # I4 satisfied if (i4 & mask) == 0.
                # i.e., all bits in i4 are 0 in mask.
                for idx in i4_idx[v]:
                    i4 = i4s[idx]
                    others = i4 ^ bit
                    # Check if other 3 bits are 0 in mask
                    if (others & current_mask) == 0:
                        # The other 3 bits are 0.
                        if old_v_bit == 1:
                            # v was 1, so I4 was NOT satisfied (because v bit was 1).
                            # Flipping to 0 makes it satisfied. Cost increases by 1.
                            delta += 1
                        else:
                            # v was 0, so I4 WAS satisfied (all 4 bits 0).
                            # Flipping to 1 makes it unsatisfied. Cost decreases by 1.
                            delta -= 1
                
                new_cost = current_cost + delta
                
                # Accept if better, or if equal (to allow moving on plateaus)
                if new_cost <= current_cost:
                    current_mask ^= bit
                    current_cost = new_cost
                    
                    if current_cost == 0:
                        return current_mask
                else:
                    # Reject move
                    pass
            
            # End of iterations for this restart
            if current_cost < best_cost:
                best_cost = current_cost
                best_mask = current_mask
        
        # If we didn't find 0 cost, return None (or best_mask if we want to try to fix it later, but usually None means fail)
        return None

    # Extension loop
    target_n = 45
    max_restarts_per_vertex = 50
    
    while n < target_n and time.time() - start_time < DEADLINE:
        added = False
        
        # Try to find a valid mask for vertex n
        for attempt in range(max_restarts_per_vertex):
            if time.time() - start_time > DEADLINE:
                break
            
            mask = find_valid_mask(n, k4_list, i4_list, k4_index, i4_index)
            
            if mask is not None:
                # Accept new vertex
                adj.append(mask)
                # Update adjacency for existing vertices
                for i in range(n):
                    if mask & (1 << i):
                        adj[i] |= (1 << n)
                
                # Update lists of K4 and I4
                # New K4s must include the new vertex n.
                # New K4s are formed by n + K3 in neighborhood of n.
                neigh_list = []
                for i in range(n):
                    if mask & (1 << i):
                        neigh_list.append(i)
                        
                new_k4s = []
                for a_idx in range(len(neigh_list)):
                    a = neigh_list[a_idx]
                    for b_idx in range(a_idx+1, len(neigh_list)):
                        b = neigh_list[b_idx]
                        if not (adj[a] & (1 << b)): continue
                        for c_idx in range(b_idx+1, len(neigh_list)):
                            c = neigh_list[c_idx]
                            if not (adj[a] & (1 << c)): continue
                            if not (adj[b] & (1 << c)): continue
                            new_k4s.append((1 << a) | (1 << b) | (1 << c) | (1 << n))
                
                # New I4s formed by n + I3 in non-neighborhood of n.
                non_neigh_list = []
                for i in range(n):
                    if not (mask & (1 << i)):
                        non_neigh_list.append(i)
                        
                new_i4s = []
                for a_idx in range(len(non_neigh_list)):
                    a = non_neigh_list[a_idx]
                    for b_idx in range(a_idx+1, len(non_neigh_list)):
                        b = non_neigh_list[b_idx]
                        if adj[a] & (1 << b): continue
                        for c_idx in range(b_idx+1, len(non_neigh_list)):
                            c = non_neigh_list[c_idx]
                            if adj[a] & (1 << c): continue
                            if adj[b] & (1 << c): continue
                            new_i4s.append((1 << a) | (1 << b) | (1 << c) | (1 << n))
                            
                k4_list.extend(new_k4s)
                i4_list.extend(new_i4s)
                
                # Rebuild indices for the new vertex and update existing indices
                # This is expensive if done naively. 
                # k4_index and i4_index need to be extended for the new vertex n.
                # Also, existing vertices' indices need to be updated if new K4s/I4s involve them.
                
                # Efficient update:
                # 1. Extend k4_index and i4_index with empty lists for vertex n.
                k4_index.append([])
                i4_index.append([])
                
                # 2. Add new K4s to the lists.
                # For each new K4, add its index to k4_index[v] for each v in K4.
                start_idx = len(k4_list) - len(new_k4s)
                for i, mask_k4 in enumerate(new_k4s):
                    idx = start_idx + i
                    m = mask_k4
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        k4_index[v].append(idx)
                        m ^= lsb
                        
                # 3. Add new I4s to the lists.
                start_idx = len(i4_list) - len(new_i4s)
                for i, mask_i4 in enumerate(new_i4s):
                    idx = start_idx + i
                    m = mask_i4
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        i4_index[v].append(idx)
                        m ^= lsb
                        
                n += 1
                added = True
                best_n = n
                best_edges = get_edges()
                break # Move to next vertex
                
        if not added:
            # Could not add vertex n
            break

    return best_n, best_edges
