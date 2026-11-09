import time

N = 12
P = 27720

def _precompute_masks():
    masks = [0] * P
    for r in range(P):
        m = 0
        for i in range(1, 13):
            if r % i == 0:
                m |= (1 << (i - 1))
        masks[r] = m
    return masks

MASKS = _precompute_masks()

def _compute_f(m):
    # adj_mask[i] is a bitmask of positions j (0-indexed) where (m+1+j) is divisible by i+1
    adj_mask = [0] * N
    # match_to_pos[i] = position j assigned to i, or -1
    match_to_pos = [-1] * N
    # pos_matched_by[j] = i assigned to position j, or -1
    # We only need up to max L. Let's assume max L is 144.
    # We will grow L.
    
    # We need to be able to reset or manage state.
    # Since we return as soon as we find a match, we can just loop L.
    
    # To check if a matching of size 12 exists in the current graph defined by adj_mask:
    # We maintain the current matching.
    # When we add edges, we try to augment.
    
    pos_matched_by = {} # j -> i
    
    def try_augment(i, visited):
        # visited is a bitmask of positions
        candidates = adj_mask[i] & ~visited
        while candidates:
            # Get lowest set bit
            lsb = candidates & -candidates
            j = lsb.bit_length() - 1
            candidates ^= lsb
            visited |= lsb
            
            if j not in pos_matched_by:
                # Free position
                pos_matched_by[j] = i
                match_to_pos[i] = j
                return True
            else:
                # Position j is matched to i2
                i2 = pos_matched_by[j]
                if try_augment(i2, visited):
                    pos_matched_by[j] = i
                    match_to_pos[i] = j
                    return True
        return False

    matched_count = 0
    # We need to track which i are matched.
    # Initially all unmatched.
    unmatched = list(range(N))
    
    for L in range(12, 145):
        # Add position j = L-1
        j = L - 1
        r = (m + L) % P
        mask = MASKS[r]
        
        # Update adj_mask for all i that divide m+L
        for i in range(N):
            if mask & (1 << i):
                adj_mask[i] |= (1 << j)
        
        # Try to augment for all currently unmatched nodes
        # Note: It's possible that adding an edge allows an already matched node to be rematched 
        # to free up space for another? 
        # Standard augmenting path logic: 
        # If we have a matching M, and we add edges, M is still a valid matching.
        # The size of maximum matching can increase by at most 1 for each added edge? 
        # Actually, we just need to check if we can increase the size.
        # We try to find an augmenting path for each free node.
        
        new_unmatched = []
        for i in unmatched:
            visited = 0
            if try_augment(i, visited):
                matched_count += 1
                # i is now matched
            else:
                new_unmatched.append(i)
        
        unmatched = new_unmatched
        
        if matched_count == N:
            return L
            
    return 144

def construct():
    best_m = 0
    best_f = _compute_f(0)
    
    # Search over all residues
    # 27720 iterations.
    # Each iteration is fast.
    # Let's check time.
    start_time = time.time()
    deadline = start_time + 80
    
    for m in range(P):
        if time.time() > deadline:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return int(best_m)
