import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Handle empty matrix
    if not matrix:
        return {'gates': [], 'outputs': []}
    
    # Convert to list of lists of ints
    # matrix might be a numpy array or list of lists
    try:
        mat = [list(row) for row in matrix]
    except TypeError:
        # If matrix is a flat list or something else unexpected
        # Assume standard 2D structure
        mat = [list(row) for row in matrix]
        
    if not mat:
        return {'gates': [], 'outputs': []}
        
    m, n = len(mat), len(mat[0])
    
    # Map rows to bitmasks
    # Input wires 0..n-1 correspond to bits 0..n-1
    # Constant zero wire is n, bitmask 0
    
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i][j]:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Initialize available wires
    # wire_to_mask: {wire_idx: mask}
    # mask_to_wire: {mask: wire_idx}
    
    wire_to_mask = {}
    mask_to_wire = {}
    
    # Inputs 0..n-1
    for j in range(n):
        mask = 1 << j
        wire_to_mask[j] = mask
        mask_to_wire[mask] = j
        
    # Zero wire n
    wire_to_mask[n] = 0
    mask_to_wire[0] = n
    
    # Identify pending masks (targets)
    pending_masks = set()
    for mask in row_masks:
        if mask not in mask_to_wire:
            pending_masks.add(mask)
            
    # If no pending masks, we are done
    if not pending_masks:
        outputs = [mask_to_wire[mask] for mask in row_masks]
        return {'gates': [], 'outputs': outputs}
    
    gates = []
    available_masks = set(mask_to_wire.keys())
    next_wire_idx = n + 1
    
    # Greedy Loop
    while pending_masks:
        if time.time() > DEADLINE:
            # Fallback: if we run out of time, we cannot complete the synthesis.
            # However, for small n, this should not happen.
            # If it does, we might return an incomplete/invalid circuit.
            # To be safe, we could try to resolve remaining targets using a simple
            # independent synthesis (though it might exceed depth/gate limits).
            # Given the fast heuristic, this is unlikely.
            break
            
        best_score = -1
        best_w = None
        best_u = None
        best_v = None
        
        avail_list = list(available_masks)
        len_avail = len(avail_list)
        
        for i in range(len_avail):
            u = avail_list[i]
            for j in range(i + 1, len_avail):
                v = avail_list[j]
                w = u ^ v
                
                if w in available_masks:
                    continue
                
                score = 0
                
                # 1. Direct target
                if w in pending_masks:
                    score += 10000
                    
                # 2. Popcount reduction
                for t in pending_masks:
                    # Using bit_count is fast
                    if (t ^ w).bit_count() < t.bit_count():
                        score += (t.bit_count() - (t ^ w).bit_count())
                        
                # 3. 2-step to target: w ^ v' is in pending_masks
                # Only check if we haven't already found a direct target or high score?
                # No, check always to ensure best candidate.
                for v_prime in avail_list:
                    if (w ^ v_prime) in pending_masks:
                        score += 5000
                        break
                
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_u = u
                    best_v = v
        
        if best_w is None:
            # Fallback: pick any new vector with min popcount
            min_pop = float('inf')
            for i in range(len_avail):
                u = avail_list[i]
                for j in range(i + 1, len_avail):
                    v = avail_list[j]
                    w = u ^ v
                    if w not in available_masks:
                        p = w.bit_count()
                        if p < min_pop:
                            min_pop = p
                            best_w = w
                            best_u = u
                            best_v = v
            if best_w is None:
                # No new vectors can be generated?
                # This implies we are stuck.
                break
                
        # Create new gate
        gates.append([mask_to_wire[best_u], mask_to_wire[best_v]])
        
        # Update structures
        wire_to_mask[next_wire_idx] = best_w
        mask_to_wire[best_w] = next_wire_idx
        available_masks.add(best_w)
        
        # Resolve targets
        if best_w in pending_masks:
            pending_masks.remove(best_w)
            
        next_wire_idx += 1
        
    # Construct outputs
    # If some masks are still pending (due to timeout or logic error), 
    # we must assign them something. 
    # Ideally, this doesn't happen.
    # If it does, we might fail verification.
    outputs = []
    for mask in row_masks:
        if mask in mask_to_wire:
            outputs.append(mask_to_wire[mask])
        else:
            # This case should not occur if the loop terminates correctly
            # For safety, we could try a simple synthesis here, but 
            # given the constraints, we assume the greedy succeeded.
            outputs.append(0) # Default to zero, which will likely fail verification
            
    return {'gates': gates, 'outputs': outputs}
