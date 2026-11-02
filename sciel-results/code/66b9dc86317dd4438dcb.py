import time
import numpy as np
from itertools import combinations

D = 11

def construct():
    deadline = time.time() + 110  # 110s safety margin

    # Type A vectors: (±2, 0, ..., 0)
    type_a = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            type_a.append(v)
    
    # Type B vectors: four ±1 entries
    # Generate all unique Type B vectors
    type_b_vectors = []
    seen = set()
    for combo in combinations(range(D), 4):
        for signs in range(16):
            v = [0] * D
            vals = []
            for k in range(4):
                val = 1 if (signs >> k) & 1 else -1
                vals.append(val)
                v[combo[k]] = val
            
            # Canonical form to avoid duplicates? 
            # Actually, signs are distinct vectors. 
            # (1,1,1,1) is different from (-1,-1,-1,-1) etc.
            # But (1,1,1,1) at indices (0,1,2,3) is unique.
            # We just need to ensure we don't add the same vector twice.
            # Since we iterate all combos and all sign patterns, and indices are fixed,
            # each tuple is unique.
            type_b_vectors.append(tuple(v))
            
    n_b = len(type_b_vectors)
    
    # Precompute conflict bitsets
    # conflict[i] is a bitmask of all j such that dot(v_i, v_j) > 2
    # Note: dot(v_i, v_i) = 4, so i is in conflict[i]. We handle this by checking j != i or masking out self.
    # Actually, for independent set, we just need to know which others conflict.
    
    # Convert to numpy for fast dot product calculation
    # But N=5280, N^2 = 27M. Numpy is fast enough.
    # Let's do it in chunks to avoid memory blowup if needed, but 27M ints is ~200MB, might be tight.
    # Better: Loop i, compute dot products with all j, build bitmask.
    
    conflict_masks = [0] * n_b
    vb_np = np.array(type_b_vectors, dtype=np.int8)
    
    # Optimize: Use NumPy for batch dot products
    # Calculate full dot product matrix? 5280x5280 float32 is ~110MB. Acceptable.
    # dot_matrix = vb_np @ vb_np.T
    # conflict = dot_matrix > 2
    # Then convert to bitsets.
    
    # Let's try to be memory efficient or just use the matrix if it fits.
    # 5280 * 5280 * 4 bytes = 111 MB. Safe.
    
    start_time = time.time()
    if time.time() - start_time > 10: # Fallback if numpy is slow or memory error? Unlikely.
         pass # Just proceed.
         
    try:
        dot_matrix = vb_np @ vb_np.T
        # Conflict if dot > 2. 
        # Note: We want to exclude self. But self has dot=4.
        # In the bitmask, we will include self. When checking compatibility, 
        # we will mask out the self-bit or just ensure we don't add self again.
        # For local search: 
        # Can we add j if (selected & conflict[j]) == 0? 
        # If j is not selected, and j is not in conflict with any selected, it's valid.
        # conflict[j] contains j itself. If j is not selected, (selected & conflict[j]) won't have bit j set anyway.
        # So it works.
        
        conflict_bool = dot_matrix > 2
        
        # Convert boolean matrix to list of Python ints (bitmasks)
        # This is the expensive part: 5280 iterations of packing bits.
        # We can do this column by column or row by row.
        # Using numpy's packbits? Or just a loop.
        # Loop is 5280 * 5280/8 operations ~ 3.5M operations. Fast.
        
        for i in range(n_b):
            row = conflict_bool[i]
            # Pack bits into an integer
            # np.packbits returns bytes. int.from_bytes is fast.
            # Ensure little-endian or big-endian consistency.
            # int.from_bytes(bytes, 'little')
            # Bit 0 corresponds to row[0] (j=0).
            packed = np.packbits(row)
            conflict_masks[i] = int.from_bytes(packed, byteorder='little')
            
    except Exception as e:
        # Fallback to pure Python if numpy fails (unlikely)
        # This is just a safety net.
        conflict_masks = [0] * n_b
        for i in range(n_b):
            mask = 0
            vi = type_b_vectors[i]
            for j in range(n_b):
                vj = type_b_vectors[j]
                dot = sum(vi[k]*vj[k] for k in range(D))
                if dot > 2:
                    mask |= (1 << j)
            conflict_masks[i] = mask
            if time.time() > deadline - 5:
                break
                
    if time.time() > deadline - 5:
        # Return just Type A if we ran out of time precomputing
        return np.array(type_a, dtype=np.int64)

    # Local Search with Bitsets
    best_selected = 0
    best_count = 0
    
    # Greedy Initialization
    indices = np.arange(n_b)
    np.random.shuffle(indices)
    
    selected = 0
    for idx in indices:
        if not (selected & conflict_masks[idx]):
            selected |= (1 << idx)
            
    best_selected = selected
    best_count = bin(selected).count('1')
    
    # Local Search Loop
    # We will iterate until no improvement or time limit.
    # To speed up, we can maintain a list of unselected indices? 
    # Or just iterate all indices.
    # With bitsets, checking if a swap is possible is fast.
    
    # Strategy: 
    # 1. Try to add any unselected vector that is compatible. (Pure addition)
    # 2. If no addition, try 1-swap: remove one selected, add one unselected.
    
    # For addition:
    # candidates = ~selected & ~ (OR of conflicts of selected?) No, that's not right.
    # A vector j is addable if (selected & conflict_masks[j]) == 0.
    
    # Let's do a simple local search:
    # Randomly pick a selected vector to remove? Or iterate?
    # Iterating all selected vectors and checking all unselected is O(N^2) per pass.
    # But with bitsets, "check all unselected" can be optimized?
    # Not directly in a single bit op because conflict_masks vary.
    
    # However, N=5280 is small. A pass takes ~5280*5280/8 bit ops? 
    # Actually, checking `selected & conflict_masks[j]` is 1 op.
    # So a full pass is O(N).
    # We can do many passes.
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try 1-swap
        # We need to find a pair (i in selected, j not in selected) such that 
        # j is compatible with (selected \ {i}).
        # Condition: (selected \ {i}) & conflict_masks[j] == 0
        # Which is: ((selected & ~(1<<i)) & conflict_masks[j]) == 0
        
        # To make this efficient, we can iterate over selected indices i.
        # For each i, we want to find a j.
        # If we iterate j for each i, it's O(N^2).
        
        # Alternative: 
        # For each unselected j, check if it conflicts with ALL selected except maybe one?
        # No, that's complex.
        
        # Let's stick to O(N^2) but with tight loops.
        # 5280^2 = 27M checks. In Python, this is slow (~1-2s per pass).
        # We need to be smarter or accept fewer passes.
        
        # Optimization: 
        # Precompute list of selected indices?
        # selected_list = [i for i in range(n_b) if (selected >> i) & 1]
        
        # Let's try a random restart greedy + limited local search.
        # Since the greedy is already decent, maybe just do a few restarts of greedy?
        # Or a simple hill climbing.
        
        # Let's implement a fast 1-swap search:
        # 1. Get list of selected indices.
        # 2. For each selected index i:
        #    temp_selected = selected ^ (1 << i)
        #    # Find any j not in selected such that temp_selected & conflict_masks[j] == 0
        #    # We can scan j from 0 to n_b-1.
        
        selected_list = []
        for i in range(n_b):
            if (selected >> i) & 1:
                selected_list.append(i)
                
        np.random.shuffle(selected_list)
        
        for i in selected_list:
            if time.time() > deadline - 2:
                break
                
            bit_i = 1 << i
            temp_selected = selected ^ bit_i
            
            # Find a candidate j
            # We can scan all j
            for j in range(n_b):
                if j == i:
                    continue
                if (selected >> j) & 1:
                    continue # Already selected
                
                if not (temp_selected & conflict_masks[j]):
                    # Swap i out, j in
                    selected = temp_selected | (1 << j)
                    improved = True
                    # Update best
                    curr_count = bin(selected).count('1')
                    if curr_count > best_count:
                        best_count = curr_count
                        best_selected = selected
                    break # Restart search for this i? Or continue?
                    # Usually restarts help avoid local minima.
                    
            if improved:
                break # Restart outer loop to update selected_list
                
    # Convert best_selected bitmask to array
    final_b_vectors = []
    for i in range(n_b):
        if (best_selected >> i) & 1:
            final_b_vectors.append(type_b_vectors[i])
            
    if final_b_vectors:
        final_vectors = np.vstack([np.array(type_a, dtype=np.int64), np.array(final_b_vectors, dtype=np.int64)])
    else:
        final_vectors = np.array(type_a, dtype=np.int64)
        
    return final_vectors
