import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)
    
    # Precompute CW masks and weights
    cw_masks = np.zeros((D, n), dtype=np.int64)
    weights = np.zeros(D, dtype=np.int64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # CW path edges
        dist = (t - s) % n
        for k in range(dist):
            e = (s + k) % n
            cw_masks[i, e] = 1
            
    # Base load if all CW
    base_load = cw_masks.T @ weights
    
    # Contribution if switched to CCW
    # diff = CCW_mask - CW_mask
    # CCW_mask = 1 - CW_mask
    # diff = 1 - 2*CW_mask
    diff_masks = 1 - 2 * cw_masks
    contrib = weights[:, np.newaxis] * diff_masks # shape (D, n)
    
    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.int64)
    
    # Gray code iteration
    # Number of steps: 2^D
    num_steps = 1 << D
    
    # Initialize load for mask 0 (all CW)
    current_load = base_load.copy()
    current_mask = 0
    current_routing = np.zeros(D, dtype=np.int64)
    
    # Check initial state
    max_load = np.max(current_load)
    if max_load < best_max_load:
        best_max_load = max_load
        best_routing = current_routing.copy()
        
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    for step in range(1, num_steps):
        if time.time() > deadline:
            break
            
        # Gray code: g = i ^ (i >> 1)
        # To find which bit changed between g_{step-1} and g_step:
        # changed_bit = g_{step-1} ^ g_step
        # Alternatively, simply track the mask and update.
        # Standard Gray code sequence generation:
        # g_k = k ^ (k >> 1)
        # But we need the routing vector x corresponding to the Gray code index?
        # No, we need to iterate through all masks.
        # Using Gray code allows us to update load incrementally.
        
        # Let's just use the property that in the sequence of Gray codes,
        # only one bit flips at each step.
        # We need to know which bit flipped.
        
        # Previous gray code
        prev_g = (step - 1) ^ ((step - 1) >> 1)
        curr_g = step ^ (step >> 1)
        
        # Bit that changed
        diff_bits = prev_g ^ curr_g
        # Find the index of the set bit in diff_bits
        # Since only one bit is set, we can find it efficiently
        # For D=15, we can use a lookup or log2
        # idx = int(np.log2(diff_bits)) is slow.
        # Use bit_length - 1
        idx = diff_bits.bit_length() - 1
        
        # Determine if the bit was set to 1 or 0 in the new mask (curr_g)
        # If curr_g has bit idx set, we are switching TO CCW (x=1) for this demand?
        # Wait, the mask `curr_g` represents the state.
        # In our formulation, x_i=1 means CCW.
        # So if bit idx is 1 in curr_g, x_idx becomes 1.
        # If bit idx is 0 in curr_g, x_idx becomes 0.
        
        # We maintain `current_load` and `current_routing`.
        # We need to update `current_routing` as well to return it.
        
        # Update load
        if (curr_g >> idx) & 1:
            # Switching from CW (0) to CCW (1)
            # Add contrib
            current_load += contrib[idx]
            current_routing[idx] = 1
        else:
            # Switching from CCW (1) to CW (0)
            # Subtract contrib
            current_load -= contrib[idx]
            current_routing[idx] = 0
            
        # Check max load
        max_load = np.max(current_load)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = current_routing.copy()
            
    # Local search polish: try 1-flip and 2-flip improvements
    # This is a refinement to ensure we are at a local optimum, though
    # exhaustive search should already guarantee global optimum.
    # However, if the exhaustive search was cut short by the deadline,
    # this can help.
    
    # 1-flip search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(D):
            # Try flipping bit i
            new_routing = best_routing.copy()
            new_routing[i] = 1 - new_routing[i]
            
            # Calculate new load
            # load = base + sum(contrib[j] for j where new_routing[j]=1)
            # We can compute this quickly from the current best state
            # But we don't have the current load of best_routing stored easily.
            # Let's recompute load for the candidate.
            # This is O(D*n) per flip, which is fast for D=15, n=10.
            mask = new_routing.astype(np.int64)
            new_load = base_load + mask @ contrib
            new_max_load = np.max(new_load)
            
            if new_max_load < best_max_load - 1e-9:
                best_max_load = new_max_load
                best_routing = new_routing
                improved = True
                break # Restart 1-flip search since state changed

    # 2-flip search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(D):
            if time.time() > deadline:
                break
            for j in range(i + 1, D):
                if time.time() > deadline:
                    break
                
                new_routing = best_routing.copy()
                new_routing[i] = 1 - new_routing[i]
                new_routing[j] = 1 - new_routing[j]
                
                mask = new_routing.astype(np.int64)
                new_load = base_load + mask @ contrib
                new_max_load = np.max(new_load)
                
                if new_max_load < best_max_load - 1e-9:
                    best_max_load = new_max_load
                    best_routing = new_routing
                    improved = True
                    break # Restart 2-flip search
            if improved:
                break

    return best_routing.astype(np.float64)
