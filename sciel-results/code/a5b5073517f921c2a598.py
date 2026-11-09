import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n = 24
    total_masks = 1 << n
    
    # Stage 1: Coarse Grid Search
    grid_size = 2048
    thetas = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    z = np.exp(1j * thetas)
    # Z shape: (grid_size, n)
    # Z[j, k] = z[j]^k
    Z = np.empty((grid_size, n), dtype=np.complex128)
    Z[:, 0] = 1
    for k in range(1, n):
        Z[:, k] = Z[:, k-1] * z
        
    best_val = float('inf')
    best_mask = 0
    
    # We will keep a small list of top candidates to handle ties/close calls
    top_candidates = [] # list of (value, mask)
    max_top_k = 10
    
    batch_size = 5000
    
    # Pre-compute powers of 2 for bit extraction?
    # We use vectorized bit ops.
    
    for start in range(0, total_masks, batch_size):
        if time.time() > deadline:
            break
            
        end = min(start + batch_size, total_masks)
        masks = np.arange(start, end)
        
        # Generate coefficient matrix C (batch_size, n) with values +/-1
        # bits = (masks[:, None] >> np.arange(n)) & 1
        # C = 1 - 2 * bits
        
        # To save memory/time, we can do this in chunks if batch is huge, 
        # but 5000x24 is small.
        bits = (masks[:, None] >> np.arange(n)) & 1
        C = 1 - 2 * bits
        
        # Compute values V = C @ Z.T
        # C: (batch, n)
        # Z.T: (n, grid)
        # V: (batch, grid)
        V = C @ Z.T
        
        # Compute max magnitude for each candidate
        # M: (batch,)
        M = np.max(np.abs(V), axis=1)
        
        # Find the minimum in this batch
        min_idx = np.argmin(M)
        min_val = M[min_idx]
        min_mask = masks[min_idx]
        
        # Update top candidates
        if len(top_candidates) < max_top_k:
            top_candidates.append((min_val, min_mask))
            if len(top_candidates) == max_top_k:
                top_candidates.sort(key=lambda x: x[0])
        else:
            # Only insert if better than the worst in top_k
            if min_val < top_candidates[-1][0]:
                top_candidates[-1] = (min_val, min_mask)
                top_candidates.sort(key=lambda x: x[0])
                
        # Also update global best
        if min_val < best_val:
            best_val = min_val
            best_mask = min_mask

    # Stage 2: Refinement of Top Candidates
    # Use a much finer grid
    fine_grid_size = 20000
    fine_thetas = np.linspace(0, 2 * np.pi, fine_grid_size, endpoint=False)
    fine_z = np.exp(1j * fine_thetas)
    
    fine_Z = np.empty((fine_grid_size, n), dtype=np.complex128)
    fine_Z[:, 0] = 1
    for k in range(1, n):
        fine_Z[:, k] = fine_Z[:, k-1] * fine_z
        
    best_refined_val = float('inf')
    best_final_mask = best_mask # Fallback
    
    for val, mask in top_candidates:
        if time.time() > deadline:
            break
            
        bits = (mask >> np.arange(n)) & 1
        c = 1 - 2 * bits
        
        # Evaluate on fine grid
        V_fine = c @ fine_Z.T
        val_fine = np.max(np.abs(V_fine))
        
        # Optional: Golden Section Refinement
        # Find the peak location
        peak_idx = np.argmax(np.abs(V_fine))
        # The peak is near fine_thetas[peak_idx]
        # We can do a local search in a window around this peak
        # But 20000 points is likely accurate enough to distinguish the winner.
        # Delta theta = 2*pi/20000 ~ 3e-4 rad.
        # Error in |P| is small.
        
        if val_fine < best_refined_val:
            best_refined_val = val_fine
            best_final_mask = mask

    # Construct final coefficients
    bits = (best_final_mask >> np.arange(n)) & 1
    c_final = 1 - 2 * bits
    
    return c_final.tolist()
