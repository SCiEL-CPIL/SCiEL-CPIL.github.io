"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    quads = []
    for d in range(1, n):
        max_a0 = n - 3 * d
        for a0 in range(max_a0):
            quads.append((a0, a0 + d, a0 + 2 * d, a0 + 3 * d))
    quads = np.array(quads, dtype=np.int32)
    num_quads = len(quads)
    
    def count_mono_vec(color):
        quad_colors = color[quads]
        min_c = quad_colors.min(axis=1)
        max_c = quad_colors.max(axis=1)
        return np.sum(min_c == max_c)
    
    def local_search(color, max_iter=1000):
        color = color.copy()
        current_score = count_mono_vec(color)
        
        # Precompute flat quads for bincount
        flat_quads = quads.ravel()
        
        for _ in range(max_iter):
            # 1. Get colors for each quad
            quad_colors = color[quads] # (num_quads, 4)
            
            # 2. Count zeros in each quad
            count_0 = (quad_colors == 0).sum(axis=1) # (num_quads,)
            
            # 3. Determine mono status
            is_mono = (count_0 == 0) | (count_0 == 4)
            
            # 4. Compute change in objective if we flip each bit in each quad
            # change[qi, j] is the delta objective if we flip bit j in quad qi
            change = np.zeros((num_quads, 4), dtype=np.int8)
            
            # Case 1: Currently Mono. Flipping any bit makes it non-mono (-1)
            change[is_mono, :] = -1
            
            # Case 2: Currently Non-Mono
            # Subcase 2a: count_0 == 1. Flipping a 0 makes it 0 (Mono, +1)
            mask_1 = (count_0 == 1) & ~is_mono
            if np.any(mask_1):
                idx_1 = np.where(mask_1)[0]
                # Set change to 1 where color is 0, else 0
                change[idx_1, :] = (quad_colors[idx_1] == 0).astype(np.int8)
                
            # Subcase 2b: count_0 == 3. Flipping a 1 makes it 4 (Mono, +1)
            mask_3 = (count_0 == 3) & ~is_mono
            if np.any(mask_3):
                idx_3 = np.where(mask_3)[0]
                # Set change to 1 where color is 1, else 0
                change[idx_3, :] = (quad_colors[idx_3] == 1).astype(np.int8)
                
            # Subcase 2c: count_0 == 2. Flipping any bit keeps it non-mono (0)
            # No action needed, change is already 0
            
            # 5. Accumulate marginal gain for each position
            # marginal[i] = sum of change[qi, j] for all quads qi containing i at position j
            flat_change = change.ravel()
            marginal = np.bincount(flat_quads, weights=flat_change, minlength=n)
            
            # 6. Flip all bits with negative marginal gain
            flip_mask = marginal < 0
            if not np.any(flip_mask):
                break
            
            color[flip_mask] ^= 1
            
            # Optional: Check score improvement to stop early if no improvement
            # But batched flips might temporarily increase score? 
            # Actually, marginal < 0 means each individual flip decreases score.
            # Since flips are independent in terms of marginal calculation (based on current state),
            # flipping all negative marginal bits simultaneously is a valid "greedy" step.
            # It is guaranteed to decrease the objective function value.
            
        return color, count_mono_vec(color)
    
    best_color = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 115 # Leave 5s buffer
    
    # Try multiple random starts
    for seed in range(20):
        if time.time() - start_time > time_limit:
            break
            
        rng = np.random.default_rng(seed)
        init_color = rng.integers(0, 2, size=n).astype(np.int8)
        
        opt_color, score = local_search(init_color)
        
        if score < best_score:
            best_score = score
            best_color = opt_color.copy()
            
    # If no valid color was found (should not happen), fallback
    if best_color is None:
        best_color = np.zeros(n, dtype=np.int8)
        
    return [int(c) for c in best_color]
# EVOLVE-BLOCK-END
