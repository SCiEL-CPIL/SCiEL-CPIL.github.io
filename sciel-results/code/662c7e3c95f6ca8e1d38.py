"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    # a, a+d, a+2d, a+3d with 1 <= a, a+3d <= n
    # In 0-indexed: a0, a0+d, a0+2d, a0+3d where 0 <= a0, a0+3d < n
    # d >= 1
    quads_list = []
    for d in range(1, n):
        max_a0 = n - 3 * d
        for a0 in range(max_a0):
            quads_list.append((a0, a0 + d, a0 + 2 * d, a0 + 3 * d))
    quads = np.array(quads_list, dtype=np.int32)
    num_quads = len(quads)
    
    def count_mono_vec(color):
        """Count monochromatic 4-term APs using vectorized operations."""
        quad_colors = color[quads]  # shape (num_quads, 4)
        min_c = quad_colors.min(axis=1)
        max_c = quad_colors.max(axis=1)
        mono_mask = (min_c == max_c)
        return np.sum(mono_mask)

    def local_search(color, max_iters=100):
        """Perform vectorized batched local search."""
        best_color = color.copy()
        best_score = count_mono_vec(color)
        
        for _ in range(max_iters):
            # Compute current state of quads
            quad_colors = color[quads]  # (num_quads, 4)
            count_0 = (quad_colors == 0).sum(axis=1)  # (num_quads,)
            is_mono = (count_0 == 0) | (count_0 == 4)
            
            # Compute marginal gain for flipping each bit in each quad
            # change[qi, j] is the change in objective if we flip quads[qi, j]
            # assuming other bits in the quad are fixed.
            # This is an approximation for batch updates, but often works well
            # if we only flip bits with strong negative marginal gain.
            
            change = np.zeros((num_quads, 4), dtype=np.int8)
            
            # If quad is mono, flipping any bit breaks it (-1 gain)
            change[is_mono, :] = -1
            
            # If quad is not mono:
            # count_0 == 1: flipping a 0 makes it 0 (mono, +1 gain)
            # count_0 == 3: flipping a 1 makes it 4 (mono, +1 gain)
            # count_0 == 2: flipping any bit keeps it non-mono (0 gain)
            
            # Mask for count_0 == 1
            mask_c1 = (count_0 == 1)
            if np.any(mask_c1):
                # For these quads, positions with color 0 contribute +1
                # We need to set change[qi, j] = 1 where quad_colors[qi, j] == 0
                # Since count_0 == 1, exactly one position is 0.
                # We can use boolean indexing carefully.
                # Simpler: create a mask of shape (num_quads, 4)
                c1_flip_mask = mask_c1[:, None] & (quad_colors == 0)
                change[c1_flip_mask] = 1
                
            # Mask for count_0 == 3
            mask_c3 = (count_0 == 3)
            if np.any(mask_c3):
                # For these quads, positions with color 1 contribute +1
                # Since count_0 == 3, exactly one position is 1.
                c3_flip_mask = mask_c3[:, None] & (quad_colors == 1)
                change[c3_flip_mask] = 1

            # Accumulate marginal gains for each position
            marginal = np.zeros(n, dtype=np.int32)
            # Add change[qi, j] to marginal[quads[qi, j]]
            np.add.at(marginal, quads.flatten(), change.flatten())
            
            # Identify bits with negative marginal gain (flipping them reduces objective)
            flip_mask = marginal < 0
            
            if not np.any(flip_mask):
                break
                
            # Flip these bits
            color[flip_mask] = 1 - color[flip_mask]
            
            new_score = count_mono_vec(color)
            if new_score < best_score:
                best_score = new_score
                best_color = color.copy()
            else:
                # If no improvement, we might be stuck in a local minimum
                # or the batch update was too aggressive. 
                # For simplicity, we break if no improvement after a few steps?
                # Or we can try smaller batches? 
                # Let's just break to save time, as we have restarts.
                # Actually, if new_score >= best_score, the current color is worse.
                # We should revert? No, we only update best_color if better.
                # But we continue local search from the current (worse) state?
                # Usually, greedy local search requires the move to be improving.
                # Here we are doing a batch move. If the batch move increases the cost,
                # we should probably revert the batch move.
                color[flip_mask] = 1 - color[flip_mask] # Revert
                break

        return best_color, best_score

    best_overall_color = None
    best_overall_score = float('inf')
    
    # Try multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    # 1. Random starts
    for _ in range(10):
        color = rng.integers(0, 2, size=n).astype(np.int8)
        c, s = local_search(color)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_color = c.copy()
            
    # 2. Structured starts (stripes)
    # Try different stripe widths
    for width in range(1, 10):
        color = np.arange(n) % width
        # Map to 0/1: if width is even, we can just do % 2. 
        # If width is odd, it's more complex. 
        # Let's stick to simple 0/1 patterns.
        # Pattern: blocks of 1s and 0s.
        pattern = np.zeros(n, dtype=np.int8)
        for i in range(n):
            if (i // width) % 2 == 0:
                pattern[i] = 0
            else:
                pattern[i] = 1
        c, s = local_search(pattern)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_color = c.copy()
            
    # 3. Modulo patterns
    for mod in range(2, 10):
        # Color based on i % mod == 0 ? 1 : 0
        color = (np.arange(n) % mod == 0).astype(np.int8)
        c, s = local_search(color)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_color = c.copy()
            
        # Color based on i % mod in {0, 1} ? 1 : 0
        color = (np.arange(n) % mod < 2).astype(np.int8)
        c, s = local_search(color)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_color = c.copy()

    return [int(c) for c in best_overall_color]
# EVOLVE-BLOCK-END
