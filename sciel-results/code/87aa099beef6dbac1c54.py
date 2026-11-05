"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 256
    # Initialize with a random coloring (using a fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    color = rng.integers(0, 2, size=n).astype(np.int8)
    # color[i] is the color of integer i+1 (0-indexed)
    
    # Precompute all 4-term arithmetic progressions
    # a, a+d, a+2d, a+3d with 1 <= a, a+3d <= n
    # In 0-indexed: a0, a0+d, a0+2d, a0+3d where 0 <= a0, a0+3d < n
    # d >= 1
    quads = []
    for d in range(1, n):
        max_a0 = n - 3 * d
        for a0 in range(max_a0):
            quads.append((a0, a0 + d, a0 + 2 * d, a0 + 3 * d))
    quads = np.array(quads, dtype=np.int32)
    num_quads = len(quads)
    
    # For each quad, we need to check if all 4 positions have the same color.
    # A quad is monochromatic if all 4 colors are equal.
    # We can represent this as: for each quad, count how many of the 4 positions
    # have color 0 and how many have color 1. If either count is 4, it's mono.
    
    # Let's build a matrix M of shape (num_quads, 4) where each row is the 4 indices
    # Actually, let's think about the objective function more carefully.
    # Objective = sum over all quads of [1 if all 4 same color else 0]
    
    # For a quad with positions (i, j, k, l), it's mono if:
    # color[i] == color[j] == color[k] == color[l]
    # This is equivalent to: (color[i] == color[j]) AND (color[j] == color[k]) AND (color[k] == color[l])
    # Or we can use: it's mono if the number of 0s is 4 or the number of 1s is 4.
    
    # Let's use a different approach: for each quad, define a "mismatch" count.
    # A quad is mono if mismatch = 0, where mismatch counts transitions.
    # Actually, simpler: for each quad, let c0 = number of positions with color 0.
    # Mono if c0 == 0 or c0 == 4.
    
    # Let's precompute for each position, which quads it belongs to.
    # pos_to_quads[pos] = list of quad indices where this position appears
    
    pos_to_quads = {i: [] for i in range(n)}
    for qi, (a, b, c, d) in enumerate(quads):
        pos_to_quads[a].append(qi)
        pos_to_quads[b].append(qi)
        pos_to_quads[c].append(qi)
        pos_to_quads[d].append(qi)
    
    # Convert to numpy arrays for speed
    pos_to_quads_arr = []
    for i in range(n):
        pos_to_quads_arr.append(np.array(pos_to_quads[i], dtype=np.int32))
    
    def count_mono(color):
        """Count monochromatic 4-term APs."""
        total = 0
        for qi in range(num_quads):
            a, b, c, d = quads[qi]
            ca, cb, cc, cd = color[a], color[b], color[c], color[d]
            if ca == cb == cc == cd:
                total += 1
        return total
    
    # Vectorized version: for each quad, get the 4 colors and check if all equal
    def count_mono_vec(color):
        # color is shape (n,)
        # quads is shape (num_quads, 4)
        quad_colors = color[quads]  # shape (num_quads, 4)
        # Check if all 4 are equal: min == max
        min_c = quad_colors.min(axis=1)
        max_c = quad_colors.max(axis=1)
        mono_mask = (min_c == max_c)
        return np.sum(mono_mask)
    
    best_color = color.copy()
    best_score = count_mono_vec(color)
    
    # Batched local search: try flipping all bits, compute marginal gain for each bit
    # For each bit i, compute how many quads containing i would change from mono to non-mono or vice versa.
    
    # Let's do a few rounds of batched local search
    for iteration in range(50):
        # For each position, compute the marginal gain of flipping it
        # Marginal gain = change in objective if we flip this bit
        # = (new objective - old objective)
        # We want to flip bits that decrease the objective (negative marginal gain)
        
        # For each quad, determine if it's currently mono
        quad_colors = color[quads]  # (num_quads, 4)
        min_c = quad_colors.min(axis=1)
        max_c = quad_colors.max(axis=1)
        is_mono = (min_c == max_c)
        
        # For each quad, if we flip one of its positions, does it stay mono or become non-mono?
        # A quad is mono if all 4 colors are equal.
        # If we flip one position, the quad becomes non-mono UNLESS the other 3 are all different from the new color.
        # Wait, let's think more carefully.
        
        # Let the 4 colors be (c1, c2, c3, c4).
        # Currently mono if c1==c2==c3==c4.
        # If we flip c1, new colors are (1-c1, c2, c3, c4).
        # New is mono if 1-c1 == c2 == c3 == c4.
        # So if currently mono (all c), flipping one makes it (1-c, c, c, c) which is NOT mono (since 1-c != c).
        # If currently not mono, flipping one position might make it mono or not.
        
        # Let's compute for each quad and each of its 4 positions, whether flipping that position
        # changes the mono status.
        
        # For quad qi with positions (p0, p1, p2, p3) and colors (c0, c1, c2, c3):
        # Currently mono: is_mono[qi]
        # If we flip p0: new colors (1-c0, c1, c2, c3). Mono if all equal, i.e., 1-c0==c1==c2==c3.
        # This means c1==c2==c3 and 1-c0==c1, i.e., c0 != c1 and c1==c2==c3.
        
        # Let's precompute for each quad:
        # count_0 = number of positions with color 0
        # count_1 = number of positions with color 1 (= 4 - count_0)
        # Mono if count_0 == 4 or count_0 == 0.
        
        # If we flip position j in quad qi:
        # If color[j] was 0, count_0 decreases by 1.
        # If color[j] was 1, count_1 decreases by 1 (count_0 increases by 1).
        
        # Let's compute for each quad the current count_0.
        quad_colors = color[quads]  # (num_quads, 4)
        count_0 = (quad_colors == 0).sum(axis=1)  # (num_quads,)
        
        # Currently mono if count_0 == 0 or count_0 == 4
        is_mono = (count_0 == 0) | (count_0 == 4)
        
        # For each quad, for each of the 4 positions, compute if flipping changes mono status.
        # Let's create a matrix flip_effect of shape (num_quads, 4)
        # flip_effect[qi, j] = 1 if flipping position j in quad qi changes mono status, else 0
        # Actually, we need the change in objective: +1 if becomes mono, -1 if stops being mono, 0 otherwise.
        
        # If currently mono (count_0 in {0, 4}):
        #   Flipping any position makes it non-mono. So change = -1 for all 4 positions.
        # If currently not mono (count_0 in {1, 2, 3}):
        #   If count_0 == 1: flipping a 0 makes count_0=0 (mono, +1). Flipping a 1 makes count_0=2 (not mono, 0).
        #   If count_0 == 2: flipping a 0 makes count_0=1 (not mono, 0). Flipping a 1 makes count_0=3 (not mono, 0).
        #   If count_0 == 3: flipping a 0 makes count_0=2 (not mono, 0). Flipping a 1 makes count_0=4 (mono, +1).
        
        # So:
        # change[qi, j] = -1 if is_mono[qi]
        #                = +1 if (not is_mono[qi]) and count_0[qi] == 1 and color[quads[qi, j]] == 0
        #                = +1 if (not is_mono[qi]) and count_0[qi] == 3 and color[quads[qi, j]] == 1
        #                = 0 otherwise
        
        # Let's compute this vectorized.
        # is_mono is shape (num_quads,)
        # count_0 is shape (num_quads,)
        # quad_colors is shape (num_quads, 4)
        
        # Initialize change matrix to 0
        change = np.zeros((num_quads, 4), dtype=np.int8)
        
        # For mono quads, flipping any position gives -1
        mono_mask = is_mono
        change[mono_mask, :] = -1
        
        # For non-mono quads with count_0 == 1, flipping a 0 gives +1
        c1_mask = (count_0 == 1)
        # For these quads, positions with color 0 should get +1
        change[c1_mask, :] = 0  # reset
        # For each such quad, set +1 where color is 0
        for qi in np.where(c1_mask)[0]:
            change[qi, quad_colors[qi] == 0] = 1
        
        # For non-mono quads with count_0 == 3, flipping a 1 gives +1
        c3_mask = (count_0 == 3)
        change[c3_mask, :] = 0
        for qi in np.where(c3_mask)[0]:
            change[qi, quad_colors[qi] == 1] = 1
        
        # Now, for each position i, the total marginal gain is the sum of change[qi, j] 
        # over all quads qi containing position i, where j is the index of i in quad qi.
        
        # Let's accumulate: marginal[i] = sum over quads containing i of the appropriate change value.
        marginal = np.zeros(n, dtype=np.int32)
        
        for qi in range(num_quads):
            a, b, c, d = quads[qi]
            marginal[a] += change[qi, 0]
            marginal[b] += change[qi, 1]
            marginal[c] += change[qi, 2]
            marginal[d] += change[qi, 3]
        
        # Flip all positions with negative marginal gain (decreases objective)
        flip_mask = marginal < 0
        if not np.any(flip_mask):
            break
        
        color[flip_mask] = 1 - color[flip_mask]
        
        new_score = count_mono_vec(color)
        if new_score < best_score:
            best_score = new_score
            best_color = color.copy()
    
    # Return the best coloring found
    return [int(c) for c in best_color]
# EVOLVE-BLOCK-END
