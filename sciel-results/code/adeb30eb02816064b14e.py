"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # 1. Precompute Comparability Matrix
    # C[i, j] is True if i and j are comparable (one is subset of the other)
    # i is subset of j iff (i & j) == i
    # We can generate all masks
    masks = np.arange(total_masks, dtype=np.int32)
    
    # To save memory and time, we can compute the matrix in blocks or use broadcasting if possible.
    # Broadcasting 4096x4096 int32 array is 67MB. 
    # Comparability: (masks[:, None] & masks[None, :]) == masks[:, None] OR (masks[:, None] & masks[None, :]) == masks[None, :]
    # This creates two 4096x4096 arrays. 
    # Let's try to do it efficiently.
    
    # Actually, we can just compute it.
    # A = masks[:, None]
    # B = masks[None, :]
    # C = (A & B) == A | (A & B) == B
    
    # This might be slow or memory heavy in pure Python/NumPy without care.
    # 4096^2 = 16.7M elements. 
    # int32 array is 67MB. Two of them is 134MB. Boolean result is 16MB.
    # This is acceptable.
    
    A = masks[:, None]
    B = masks[None, :]
    inter = A & B
    C = (inter == A) | (inter == B)
    # C is a boolean array of shape (4096, 4096)
    
    # 2. Greedy Initialization
    # Start with a random or structured set.
    # Let's start with a greedy approach: pick the mask that maximizes comparability with already picked ones.
    # Initially, any mask is fine. Let's pick 0 (empty set) as it is subset of all.
    selected = [0]
    available = set(range(1, total_masks))
    
    # To speed up greedy, we can just pick the next best.
    # But 40 steps is small.
    for _ in range(m - 1):
        best_score = -1
        best_mask = -1
        # We can vectorize the score calculation for all available masks
        # Score of mask u is C[u, selected].sum()
        # Let's just loop, it's only 40 * 4096 ops total for init.
        for u in available:
            score = C[u, selected].sum()
            if score > best_score:
                best_score = score
                best_mask = u
        selected.append(best_mask)
        available.remove(best_mask)
        
    selected_arr = np.array(selected, dtype=np.int32)
    unselected_set = available
    
    # 3. Local Search with NumPy Acceleration
    # We maintain:
    # selected_arr: numpy array of current selected masks
    # C_sel_sums: numpy array of length 4096, where C_sel_sums[u] = C[u, selected_arr].sum()
    
    C_sel_sums = C[:, selected_arr].sum(axis=1)
    
    # Current total score
    # Pairs = (C[sel, sel].sum() - m) / 2
    current_score = (C[selected_arr, selected_arr].sum() - m) / 2.0
    
    improved = True
    passes = 0
    max_passes = 5000 # Limit to ensure termination
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        best_gain = 0
        best_i = -1
        best_u = -1
        
        # For each i in selected, find the best u in unselected to swap with
        for i in range(m):
            x = selected_arr[i]
            
            # Calculate gain for swapping x with any u not in selected
            # gain_u = (C_sel_sums[u] - C[u, x]) - (C[x, selected_arr].sum() - 1)
            # Note: C[x, selected_arr].sum() includes C[x, x] which is 1.
            # So sum over b != x is C[x, selected_arr].sum() - 1.
            
            old_contrib = C[x, selected_arr].sum() - 1
            
            # Vectorized gain calculation for all u
            # gains[u] = C_sel_sums[u] - C[u, x] - old_contrib
            gains = C_sel_sums - C[:, x] - old_contrib
            
            # Mask out selected items (we can't swap with themselves, and they are not in unselected)
            # We need to find max gain among u not in selected_arr
            # Create a mask of unselected
            # Since selected_arr is small, we can just set gains[sel] to -inf
            gains[selected_arr] = -1
            
            # Find the best u
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain > best_gain:
                best_gain = max_gain
                best_i = i
                best_u = max_gain_idx
                
        if best_gain > 0:
            # Perform swap
            x_old = selected_arr[best_i]
            u_new = best_u
            
            # Update selected_arr
            selected_arr[best_i] = u_new
            
            # Update unselected_set
            unselected_set.remove(u_new)
            unselected_set.add(x_old)
            
            # Update C_sel_sums
            # C_sel_sums changes for all u because the set 'selected' changed.
            # C_sel_sums[u] = C[u, selected_arr].sum()
            # We can recompute it? 
            # C[:, selected_arr].sum(axis=1) is 4096 * 40 operations.
            # 160k ops per pass. 
            # If we do 5000 passes, 800M ops. Might be too slow in Python/NumPy overhead?
            # Let's try to update incrementally.
            # C_sel_sums[u] = C[u, selected_arr].sum()
            # New selected_arr has u_new instead of x_old.
            # Delta for any u: C[u, u_new] - C[u, x_old]
            # So C_sel_sums += C[:, u_new] - C[:, x_old]
            C_sel_sums = C_sel_sums + C[:, u_new] - C[:, x_old]
            
            # Update current_score (optional, for tracking)
            # current_score += best_gain / 2.0 ? 
            # Gain calculated was sum of (C[u,b] - C[x,b]) for b in S\{x}.
            # This is the change in the sum of the row for u minus the row for x?
            # The total score change is exactly best_gain / 2?
            # No, the formula for gain was derived for the sum of off-diagonal elements involving the swapped items.
            # Let's just recalculate score periodically or trust the search.
            
            improved = True
            
    # Convert to lists
    result = []
    for mask in selected_arr:
        # mask is numpy int, convert to python int
        m_int = int(mask)
        result.append([b for b in range(n) if (m_int >> b) & 1])
        
    return result

# EVOLVE-BLOCK-END
