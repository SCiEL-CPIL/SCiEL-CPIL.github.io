import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit = 145.0  # Safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (total_aps, 4)
    
    # Build position-to-AP-col mapping
    pos_to_ap_col_lists = [[] for _ in range(n)]
    for idx in range(total_aps):
        for k in range(4):
            pos = aps_arr[idx, k]
            pos_to_ap_col_lists[pos].append((idx, k))
            
    pos_to_ap_col = [np.array(lst, dtype=np.int32) for lst in pos_to_ap_col_lists]
    
    def count_mono(coloring):
        c = coloring[aps_arr]  # shape (total_aps, 4)
        return int(np.all(c == c[:, 0:1], axis=1).sum())
    
    def compute_single_delta(coloring, pos):
        """Compute delta for flipping a single position. Fast."""
        ap_cols = pos_to_ap_col[pos]
        if len(ap_cols) == 0:
            return 0
            
        ap_indices = ap_cols[:, 0]
        col_indices = ap_cols[:, 1]
        
        # Get colors of these APs
        c = coloring[aps_arr[ap_indices]]  # shape (M, 4)
        
        # Current state
        n0 = (c == 0).sum(axis=1)
        n1 = 4 - n0
        is_mono = (n0 == 4) | (n1 == 4)
        
        # After flip at col_indices
        flipped_colors = c[np.arange(len(ap_indices)), col_indices]
        
        new_n0 = n0.copy()
        new_n1 = n1.copy()
        
        mask_0 = (flipped_colors == 0)
        new_n0[mask_0] -= 1
        new_n1[mask_0] += 1
        
        mask_1 = (flipped_colors == 1)
        new_n0[mask_1] += 1
        new_n1[mask_1] -= 1
        
        new_is_mono = (new_n0 == 4) | (new_n1 == 4)
        
        delta = int(new_is_mono.sum() - is_mono.sum())
        return delta

    def compute_deltas(coloring):
        """Compute delta for flipping each position. Fully vectorized."""
        c = coloring[aps_arr]  # (total_aps, 4)
        n0 = (c == 0).sum(axis=1)  # (total_aps,)
        n1 = 4 - n0
        
        deltas = np.zeros(n, dtype=np.int32)
        
        # Mono APs: flipping any position gives delta -1
        mono_mask = (n0 == 4) | (n1 == 4)
        if mono_mask.any():
            mono_aps = aps_arr[mono_mask]
            mono_flat = mono_aps.flatten()
            deltas -= np.bincount(mono_flat, minlength=n).astype(np.int32)
        
        # 3-1 APs (n0==3, n1==1): flipping the odd one out (color 1) gives delta +1
        mask_31 = (n0 == 3) & (n1 == 1)
        if mask_31.any():
            ap_indices_31 = np.where(mask_31)[0]
            row_idx, col_idx = np.where(c[ap_indices_31] == 1)
            target_positions = aps_arr[ap_indices_31[row_idx], col_idx]
            deltas += np.bincount(target_positions, minlength=n).astype(np.int32)
            
        # 1-3 APs (n0==1, n1==3): flipping the odd one out (color 0) gives delta +1
        mask_13 = (n0 == 1) & (n1 == 3)
        if mask_13.any():
            ap_indices_13 = np.where(mask_13)[0]
            row_idx, col_idx = np.where(c[ap_indices_13] == 0)
            target_positions = aps_arr[ap_indices_13[row_idx], col_idx]
            deltas += np.bincount(target_positions, minlength=n).astype(np.int32)
        
        return deltas

    def simulated_annealing(coloring, time_limit):
        current_score = count_mono(coloring)
        best_score = current_score
        best_coloring = coloring.copy()
        
        T = 1.0
        cooling_rate = 0.9999
        min_T = 0.001
        
        rng = np.random.default_rng(42)
        
        while time.time() < time_limit and T > min_T:
            # Pick a random position to flip
            pos = rng.integers(0, n)
            
            # Calculate delta for this specific flip
            delta = compute_single_delta(coloring, pos)
            
            # Accept or reject
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                coloring[pos] = 1 - coloring[pos]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = coloring.copy()
            
            T *= cooling_rate
            
        return best_coloring, best_score

    best_coloring = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Initialize with random coloring
    coloring = rng.integers(0, 2, n).astype(np.int32)
    
    # Run SA
    coloring, score = simulated_annealing(coloring, limit - 10.0)
    
    if score < best_score:
        best_score = score
        best_coloring = coloring.copy()
        
    # Final polishing with local search (greedy)
    if best_coloring is not None and time.time() < limit - 2.0:
        coloring = best_coloring.copy()
        current_score = count_mono(coloring)
        
        # Greedy local search
        for _ in range(100):
            if time.time() > limit - 2.0:
                break
            deltas = compute_deltas(coloring)
            min_delta = deltas.min()
            if min_delta >= 0:
                break
            flip_pos = int(np.argmin(deltas))
            coloring[flip_pos] = 1 - coloring[flip_pos]
            current_score += min_delta
            if current_score < best_score:
                best_score = current_score
                best_coloring = coloring.copy()
        
    if best_coloring is None:
        return [int(x) for x in np.random.randint(0, 2, n)]
    return [int(x) for x in best_coloring]
# EVOLVE-BLOCK-END
