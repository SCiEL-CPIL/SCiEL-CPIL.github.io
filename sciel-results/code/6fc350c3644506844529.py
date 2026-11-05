import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: 150s kill, aim for 130s limit
    limit = 130.0
    
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
    
    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        c = coloring[aps_arr]
        return int(np.all(c == c[:, 0:1], axis=1).sum())
    
    def compute_deltas(coloring):
        """
        Compute the change in mono-AP count if each position is flipped.
        Returns an array of shape (n,) with deltas.
        """
        c = coloring[aps_arr]  # shape (K, 4)
        n0 = (c == 0).sum(axis=1)  # shape (K,)
        
        deltas = np.zeros(n, dtype=np.int32)
        
        # Case 1: n0 == 4 (All 0s, Mono). Flipping any of the 4 positions breaks mono.
        # Delta is -1 for each of the 4 positions.
        mask_40 = (n0 == 4)
        if mask_40.any():
            # Get the indices of the positions in these APs
            # We need to subtract 1 from deltas for each position in these APs
            # We can flatten the AP indices and use bincount
            affected_aps = aps_arr[mask_40]
            flat_indices = affected_aps.flatten()
            # Subtract 1 for each occurrence
            deltas -= np.bincount(flat_indices, minlength=n).astype(np.int32)
            
        # Case 2: n0 == 1 (One 0, Three 1s). Flipping the 0 to 1 makes it Mono (All 1s).
        # Delta is +1 for the position with 0.
        mask_10 = (n0 == 1)
        if mask_10.any():
            affected_aps = aps_arr[mask_10]
            affected_colors = c[mask_10]
            # Find the position in each AP where color is 0
            # We can use np.where or argmax/argmin tricks
            # Since there's exactly one 0, the index of the 0 is where color == 0
            # Let's find the column index of the 0 for each row
            # This is a bit tricky to vectorize perfectly without a loop or advanced indexing
            # However, K is ~10k, so a loop over affected APs is fine if count is small, 
            # but we want vectorization.
            
            # Alternative: 
            # For each AP, we know the 4 indices. We know which one is 0.
            # We can construct a list of indices to increment.
            # Let's use a loop for this part as it's likely sparse or manageable, 
            # OR use advanced indexing.
            
            # Advanced indexing approach:
            # We have rows `r` and columns `col` where `affected_colors[r, col] == 0`.
            rows, cols = np.where(affected_colors == 0)
            # The actual position index is affected_aps[r, cols]
            positions_to_inc = affected_aps[rows, cols]
            deltas += np.bincount(positions_to_inc, minlength=n).astype(np.int32)

        # Case 3: n0 == 3 (Three 0s, One 1). Flipping the 1 to 0 makes it Mono (All 0s).
        # Delta is +1 for the position with 1.
        mask_30 = (n0 == 3)
        if mask_30.any():
            affected_aps = aps_arr[mask_30]
            affected_colors = c[mask_30]
            # Find the position where color is 1
            rows, cols = np.where(affected_colors == 1)
            positions_to_inc = affected_aps[rows, cols]
            deltas += np.bincount(positions_to_inc, minlength=n).astype(np.int32)
            
        return deltas

    def local_search(coloring, time_limit):
        """
        Greedy local search with vectorized delta computation.
        """
        current_score = count_mono(coloring)
        
        while time.time() < time_limit:
            deltas = compute_deltas(coloring)
            min_delta = deltas.min()
            
            if min_delta >= 0:
                break  # Local optimum reached
            
            # Flip all positions with the most negative delta
            # This is a "steepest descent" step
            flip_mask = (deltas == min_delta)
            flip_positions = np.where(flip_mask)[0]
            
            coloring[flip_positions] = 1 - coloring[flip_positions]
            current_score += min_delta * len(flip_positions)
            
        return coloring, current_score

    def perturb(coloring, rng, k):
        """
        Flip k random positions.
        """
        positions = rng.integers(0, n, k)
        coloring[positions] = 1 - coloring[positions]

    # Main ILS Loop
    best_coloring = None
    best_score = float('inf')
    
    # Initialize with a random balanced coloring
    # Using a fixed seed for reproducibility in the first few, then random
    rng = np.random.default_rng(42)
    
    # Initial coloring: Random
    coloring = rng.integers(0, 2, n)
    
    # Ensure balance if possible? Random is usually fine.
    
    while time.time() < limit - 5.0:
        # Run local search
        current_coloring = coloring.copy()
        current_coloring, current_score = local_search(current_coloring, time.time() + 2.0) # Give it 2s max per LS
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = current_coloring.copy()
        
        # Perturb the best solution found so far, or the current one?
        # Standard ILS: Perturb the current local optimum.
        # If we just found a new best, we might want to explore around it.
        # Let's perturb the current local optimum.
        # Number of perturbations: Start small, increase?
        # Fixed small perturbation is often robust.
        k_perturb = 10
        perturb(current_coloring, rng, k_perturb)
        
        # Set current_coloring as the starting point for next LS
        coloring = current_coloring
        
        # Check time
        if time.time() > limit - 10.0:
            break

    # Final polish: Run local search on the best solution found with remaining time
    if best_coloring is not None and time.time() < limit - 1.0:
        final_coloring = best_coloring.copy()
        final_coloring, final_score = local_search(final_coloring, limit - 0.5)
        if final_score < best_score:
            best_score = final_score
            best_coloring = final_coloring

    if best_coloring is None:
        return [int(x) for x in rng.integers(0, 2, n)]
        
    return [int(x) for x in best_coloring]
# EVOLVE-BLOCK-END
