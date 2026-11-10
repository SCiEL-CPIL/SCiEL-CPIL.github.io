# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Uses Simulated Annealing to maximize the number of points in an 8x8x8 grid
    such that no five points are co-spherical (or co-planar).
    
    The strategy involves:
    1. Initializing with a random greedy set.
    2. Iteratively proposing moves:
       - Add a random candidate point if valid.
       - Remove a random point (to allow exploration).
       - Swap a random existing point with a random candidate point.
    3. Accepting moves based on the change in set size (k).
       - If k increases: Always accept.
       - If k decreases: Accept with probability exp(-delta / T).
       - If k is same: Accept with probability exp(-delta / T) where delta is 0 (always accept for diversity, or slight penalty if we want to prune). 
       Actually, since we want to maximize k, and validity is binary:
       - Move is valid if new set is valid.
       - Score = k.
       - If new_k > old_k: Accept.
       - If new_k < old_k: Accept with prob exp((new_k - old_k) / T).
       - If new_k == old_k: Accept with prob 0.5 (random walk) or exp(0/T)=1? 
         Let's use 0.5 to encourage exploration without changing size.
    """
    n = 8
    deadline = time.time() + 570  # Safety margin for 600s limit
    
    # Pre-compute grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Pre-compute squared norms for all grid points to speed up determinant check
    # |p|^2 = x^2 + y^2 + z^2
    grid_sums = (grid ** 2).sum(axis=1)
    
    def is_valid_subset(indices):
        """
        Check if the subset of grid indices is valid (no 5 co-spherical).
        indices: np.array of int
        """
        if len(indices) < 5:
            return True
        
        pts = grid[indices]
        sums = grid_sums[indices]
        k = len(pts)
        
        # We need to check all combinations of 5 points.
        # C(k, 5) can be large. 
        # Optimization: If k is small, just check.
        # If k is large, we might need a smarter check, but for SA moves,
        # we only change 1 point at a time. 
        # However, for the initial greedy, we need a full check.
        # For SA, we can reuse the "can_add" logic relative to the existing set.
        
        # This generic checker is slow for large k. 
        # Better to use the incremental check for SA moves.
        # But for the initial greedy, we need a fast validity check.
        
        # Let's implement a specific check for a set of points.
        # 5 points are co-spherical iff det of 5x5 matrix is 0.
        # Matrix rows: [x, y, z, x^2+y^2+z^2, 1]
        
        # To speed this up for the initial greedy, we can use the incremental method.
        # But for a general subset, we have to check all 5-combos.
        # C(50, 5) is ~2M. C(100, 5) is ~75M. Too slow.
        
        # So, for the initial greedy, we will use the incremental 'can_add' logic.
        # For SA, we will use a specialized check that only checks the new point
        # against the existing 4-combos of the set, plus the 4-combos involving
        # the removed point (if swap) or just the new point (if add).
        
        # This function is only used if we need to re-validate a whole set from scratch,
        # which we try to avoid.
        return True # Placeholder, logic handled in main loop via incremental checks

    def can_add_to_set(set_indices, p_idx):
        """
        Check if adding grid[p_idx] to the set defined by set_indices keeps it valid.
        set_indices: np.array of int
        p_idx: int
        """
        if len(set_indices) < 4:
            return True
        
        p = grid[p_idx]
        p_sum = grid_sums[p_idx]
        
        # Existing points
        pts = grid[set_indices]
        sums = grid_sums[set_indices]
        m = len(pts)
        
        # We need to check if p forms a co-spherical set with any 4 existing points.
        # The condition is that the 4x4 matrix with rows (q_i - p, |q_i|^2 - |p|^2) has det 0.
        # q_i - p is vector. |q_i|^2 - |p|^2 is scalar.
        # Let's construct the matrix W of shape (m, 4).
        # W[i] = [q_i - p, |q_i|^2 - |p|^2]
        
        # q_i - p:
        diffs = pts - p
        # |q_i|^2 - |p|^2:
        diff_sums = sums - p_sum
        
        W = np.hstack([diffs, diff_sums[:, None]])
        
        # We need to check if any 4 rows of W are linearly dependent.
        # i.e., det of any 4x4 submatrix is 0.
        
        # If m < 4, already handled.
        # If m is small, iterate combos.
        # If m is large, this is expensive.
        # C(m, 4) grows as m^4.
        
        # Optimization: 
        # We can compute the rank of W. If rank < 4, then there exist 4 dependent rows?
        # No. Rank < 4 means ALL rows are dependent in a way that some 4 might be independent?
        # Actually, if rank(W) < 4, then NO 4 rows are linearly independent.
        # So if rank(W) < 4, it means EVERY 4-combo has det 0? 
        # Wait. If rank is 3, then any 4 rows are dependent. So det is 0 for ALL combos.
        # So if rank(W) < 4, it's INVALID.
        
        # If rank(W) == 4, then SOME 4 rows are independent.
        # But we need to know if ANY 4 rows are dependent.
        # It is possible that rank is 4, but a specific 4-combo is dependent.
        
        # So rank check is a necessary condition for validity?
        # If rank < 4, invalid.
        # If rank == 4, it might still be invalid if a specific 4-combo is dependent.
        
        # However, for integer points, determinants are integers.
        # If det is non-zero, it is >= 1 in magnitude.
        
        # Let's stick to checking combos if m is small (< 20?).
        # If m is large, we need a better way.
        
        # Actually, in the greedy phase, m grows from 0 to ~50.
        # C(50, 4) is ~230k. That's fast enough.
        # In SA, m is ~50-60.
        # C(60, 4) is ~487k.
        # Doing this for every move in SA might be too slow if we have many moves.
        
        # Let's optimize the combo check using vectorized numpy.
        
        if m < 4:
            return True
            
        # Generate indices for combos
        # Using itertools.combinations is slow in Python loop.
        # We can pre-generate combos for a fixed size? No, m changes.
        
        # Alternative:
        # Compute the determinant of all 4x4 submatrices.
        # We can use a sliding window or just iterate.
        
        # Let's try to vectorize the determinant calculation.
        # It's hard to vectorize det directly.
        
        # Heuristic: 
        # Most points are NOT co-spherical.
        # We can sample a few random 4-combos first? No, must be exact.
        
        # Let's use the fact that we only add ONE point.
        # The new point p is fixed.
        # We check combos of existing points.
        
        # If m > 30, C(m,4) is large.
        # But we can break early if we find a det < 0.5.
        
        # Let's implement a fast loop for small m, and maybe a smarter check for large m.
        # For now, assume m < 40 in most iterations.
        
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = W[combos]
        
        # Calculate determinants
        # np.linalg.det is batched? Yes, for (..., M, M).
        dets = np.linalg.det(sub)
        
        return not np.any(np.abs(dets) < 0.5)

    def remove_and_check(set_indices, p_idx_to_remove):
        """
        Check if removing p_idx_to_remove from set_indices keeps it valid.
        Actually, removing a point CANNOT create a new co-spherical 5-tuple.
        It can only destroy existing ones.
        So the remaining set is ALWAYS valid if the original was valid.
        So this check is always True.
        """
        return True

    # --- Initial Greedy Construction ---
    np.random.seed(42)
    random.seed(42)
    
    best_set = []
    best_set_indices = np.array([], dtype=np.int64)
    
    # Run a few greedy restarts to find a good starting point
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        order = np.random.permutation(N)
        current_set = []
        current_indices = []
        
        for p_idx in order:
            # Check if we can add p_idx
            if len(current_indices) < 4:
                current_set.append(grid[p_idx])
                current_indices.append(p_idx)
                continue
                
            if can_add_to_set(np.array(current_indices), p_idx):
                current_set.append(grid[p_idx])
                current_indices.append(p_idx)
                
        if len(current_indices) > len(best_set_indices):
            best_set = current_set
            best_set_indices = np.array(current_indices, dtype=np.int64)
            
    # --- Simulated Annealing ---
    if len(best_set_indices) == 0:
        return np.array(best_set).astype(np.int64)
        
    current_indices = best_set_indices.copy()
    current_k = len(current_indices)
    best_indices = current_indices.copy()
    best_k = current_k
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # We want to maximize k.
    # Move:
    # 1. Try to add a random point not in set.
    # 2. Try to remove a random point from set.
    # 3. Try to swap a random point in set with a random point not in set.
    
    # Pre-compute list of all indices
    all_indices = np.arange(N)
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min # Keep exploring at low T or break? 
            # Let's restart with slightly higher T if stuck?
            # Or just continue.
            
        # Select move type
        r = random.random()
        
        if r < 0.5:
            # Add move
            # Pick random point not in set
            # Use a set for O(1) lookup
            current_set_set = set(current_indices.tolist())
            candidates = np.array([i for i in all_indices if i not in current_set_set])
            
            if len(candidates) == 0:
                continue
                
            p_idx = candidates[random.randint(0, len(candidates)-1)]
            
            if can_add_to_set(current_indices, p_idx):
                new_indices = np.append(current_indices, p_idx)
                new_k = current_k + 1
                # Always accept if valid and increases k
                current_indices = new_indices
                current_k = new_k
                if current_k > best_k:
                    best_k = current_k
                    best_indices = current_indices.copy()
            else:
                # Invalid, reject
                pass
                
        elif r < 0.7:
            # Remove move
            if current_k > 5: # Don't go too small
                remove_idx = random.randint(0, current_k - 1)
                # Removing is always valid
                new_indices = np.delete(current_indices, remove_idx)
                new_k = current_k - 1
                
                # Accept with prob exp(-1/T)
                if random.random() < np.exp(-1.0 / T):
                    current_indices = new_indices
                    current_k = new_k
        else:
            # Swap move
            if current_k > 0:
                # Pick point to remove
                remove_idx = random.randint(0, current_k - 1)
                p_remove = current_indices[remove_idx]
                
                # Pick point to add
                current_set_set = set(current_indices.tolist())
                candidates = np.array([i for i in all_indices if i not in current_set_set])
                
                if len(candidates) == 0:
                    continue
                    
                p_add = candidates[random.randint(0, len(candidates)-1)]
                
                # Check if adding p_add to (set - p_remove) is valid
                # We need to check p_add against the remaining 4-combos.
                # The remaining set is current_indices without p_remove.
                
                # Construct new indices temporarily
                # To avoid copying large arrays, we can filter
                # But can_add_to_set takes an array.
                
                # Optimization: 
                # The new set is S' = S \ {p_remove} U {p_add}
                # We need to check if p_add is valid with S \ {p_remove}.
                # Note: p_add might form a co-sphere with 4 points from S \ {p_remove}.
                
                # Let's create the filtered array
                mask = current_indices != p_remove
                remaining_indices = current_indices[mask]
                
                if can_add_to_set(remaining_indices, p_add):
                    # Valid swap
                    new_indices = remaining_indices
                    new_indices = np.append(new_indices, p_add)
                    new_k = current_k # Size unchanged
                    
                    # Accept with prob 0.5 to allow random walk
                    if random.random() < 0.5:
                        current_indices = new_indices
                        # current_k stays same
                        if current_k > best_k:
                            best_k = current_k
                            best_indices = current_indices.copy()
                else:
                    # Invalid swap
                    pass

        # Cool down
        T *= cooling_rate

    # Return the best found set
    best_points = grid[best_indices]
    return best_points.astype(np.int64)

# EVOLVE-BLOCK-END
