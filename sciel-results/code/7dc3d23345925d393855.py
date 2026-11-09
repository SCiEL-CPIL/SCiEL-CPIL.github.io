# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    No 5 on a sphere in the 8x8x8 grid.
    Select as many points as possible from {0,...,7}^3 such that no five
    selected points are co-spherical (or co-planar).
    """
    n = 8
    deadline = time.monotonic() + 600.0
    safety_margin = 30.0

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    N = len(grid)

    # Precompute index mapping for quick lookup if needed, but we work with indices
    # We store selected points as indices into 'grid'
    
    def is_valid_addition(selected_indices: np.ndarray, candidate_idx: int) -> bool:
        """
        Check if adding candidate_idx to selected_indices maintains validity.
        Only checks combinations involving candidate_idx and 4 other selected points.
        """
        m = len(selected_indices)
        if m < 4:
            return True
        
        p = grid[candidate_idx]
        p_sq = grid_sq[candidate_idx]
        
        # Get the 4 points from selected_indices
        # We need to check all C(m, 4) subsets of selected_indices
        # For each subset q1..q4, check if p, q1..q4 are co-spherical
        # Matrix rows: (q_i - p, |q_i|^2 - |p|^2)
        
        sel_pts = grid[selected_indices]
        sel_sqs = grid_sq[selected_indices]
        
        # Vectorized construction of the m x 4 matrix W
        # W[i] = (sel_pts[i] - p, sel_sqs[i] - p_sq)
        W = np.empty((m, 4), dtype=np.float64)
        W[:, 0:3] = sel_pts - p
        W[:, 3] = sel_sqs - p_sq
        
        # Check all combinations of 4 rows
        # If m is small, batch it. If m is large, iterate.
        # m is typically < 100 in our greedy runs, so C(m,4) is manageable.
        # C(50, 4) = 230300. C(100, 4) = 3.9M. 
        # We should be careful with very large m.
        
        if m <= 60:
            combos = np.array(list(itertools.combinations(range(m), 4)))
            if len(combos) == 0:
                return True
            sub = W[combos] # (C, 4, 4)
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # Sequential check for large m to avoid memory explosion
            for combo in itertools.combinations(range(m), 4):
                sub = W[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True

    def is_valid_swap(selected_indices: np.ndarray, out_idx: int, in_idx: int) -> bool:
        """
        Check if swapping out_idx with in_idx in selected_indices maintains validity.
        New set S' = (S \ {out_idx}) U {in_idx}.
        We only need to check subsets involving in_idx and 4 points from S \ {out_idx}.
        """
        # Create the set of indices in S excluding out_idx
        mask = selected_indices != out_idx
        new_selected_indices = selected_indices[mask]
        
        return is_valid_addition(new_selected_indices, in_idx)

    def greedy_pass(order: np.ndarray, deadline: float) -> tuple:
        """Run a greedy pass over the given order of point indices."""
        arr_idx = []
        skipped = []

        for p_idx in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            if is_valid_addition(np.array(arr_idx, dtype=np.int64) if arr_idx else np.empty(0, dtype=np.int64), p_idx):
                arr_idx.append(p_idx)
            else:
                skipped.append(p_idx)

        return np.array(arr_idx, dtype=np.int64), skipped

    def local_improve(arr_idx: np.ndarray, skipped: list, deadline: float) -> np.ndarray:
        """
        Improve the set by trying insertions and swaps.
        """
        arr_idx = arr_idx.copy()
        improved = True
        
        # Phase 1: Insertions
        while improved and (time.monotonic() + safety_margin < deadline):
            improved = False
            new_skipped = []
            for p_idx in skipped:
                if time.monotonic() + safety_margin >= deadline:
                    new_skipped.append(p_idx)
                    continue
                if is_valid_addition(arr_idx, p_idx):
                    arr_idx = np.append(arr_idx, p_idx)
                    improved = True
                else:
                    new_skipped.append(p_idx)
            skipped = new_skipped

        # Phase 2: Swaps
        # Try to swap any selected point with any skipped point
        # To keep it efficient, we limit the number of swap attempts or time
        swap_improved = True
        while swap_improved and (time.monotonic() + safety_margin < deadline):
            swap_improved = False
            if not skipped:
                break
            
            # Create a list of (out_idx, in_idx) pairs to try
            # We can randomize the order of checks to avoid bias
            out_indices = list(arr_idx)
            in_indices = list(skipped)
            
            # Shuffle for variety
            rng_swap = np.random.default_rng()
            rng_swap.shuffle(out_indices)
            rng_swap.shuffle(in_indices)
            
            for out_idx in out_indices:
                if time.monotonic() + safety_margin >= deadline:
                    break
                for in_idx in in_indices:
                    if time.monotonic() + safety_margin >= deadline:
                        break
                    if out_idx == in_idx:
                        continue
                    if is_valid_swap(arr_idx, out_idx, in_idx):
                        # Perform swap
                        arr_idx[arr_idx == out_idx] = in_idx
                        # Update skipped list: remove in_idx, add out_idx
                        skipped = [x for x in skipped if x != in_idx]
                        skipped.append(out_idx)
                        swap_improved = True
                        # Restart the loop to re-evaluate with the new set
                        # We break out of inner loop and restart outer while
                        break
                if swap_improved:
                    break

        return arr_idx

    rng = np.random.default_rng(0)
    best_idx = None
    num_restarts = 1000 # Upper bound, limited by time
    
    for restart in range(num_restarts):
        if time.monotonic() + safety_margin >= deadline:
            break
        order = rng.permutation(N)
        arr_idx, skipped = greedy_pass(order, deadline)
        arr_idx = local_improve(arr_idx, skipped, deadline)

        if best_idx is None or len(arr_idx) > len(best_idx):
            best_idx = arr_idx

    if best_idx is None:
        best_idx = np.empty(0, dtype=np.int64)

    return grid[best_idx].astype(np.int64)

# EVOLVE-BLOCK-END
