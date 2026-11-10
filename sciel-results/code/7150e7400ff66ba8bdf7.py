import numpy as np
import itertools
import time
from scipy.optimize import minimize # Not strictly needed but available

def no_5_on_a_sphere8() -> np.ndarray:
    """
    No 5 on a sphere in the 8x8x8 grid.
    """
    n = 8
    deadline = time.monotonic() + 600.0
    safety_margin = 5.0
    
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    
    # Map point index to squared norm for quick lookup
    # Actually, we can just use grid_sq directly if we keep indices
    
    def get_p_sq(p):
        # p is a tuple or array
        if isinstance(p, np.ndarray):
            return float((p ** 2).sum())
        return float((np.array(p) ** 2).sum())

    def can_add_vectorized(arr_idx: np.ndarray, p_idx: int) -> bool:
        """
        Check if point p_idx can be added to the set of points specified by arr_idx.
        arr_idx contains indices into the grid.
        """
        m = len(arr_idx)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        
        # Get the points in the current set
        q_points = grid[arr_idx]
        q_sqs = grid_sq[arr_idx]
        
        # Build the matrix W where rows are (q - p, |q|^2 - |p|^2)
        # Shape: (m, 4)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = q_points - p
        w[:, 3] = q_sqs - p_sq
        
        # We need to check all combinations of 4 rows
        # If m is large, this is expensive.
        # For m > 30, the number of combinations is > 27k.
        # For m > 50, it is > 230k.
        
        # Generate indices for combinations
        # Using itertools is slow. 
        # We can use a trick: if m is small, use itertools.
        # If m is large, we might need a different approach or accept the cost if vectorized.
        
        if m <= 40:
            # Use itertools for simplicity and low memory
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True
        else:
            # For larger m, we can try to batch.
            # But generating all combinations in a list is memory heavy.
            # Let's stick to itertools but break early.
            # The early break is the key.
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True

    def greedy_pass(order: np.ndarray, deadline: float) -> np.ndarray:
        """
        Run a greedy pass over the given order of point indices.
        Returns the array of selected point indices.
        """
        selected = []
        selected_set = set()
        
        for p_idx in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            
            p_idx_int = int(p_idx)
            
            # Check if can add
            if can_add_vectorized(np.array(selected, dtype=np.int64), p_idx_int):
                selected.append(p_idx_int)
                selected_set.add(p_idx_int)
        
        return np.array(selected, dtype=np.int64)

    best_sel = None
    best_len = 0
    
    # Phase 1: Random Restarts
    # We will run many random restarts
    # Each restart picks a random permutation of the 512 points
    
    rng = np.random.default_rng(42)
    
    # Estimate time per restart
    # A restart involves up to 512 checks.
    # Each check involves up to C(m,4) dets.
    # If we assume average m is 50, C(50,4) is 230k dets.
    # 512 * 230k is too slow.
    # We need to be smarter.
    
    # Optimization: 
    # The check `can_add_vectorized` is the bottleneck.
    # We can limit the number of points we try to add if the set is already large?
    # No, we want max k.
    
    # Let's try to optimize the check further.
    # Precompute the "lifted" points?
    # L(p) = (p_x, p_y, p_z, |p|^2)
    # The condition is det([L(q_i) - L(p) ... ]) ? No.
    # The condition is that p, q1, q2, q3, q4 are cospherical.
    # This is equivalent to the rank of the matrix of lifted points being < 5?
    # Matrix M with rows [L(p_i), 1].
    # If rank < 5, they are cospherical.
    
    # Let's use this formulation.
    # L(p) = (x, y, z, x^2+y^2+z^2)
    # We augment with 1.
    # Check if 5 points are linearly dependent in R^5.
    
    lifted = np.hstack([grid, grid_sq[:, None]]) # (512, 4)
    lifted_aug = np.hstack([lifted, np.ones((N, 1))]) # (512, 5)
    
    def can_add_rank(arr_idx: np.ndarray, p_idx: int) -> bool:
        m = len(arr_idx)
        if m < 4:
            return True
        
        # We need to check if p_idx and any 4 from arr_idx are linearly dependent in R^5.
        # This is equivalent to checking if the 5x5 matrix has det 0.
        
        p_lifted = lifted_aug[p_idx]
        
        # Get the 4 points from arr_idx
        # We need to check all 4-combinations of arr_idx.
        
        # Optimization: 
        # If we maintain the set of "spheres" (hyperplanes in lifted space) defined by 4 points?
        # No, let's stick to the determinant check but optimize the loop.
        
        # For small m, just do it.
        # For large m, we can use the fact that if we find one conflict, we stop.
        
        # To speed up, we can precompute the matrix for the current set?
        # No, p changes.
        
        # Let's try to use a faster determinant check for the 4x4 case derived from the 5x5.
        # The 4x4 matrix in the problem description is:
        # Rows: (q - p, |q|^2 - |p|^2)
        # This is exactly the Schur complement or similar.
        
        # Let's stick to the previous `can_add_vectorized` but make it faster.
        # The bottleneck is `np.linalg.det` in a Python loop.
        
        # Can we vectorize the determinant calculation for all 4-combinations?
        # Yes, if we can generate the indices efficiently.
        
        if m <= 30:
            # Small set, just loop
            q_points = grid[arr_idx]
            q_sqs = grid_sq[arr_idx]
            p = grid[p_idx]
            p_sq = grid_sq[p_idx]
            
            w = np.empty((m, 4), dtype=np.float64)
            w[:, 0:3] = q_points - p
            w[:, 3] = q_sqs - p_sq
            
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True
        else:
            # Larger set. 
            # We can sample? No, must be exact.
            # We can break early.
            q_points = grid[arr_idx]
            q_sqs = grid_sq[arr_idx]
            p = grid[p_idx]
            p_sq = grid_sq[p_idx]
            
            w = np.empty((m, 4), dtype=np.float64)
            w[:, 0:3] = q_points - p
            w[:, 3] = q_sqs - p_sq
            
            # Iterate
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True

    # Run restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.monotonic() + safety_margin >= deadline:
            break
        
        # Random order
        order = rng.permutation(N)
        
        sel = greedy_pass(order, deadline)
        
        if len(sel) > best_len:
            best_len = len(sel)
            best_sel = sel
            
            # Try to improve locally?
            # For now, just keep the best.
            
        # Optional: Local improvement
        # Try to add skipped points?
        # The greedy pass already tries all points in the order.
        # If a point was skipped, it was because it conflicted with the current set.
        # It might become valid if we removed some points? No, we only add.
        # So local improvement would require swapping.
        
        # Let's try a simple swap-based local search if time permits.
        # But first, let's see how good the greedy is.

    if best_sel is None:
        return np.empty((0, 3), dtype=np.int64)
    
    return grid[best_sel].astype(np.int64)
