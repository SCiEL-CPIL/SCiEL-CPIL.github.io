# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a set of points in the 8x8x8 grid such that no five points are co-spherical.
    Uses greedy construction with local search (kick and re-greedy).
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    
    # Pre-compute squared norms
    norms = (grid ** 2).sum(axis=1).astype(np.float64)

    def det_cospherical(p_idx, q1_idx, q2_idx, q3_idx, q4_idx):
        """
        Computes the determinant for the 5 points.
        Returns 0 if cospherical (or nearly so).
        """
        p = grid[p_idx]
        q1 = grid[q1_idx]
        q2 = grid[q2_idx]
        q3 = grid[q3_idx]
        q4 = grid[q4_idx]
        
        # Vectors
        v1 = q1 - p
        v2 = q2 - p
        v3 = q3 - p
        v4 = q4 - p
        
        # Norm diffs
        d1 = norms[q1_idx] - norms[p_idx]
        d2 = norms[q2_idx] - norms[p_idx]
        d3 = norms[q3_idx] - norms[p_idx]
        d4 = norms[q4_idx] - norms[p_idx]
        
        M = np.array([
            [v1[0], v1[1], v1[2], d1],
            [v2[0], v2[1], v2[2], d2],
            [v3[0], v3[1], v3[2], d3],
            [v4[0], v4[1], v4[2], d4]
        ], dtype=np.float64)
        
        return np.linalg.det(M)

    def can_add_point(points_set, p_idx):
        """
        Checks if adding p_idx to points_set (list of indices) is valid.
        points_set is a list of indices.
        """
        if len(points_set) < 4:
            return True
        
        # Check all combinations of 4 points in points_set
        # Optimization: Early exit
        for combo in itertools.combinations(points_set, 4):
            d = det_cospherical(p_idx, combo[0], combo[1], combo[2], combo[3])
            if abs(d) < 0.5:
                return False
        return True

    def greedy_build(order, current_set=None, start_time=0, deadline=0):
        """
        Greedily adds points from 'order' to 'current_set'.
        """
        if current_set is None:
            current_set = []
        
        for idx in order:
            if time.time() > deadline:
                break
            if can_add_point(current_set, idx):
                current_set.append(idx)
        
        return current_set

    def local_search(current_set, order, start_time, deadline):
        """
        Tries to improve the set by removing a point and re-greedy filling.
        Returns the best set found.
        """
        best_set = current_set.copy()
        best_size = len(best_set)
        
        # Try a few kicks
        num_kicks = 10
        for _ in range(num_kicks):
            if time.time() > deadline:
                break
            
            if len(current_set) == 0:
                break
                
            # Pick a random point to remove
            remove_idx_pos = np.random.randint(len(current_set))
            remove_idx = current_set[remove_idx_pos]
            
            # Remove it
            new_set = current_set.copy()
            new_set.remove(remove_idx)
            
            # Re-greedy fill from the remaining order
            # We should try to add points that were previously rejected or just iterate through all available
            # To be efficient, we iterate through the original order and add if possible
            # But we must skip the removed point and those already in new_set
            # Actually, just iterating through the full order and adding if valid is safe.
            
            # Optimization: Only try to add points not in new_set
            # We can just use the full order again, as can_add_point checks validity.
            # However, to speed up, we can prioritize points not in the set.
            
            # Simple approach: Iterate through the same 'order'
            for idx in order:
                if time.time() > deadline:
                    break
                if idx in new_set:
                    continue
                if can_add_point(new_set, idx):
                    new_set.append(idx)
            
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = new_set.copy()
                current_set = new_set.copy() # Update current for next kick
            else:
                # If no improvement, revert to best_set for next kick?
                # Or keep current_set?
                # Usually, keeping the current state allows exploring nearby configurations.
                # But if we didn't improve, we might loop.
                # Let's revert to best_set to ensure progress or at least stability.
                pass 

        return best_set

    start_time = time.time()
    deadline = start_time + 580 # 20s safety margin
    
    best_set_overall = []
    best_size_overall = 0
    
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        np.random.seed(r)
        order = np.random.permutation(num_points_total)
        
        # Greedy Phase
        current_set = greedy_build(order, None, start_time, deadline)
        
        if len(current_set) > best_size_overall:
            best_size_overall = len(current_set)
            best_set_overall = current_set.copy()
        
        # Local Search Phase
        # Only do local search if we have some time left
        if time.time() < deadline - 5: # 5s margin for finalization
             improved_set = local_search(current_set, order, start_time, deadline)
             if len(improved_set) > best_size_overall:
                 best_size_overall = len(improved_set)
                 best_set_overall = improved_set.copy()

    if not best_set_overall:
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[best_set_overall].astype(np.int64)

# EVOLVE-BLOCK-END
