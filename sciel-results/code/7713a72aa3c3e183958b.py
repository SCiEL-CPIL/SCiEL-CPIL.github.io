import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    N = len(grid)
    
    def check_point(p_idx, S_indices, S_points):
        """Check if p_idx can be added to S without creating 5 cospherical/coplanar points."""
        m = len(S_indices)
        if m < 4:
            return True
        
        p = grid[p_idx].astype(np.float64)
        p_sq = grid_sq[p_idx]
        
        # Build W matrix: rows are (S_points[i] - p, |S_points[i]|^2 - |p|^2)
        W = np.empty((m, 4), dtype=np.float64)
        W[:, 0:3] = S_points.astype(np.float64) - p
        W[:, 3] = grid_sq[S_indices] - p_sq
        
        # Check all 4-combinations
        if m >= 4:
            for combo in itertools.combinations(range(m), 4):
                sub = W[list(combo)]
                det = np.linalg.det(sub)
                if abs(det) < 0.5:
                    return False
        return True
    
    def count_violations(p_idx, S_indices, S_points):
        """Count how many 4-tuples in S would violate with p."""
        m = len(S_indices)
        if m < 4:
            return 0
        
        p = grid[p_idx].astype(np.float64)
        p_sq = grid_sq[p_idx]
        
        W = np.empty((m, 4), dtype=np.float64)
        W[:, 0:3] = S_points.astype(np.float64) - p
        W[:, 3] = grid_sq[S_indices] - p_sq
        
        count = 0
        for combo in itertools.combinations(range(m), 4):
            sub = W[list(combo)]
            det = np.linalg.det(sub)
            if abs(det) < 0.5:
                count += 1
        return count
    
    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 540
    
    np.random.seed(123)
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Phase 1: Randomized greedy with lookahead
        current_indices = []
        current_points = np.empty((0, 3), dtype=np.int64)
        
        # Try multiple random orderings
        for order_attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            
            order = np.random.permutation(N)
            in_set = set()
            current_indices = []
            current_points = np.empty((0, 3), dtype=np.int64)
            
            for idx in order:
                if time.time() - start_time > time_limit:
                    break
                
                if not check_point(idx, current_indices, current_points):
                    continue
                
                # Lookahead: how many of the remaining points would this block?
                # Sample a few candidates to estimate
                remaining = [i for i in current_indices]  # placeholder
                # Simple heuristic: just accept if valid
                current_indices.append(idx)
                if len(current_points) == 0:
                    current_points = grid[idx][None, :].copy()
                else:
                    current_points = np.vstack([current_points, grid[idx]])
                in_set.add(idx)
            
            if len(current_indices) > best_size:
                best_size = len(current_indices)
                best_set = list(current_indices)
        
        # Phase 2: Swap-based local search
        if time.time() - start_time > time_limit:
            break
        
        current_indices = list(best_set) if best_size > 0 else []
        if current_indices:
            current_points = grid[current_indices].copy()
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            m = len(current_indices)
            if m == 0:
                break
            
            in_set = set(current_indices)
            
            # Try swapping each point with a non-member
            for i in range(m):
                if time.time() - start_time > time_limit:
                    break
                
                # Try to replace current_indices[i] with some other point
                # Sample candidates
                candidates = np.random.choice(N, min(50, N - m), replace=False)
                
                for new_idx in candidates:
                    if new_idx in in_set:
                        continue
                    
                    # Check if removing current_indices[i] and adding new_idx is valid
                    # First, check if new_idx is valid against S \ {current_indices[i]}
                    S_without_i = [j for j in range(m) if j != i]
                    S_indices_without_i = [current_indices[j] for j in S_without_i]
                    S_points_without_i = current_points[S_without_i]
                    
                    if not check_point(new_idx, S_indices_without_i, S_points_without_i):
                        continue
                    
                    # Also need to verify the full set after swap is valid
                    # Build new set
                    new_indices = [current_indices[j] for j in S_without_i] + [new_idx]
                    new_points = np.vstack([S_points_without_i, grid[new_idx]])
                    
                    # Verify no violations in the new set
                    valid = True
                    for k_idx in range(len(new_indices)):
                        if not check_point(new_indices[k_idx], 
                                           [new_indices[j] for j in range(k_idx)],
                                           new_points[:k_idx]):
                            valid = False
                            break
                    
                    if valid:
                        current_indices = new_indices
                        current_points = new_points
                        in_set = set(current_indices)
                        improved = True
                        
                        if len(current_indices) > best_size:
                            best_size = len(current_indices)
                            best_set = list(current_indices)
                        break
            
            if time.time() - start_time > time_limit:
                break
    
    # Final verification and return
    if not best_set:
        return np.empty((0, 3), dtype=np.int64)
    
    return grid[best_set].astype(np.int64)
