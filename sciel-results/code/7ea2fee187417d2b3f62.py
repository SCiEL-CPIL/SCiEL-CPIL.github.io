# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import sys
import random

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large set of points in the 8x8x8 grid such that no 5 points are co-spherical.
    Uses a combination of randomized greedy initialization and advanced local search
    (add, swap, kick) to maximize the set size.
    """
    n = 8
    # Safety margin: Evaluator kills at 630s. We aim to finish by 600s.
    deadline = time.time() + 590
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    
    # Precompute squared norms for all grid points
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Map point tuple to index in grid for fast lookup
    point_to_idx = {tuple(p): i for i, p in enumerate(grid)}
    
    def is_co_spherical_5(p0, p1, p2, p3, p4):
        """
        Check if 5 points are co-spherical.
        Returns True if they are (which is invalid for us).
        Uses the 4-vector relative determinant method.
        """
        # Relative vectors from p0
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        v4 = p4 - p0
        
        # Squared norm differences
        d1 = sq_norms[point_to_idx[tuple(p1)]] - sq_norms[point_to_idx[tuple(p0)]]
        d2 = sq_norms[point_to_idx[tuple(p2)]] - sq_norms[point_to_idx[tuple(p0)]]
        d3 = sq_norms[point_to_idx[tuple(p3)]] - sq_norms[point_to_idx[tuple(p0)]]
        d4 = sq_norms[point_to_idx[tuple(p4)]] - sq_norms[point_to_idx[tuple(p0)]]
        
        M = np.array([
            [v1[0], v1[1], v1[2], d1],
            [v2[0], v2[1], v2[2], d2],
            [v3[0], v3[1], v3[2], d3],
            [v4[0], v4[1], v4[2], d4]
        ], dtype=np.float64)
        
        det_val = np.linalg.det(M)
        return abs(det_val) < 0.5

    def can_add(arr, p):
        """
        Check if adding point p to array arr (of indices) keeps the set valid.
        arr: list of indices into grid.
        p: index into grid.
        """
        m = len(arr)
        if m < 4:
            return True
        
        p_idx = p
        p_pt = grid[p_idx]
        p_sq = sq_norms[p_idx]
        
        # Precompute relative vectors and norm diffs for all points in arr
        # w[i] = [q_i - p, |q_i|^2 - |p|^2]
        w = np.empty((m, 4), dtype=np.float64)
        for i, idx in enumerate(arr):
            q_pt = grid[idx]
            w[i, 0:3] = q_pt - p_pt
            w[i, 3] = sq_norms[idx] - p_sq
        
        # Check all 4-combinations
        # Optimization: If m is small, itertools is fine.
        # For m > 30, this becomes slow. We might need to prune or use vectorized det.
        # Let's use itertools for now, but break early on failure.
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def get_valid_set(indices):
        """Convert list of indices to numpy array of points."""
        return grid[indices].copy()

    # Phase 1: Randomized Greedy Initialization
    best_indices = None
    best_size = 0
    
    # Try multiple random orders and heuristic orders
    orders = []
    # Random orders
    for i in range(20):
        orders.append(np.random.permutation(N))
    # Heuristic orders: by x, y, z, norm
    orders.append(np.argsort(grid[:, 0]))
    orders.append(np.argsort(grid[:, 1]))
    orders.append(np.argsort(grid[:, 2]))
    orders.append(np.argsort(sq_norms))
    orders.append(np.argsort(-sq_norms))
    
    for order in orders:
        if time.time() > deadline - 300: # Leave time for search
            break
            
        current_indices = []
        for idx in order:
            if can_add(current_indices, idx):
                current_indices.append(idx)
        
        if len(current_indices) > best_size:
            best_size = len(current_indices)
            best_indices = current_indices.copy()
            
    if best_indices is None:
        return np.empty((0, 3), dtype=np.int64)
        
    current_indices = best_indices.copy()
    current_size = best_size
    
    # Phase 2: Local Search
    # We will try to add points, and if stuck, try swaps/kicks.
    
    def try_add_any(indices):
        """Try to add any single point not in indices. Returns (new_indices, added_idx) or (None, None)."""
        current_set = set(indices)
        # Shuffle candidates for randomness
        candidates = [i for i in range(N) if i not in current_set]
        random.shuffle(candidates)
        
        for idx in candidates:
            if can_add(indices, idx):
                return indices + [idx], idx
        return None, None

    def try_swap(indices):
        """
        Try to swap one point out and one point in.
        Returns (new_indices, removed_idx, added_idx) or (None, None, None).
        """
        m = len(indices)
        # Try removing each point and adding a new one
        for i in range(m):
            if time.time() > deadline - 100:
                break
            removed_idx = indices[i]
            reduced_indices = indices[:i] + indices[i+1:]
            reduced_set = set(reduced_indices)
            
            # Try adding any point not in reduced_indices
            candidates = [j for j in range(N) if j not in reduced_set and j != removed_idx]
            random.shuffle(candidates)
            
            for idx in candidates:
                if can_add(reduced_indices, idx):
                    new_indices = reduced_indices + [idx]
                    return new_indices, removed_idx, idx
        return None, None, None

    def try_kick(indices, kick_size=2):
        """
        Try to remove kick_size points and add kick_size new points.
        """
        m = len(indices)
        if m < kick_size:
            return None, None, None
            
        # Select kick_size points to remove
        remove_indices = random.sample(range(m), kick_size)
        remove_set = set(remove_indices)
        
        # Build reduced list
        reduced_indices = [idx for i, idx in enumerate(indices) if i not in remove_set]
        reduced_set = set(reduced_indices)
        
        # Try to add kick_size new points
        added_indices = []
        candidates = [j for j in range(N) if j not in reduced_set]
        random.shuffle(candidates)
        
        for idx in candidates:
            if len(added_indices) >= kick_size:
                break
            # Check if adding idx to (reduced + added) is valid
            temp_indices = reduced_indices + added_indices
            if can_add(temp_indices, idx):
                added_indices.append(idx)
                
        if len(added_indices) == kick_size:
            new_indices = reduced_indices + added_indices
            return new_indices, list(remove_indices), added_indices
            
        return None, None, None

    # Main search loop
    stuck_count = 0
    max_stuck = 20
    
    while time.time() < deadline - 50:
        # Try to add
        new_indices, added = try_add_any(current_indices)
        if new_indices:
            current_indices = new_indices
            current_size += 1
            stuck_count = 0
            continue
        
        # If stuck, try swap
        new_indices, rem, add = try_swap(current_indices)
        if new_indices:
            current_indices = new_indices
            # Size unchanged, but configuration changed
            stuck_count += 1
            if stuck_count > max_stuck:
                # Try kick
                new_indices, rem, add = try_kick(current_indices, kick_size=2)
                if new_indices:
                    current_indices = new_indices
                    stuck_count = 0
                else:
                    # Try kick 3
                    new_indices, rem, add = try_kick(current_indices, kick_size=3)
                    if new_indices:
                        current_indices = new_indices
                        stuck_count = 0
                    else:
                        # Restart local search with a fresh randomization or break
                        # For now, just break if we can't improve
                        break
            continue
            
        # If swap failed, try kick directly
        new_indices, rem, add = try_kick(current_indices, kick_size=2)
        if new_indices:
            current_indices = new_indices
            stuck_count = 0
            continue
            
        # If all failed, we are likely at a local optimum
        # Try a few random kicks of larger size
        found_improvement = False
        for kick_size in [3, 4]:
            new_indices, rem, add = try_kick(current_indices, kick_size=kick_size)
            if new_indices:
                current_indices = new_indices
                found_improvement = True
                break
                
        if not found_improvement:
            # Give up on this configuration, maybe try a new greedy restart if time permits
            # But we already did greedy. Let's just break.
            break

    # Final verification and cleanup
    # The can_add function ensures validity, but let's be safe.
    # We don't need to re-verify all 5-tuples if can_add was correct.
    
    return get_valid_set(current_indices)

# EVOLVE-BLOCK-END
