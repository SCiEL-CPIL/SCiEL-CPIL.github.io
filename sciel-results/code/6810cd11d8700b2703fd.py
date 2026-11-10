# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    grid_f = grid.astype(np.float64)

    # Precompute all 5-point conflict sets
    # A set of 5 points is "conflicting" if they lie on a common sphere or plane
    # For 5 points p0..p4, they are cospherical/coplanar iff the 5x5 matrix
    # with rows (pi - p0, |pi|^2 - |p0|^2) has rank < 5 (det of any 4x4 submatrix involving p0 is 0)
    # Actually, 5 points are cospherical iff the 5x5 determinant of
    # [[x, y, z, x^2+y^2+z^2, 1] for each point] is 0.
    
    # Precompute the 5x5 matrix for all 5-point subsets
    # This is C(512,5) = 2,290,515,000 combinations - too many to precompute all
    # Instead, we'll use a different approach: for each point, precompute which other points
    # form conflicts with 4 others
    
    # Better approach: precompute for each pair of points, which 3rd points complete a "bad" 5-set
    # Actually, let's use the determinant approach more cleverly.
    
    # For a set S of points, a new point p can be added iff for every 4-tuple q1..q4 in S,
    # the 4x4 matrix with rows (qi - p, |qi|^2 - |p|^2) has |det| >= 0.5
    
    # Key insight: we can precompute, for each point p, and each 4-subset of other points,
    # whether they form a conflict with p. But that's still too many.
    
    # Alternative: use the fact that the condition is about the 5 points being affinely dependent
    # in the lifted space. 5 points are on a sphere/plane iff their lifted vectors are linearly dependent.
    
    # Let's precompute the lifted vectors: v_i = (x_i, y_i, z_i, x_i^2+y_i^2+z_i^2, 1)
    lifted = np.column_stack([grid_f, grid_sq, np.ones(N)])  # (N, 5)
    
    # Precompute all 5-point determinants using a smarter method
    # Actually, let's just do the search with fast checking
    
    # For the search, we need to check: can point p be added to set S?
    # This requires checking C(|S|, 4) determinants.
    # For |S| up to ~30, that's manageable.
    
    # Precompute combination indices
    combos_cache = {}
    for m in range(4, 50):
        idx = np.arange(m)
        combos = list(itertools.combinations(idx, 4))
        if combos:
            combos_cache[m] = np.array(combos, dtype=np.int32)
        else:
            combos_cache[m] = np.empty((0, 4), dtype=np.int32)
    
    def can_add_fast(S, p_idx):
        """Check if point p_idx can be added to set S (list of indices)."""
        m = len(S)
        if m < 4:
            return True
        
        p = grid_f[p_idx]
        p_sq = grid_sq[p_idx]
        
        S_arr = np.array(S)
        S_pts = grid_f[S_arr]
        diffs = S_pts - p  # (m, 3)
        sq_diffs = grid_sq[S_arr] - p_sq  # (m,)
        rows = np.column_stack([diffs, sq_diffs])  # (m, 4)
        
        if m not in combos_cache:
            return True  # fallback
        
        combo_arr = combos_cache[m]
        if combo_arr.shape[0] == 0:
            return True
        
        sub = rows[combo_arr]  # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)
    
    def greedy_independence(weights=None, max_points=40):
        """
        Greedy independence set: repeatedly pick the point with fewest conflicts
        with the current set, weighted by the given weights.
        """
        if weights is None:
            weights = np.ones(N)
        
        S = []
        S_set = set()
        remaining = set(range(N))
        
        for _ in range(max_points):
            if not remaining:
                break
            if time.time() > deadline - 30:
                break
            
            # For each candidate, compute conflict count with S
            best_cand = None
            best_score = -np.inf
            best_conflicts = np.inf
            
            # Sample if too many candidates
            cand_list = list(remaining)
            if len(cand_list) > 200:
                cand_list = np.random.choice(cand_list, size=200, replace=False).tolist()
            
            for c in cand_list:
                if c in S_set:
                    continue
                if can_add_fast(S, c):
                    # Score: prefer points with high weight
                    score = weights[c]
                    if score > best_score:
                        best_score = score
                        best_cand = c
            
            if best_cand is None:
                break
            
            S.append(best_cand)
            S_set.add(best_cand)
            remaining.discard(best_cand)
        
        return S
    
    def local_search_improve(S, max_rounds=30):
        """
        Improve S by trying swaps: remove one point, add another.
        """
        if len(S) < 4:
            return S
        
        current = list(S)
        current_set = set(current)
        improved = True
        rounds = 0
        
        while improved and rounds < max_rounds and time.time() < deadline - 30:
            improved = False
            rounds += 1
            for i in range(len(current)):
                if time.time() > deadline - 30:
                    break
                
                # Try removing current[i]
                remaining = [x for x in current if x != current[i]]
                remaining_set = set(remaining)
                
                # Try adding candidates not in remaining_set
                candidates = [j for j in range(N) if j not in remaining_set]
                np.random.shuffle(candidates)
                
                found = False
                for j in candidates[:80]:
                    if can_add_fast(remaining, j):
                        current = remaining + [j]
                        current_set = set(current)
                        improved = True
                        found = True
                        break
                
                if found:
                    break  # restart scan after a successful swap
        
        return current
    
    # --- Main search ---
    np.random.seed(42)
    best_set = []
    best_size = 0
    
    num_restarts = 300
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Vary the weighting strategy
        r = restart % 6
        if r == 0:
            weights = np.ones(N)
        elif r == 1:
            # Bias toward corners (far from center)
            center = np.array([3.5, 3.5, 3.5])
            dist = np.sum((grid_f - center) ** 2, axis=1)
            weights = dist + 1e-6
        elif r == 2:
            # Bias toward center
            center = np.array([3.5, 3.5, 3.5])
            dist = np.sum((grid_f - center) ** 2, axis=1)
            weights = (dist.max() + 1.0 - dist) + 1e-6
        elif r == 3:
            # Random weights
            weights = np.random.dirichlet(np.ones(N)) * N
        elif r == 4:
            # Bias toward face centers
            center = np.array([3.5, 3.5, 3.5])
            dist = np.sum((grid_f - center) ** 2, axis=1)
            # Prefer points on faces (where one coord is 0 or 7)
            on_face = (grid[:, 0] == 0) | (grid[:, 0] == 7) | \
                       (grid[:, 1] == 0) | (grid[:, 1] == 7) | \
                       (grid[:, 2] == 0) | (grid[:, 2] == 7)
            weights = np.where(on_face, dist + 2.0, dist) + 1e-6
        else:
            # Bias toward edges
            center = np.array([3.5, 3.5, 3.5])
            dist = np.sum((grid_f - center) ** 2, axis=1)
            edge_count = ((grid == 0) | (grid == 7)).sum(axis=1)
            weights = edge_count * 1.0 + dist + 1e-6
        
        # Normalize weights
        weights = weights / (weights.sum() + 1e-12) * N
        
        # Greedy fill
        current = greedy_independence(weights, max_points=40)
        
        # Local search improvement
        if len(current) >= 4 and time.time() < deadline - 30:
            current = local_search_improve(current, max_rounds=20)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
        
        # Try to extend best_set further
        if best_size > 0 and time.time() < deadline - 30:
            # Try adding more points to best_set
            extended = list(best_set)
            extended_set = set(extended)
            candidates = [i for i in range(N) if i not in extended_set]
            np.random.shuffle(candidates)
            
            for c in candidates[:50]:
                if time.time() > deadline - 30:
                    break
                if can_add_fast(extended, c):
                    extended.append(c)
                    extended_set.add(c)
            
            if len(extended) > best_size:
                best_size = len(extended)
                best_set = extended
    
    # Final verification and cleanup
    if best_size > 0:
        result = np.array([grid[i] for i in best_set], dtype=np.int64)
    else:
        result = np.empty((0, 3), dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
