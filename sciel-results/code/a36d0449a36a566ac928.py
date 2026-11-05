"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    
    # Precompute all grid points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    n_pts = len(all_points)
    
    # Precompute squared distances between all pairs using NumPy
    pts_arr = np.array(all_points, dtype=np.int32)
    # Compute squared distances: (N*N, N*N) matrix
    # Use broadcasting: (n,1,2) - (1,n,2)
    diff = pts_arr[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
    sq_dists = (diff ** 2).sum(axis=2)  # shape (n_pts, n_pts)
    
    # Map from point tuple to index
    pt_to_idx = {p: i for i, p in enumerate(all_points)}
    
    def can_add_idx(idx, S_indices, dist_sets):
        """
        Check if point at index idx can be added to set S_indices.
        S_indices: set of indices in all_points
        dist_sets: dict idx -> set of squared distances to other points in S
        """
        # 1. Check for isosceles triangles with p as the vertex.
        # All distances from p to points in S must be distinct.
        dists_to_p = []
        for j in S_indices:
            d = sq_dists[idx, j]
            dists_to_p.append(d)
        
        if len(set(dists_to_p)) != len(dists_to_p):
            return False
            
        # 2. Check for isosceles triangles with a point q in S as the vertex.
        for j in S_indices:
            d = sq_dists[idx, j]
            if d in dist_sets[j]:
                return False
                
        return True

    def add_point_idx(idx, S_indices, dist_sets):
        """Add point at index idx to S_indices and update dist_sets."""
        S_indices.add(idx)
        dists_to_p = []
        for j in S_indices:
            if j == idx:
                continue
            d = sq_dists[idx, j]
            dists_to_p.append(d)
            dist_sets[j].add(d)
        
        dist_sets[idx] = set(dists_to_p)

    def greedy_run_idx(order_indices):
        """
        Greedily construct a valid set using the given order of indices.
        Returns (set of indices, dist_sets).
        """
        S_indices = set()
        dist_sets = {}
        
        for idx in order_indices:
            if time.time() > deadline:
                break
            if can_add_idx(idx, S_indices, dist_sets):
                add_point_idx(idx, S_indices, dist_sets)
                
        return S_indices, dist_sets

    def local_search(S_indices, dist_sets, max_steps=500):
        """
        Try to improve the set by swapping points.
        Remove a point from S and try to add a different point.
        """
        current_size = len(S_indices)
        improved = True
        steps = 0
        
        while improved and steps < max_steps:
            improved = False
            steps += 1
            if time.time() > deadline:
                break
            
            # Try removing each point and see if we can add a better one
            for idx_to_remove in list(S_indices):
                if time.time() > deadline:
                    break
                
                # Temporarily remove idx_to_remove
                S_indices.remove(idx_to_remove)
                # Remove distances from idx_to_remove
                saved_dist_set = dist_sets.pop(idx_to_remove, None)
                # Remove idx_to_remove's distances from other points' dist_sets
                for j in list(S_indices):
                    d = sq_dists[idx_to_remove, j]
                    if d in dist_sets[j]:
                        dist_sets[j].discard(d)
                
                # Try to add each point not in S
                best_add = -1
                for idx_to_add in range(n_pts):
                    if idx_to_add in S_indices:
                        continue
                    if can_add_idx(idx_to_add, S_indices, dist_sets):
                        best_add = idx_to_add
                        break
                
                if best_add >= 0:
                    # Add the new point
                    add_point_idx(best_add, S_indices, dist_sets)
                    if len(S_indices) > current_size:
                        current_size = len(S_indices)
                    improved = True
                else:
                    # Restore idx_to_remove
                    S_indices.add(idx_to_remove)
                    dist_sets[idx_to_remove] = saved_dist_set if saved_dist_set is not None else set()
                    # Restore distances
                    for j in S_indices:
                        if j != idx_to_remove:
                            d = sq_dists[idx_to_remove, j]
                            dist_sets[j].add(d)
        
        return S_indices, dist_sets

    best_set_indices = set()
    best_dist_sets = {}
    best_size = 0
    
    rng = random.Random(42)
    
    # Try multiple different orderings
    # 1. Row-major order
    order1 = list(range(n_pts))
    S, ds = greedy_run_idx(order1)
    if len(S) > best_size:
        best_set_indices, best_dist_sets, best_size = S.copy(), {k: v.copy() for k, v in ds.items()}, len(S)
    
    # 2. Reversed order
    order2 = list(range(n_pts - 1, -1, -1))
    S, ds = greedy_run_idx(order2)
    if len(S) > best_size:
        best_set_indices, best_dist_sets, best_size = S.copy(), {k: v.copy() for k, v in ds.items()}, len(S)
    
    # 3. Random orders
    while time.time() < deadline - 30:
        order = list(range(n_pts))
        rng.shuffle(order)
        S, ds = greedy_run_idx(order)
        
        if len(S) > best_size:
            best_set_indices, best_dist_sets, best_size = S.copy(), {k: v.copy() for k, v in ds.items()}, len(S)
    
    # Local search refinement on the best set
    if best_size > 0 and time.time() < deadline - 5:
        best_set_indices, best_dist_sets = local_search(best_set_indices, best_dist_sets)
        best_size = len(best_set_indices)
    
    # Convert indices back to points
    result = [all_points[i] for i in best_set_indices]
    return result
# EVOLVE-BLOCK-END
