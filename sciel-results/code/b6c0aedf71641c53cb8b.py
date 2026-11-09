import time
import random
import numpy as np

N = 64
TOTAL_POINTS = N * N

# Precompute distance matrix
def get_distance_matrix():
    xs = np.arange(N)
    ys = np.arange(N)
    # Create grid of points
    # Shape (N, N, 2)
    # Flatten to (N*N, 2)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Compute squared distances
    # (i, j) -> dist(points[i], points[j])
    # Using broadcasting: (N*N, 1, 2) - (1, N*N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return dist_sq.astype(np.int32)

DIST_MAT = get_distance_matrix()

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s margin

    all_indices = np.arange(TOTAL_POINTS)
    
    # Helper to check if a point can be added to a set of indices
    def can_add(p_idx, s_indices):
        """
        Checks if p_idx can be added to the set of points represented by s_indices.
        s_indices is a numpy array of indices.
        """
        n_s = len(s_indices)
        if n_s < 2:
            return True
        
        # Distances from p to all points in S
        d_ps = DIST_MAT[p_idx, s_indices]
        
        # Check 1: Distances from p to S must be unique
        # If any two are equal, it's isosceles with p as apex
        if len(np.unique(d_ps)) != n_s:
            return False
            
        # Check 2 & 3: For each point A in S, distance(p, A) must not equal distance(A, B) for any B in S
        # This means for each A in S, d(p, A) should not be in the set of distances from A to S
        for i in range(n_s):
            a_idx = s_indices[i]
            d_as = DIST_MAT[a_idx, s_indices]
            # d_ps[i] is the distance between p and a_idx
            # We check if this distance exists in d_as
            # Note: d_as contains 0 at index i (dist A to A), but d_ps[i] > 0 since p != a_idx (p not in S)
            # So we can just check if d_ps[i] is in d_as
            if np.any(d_as == d_ps[i]):
                return False
                
        return True

    def greedy_construct():
        s_indices = []
        s_set = set()
        # Random order
        order = np.random.permutation(TOTAL_POINTS)
        
        for p in order:
            if time.time() > deadline - 30:
                break
            # Optimization: if s_set is large, can_add is slower.
            # But let's try.
            if can_add(p, np.array(s_indices, dtype=np.int32)):
                s_indices.append(p)
                s_set.add(p)
                
        return np.array(s_indices, dtype=np.int32), s_set

    # Initial Solution
    current_s, current_s_set = greedy_construct()
    best_s = current_s.copy()
    best_size = len(current_s)
    
    # Local Search
    # We will alternate between trying to add points and swapping points.
    # To speed up can_add, we keep s_indices as a numpy array.
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 10:
        improved = False
        current_size = len(current_s)
        
        # 1. Try to add a random point
        # Pick a few random candidates
        for _ in range(10):
            if time.time() > deadline - 10: break
            p = np.random.randint(TOTAL_POINTS)
            if p not in current_s_set:
                if can_add(p, current_s):
                    current_s = np.append(current_s, p)
                    current_s_set.add(p)
                    improved = True
                    no_improve_count = 0
                    break
        
        # 2. Try to swap: remove one, add another
        if not improved and len(current_s) > 0:
            for _ in range(20):
                if time.time() > deadline - 10: break
                # Pick a point to remove
                rem_idx = np.random.randint(len(current_s))
                rem_point = current_s[rem_idx]
                
                # Pick a point to add
                add_point = np.random.randint(TOTAL_POINTS)
                if add_point in current_s_set:
                    continue
                
                # Create new set without rem_point
                # Efficient way: boolean mask or just slicing
                if rem_idx == 0:
                    new_s = current_s[1:]
                elif rem_idx == len(current_s) - 1:
                    new_s = current_s[:-1]
                else:
                    new_s = np.concatenate((current_s[:rem_idx], current_s[rem_idx+1:]))
                    
                if can_add(add_point, new_s):
                    current_s = np.append(new_s, add_point)
                    current_s_set.remove(rem_point)
                    current_s_set.add(add_point)
                    improved = True
                    no_improve_count = 0
                    break
        
        if improved:
            if len(current_s) > best_size:
                best_s = current_s.copy()
                best_size = len(current_s)
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart or randomize
                # Let's do a random restart with a fresh greedy
                if time.time() < deadline - 60:
                    current_s, current_s_set = greedy_construct()
                    no_improve_count = 0
                else:
                    break

    # Convert best indices to (x, y) coordinates
    # Index i corresponds to (i // N, i % N)
    best_points = [(int(idx) // N, int(idx) % N) for idx in best_s]
    
    return best_points
