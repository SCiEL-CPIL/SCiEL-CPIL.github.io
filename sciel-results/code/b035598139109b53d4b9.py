import random
import time
import numpy as np

N = 64
TOTAL_POINTS = N * N
TIME_LIMIT = 280.0

# Precompute squared distances
# Index mapping: (x, y) -> x * N + y
def get_idx(x, y):
    return x * N + y

# dist_matrix[i][j] is squared distance between point i and point j
# We only need to store it for i < j? Or full matrix?
# Full matrix is 4096x4096.
# Let's use a flat array or 2D numpy array.
# 4096 * 4096 * 8 bytes (int64) = 128 MB.
# 4096 * 4096 * 4 bytes (int32) = 64 MB. Max dist is 63^2 + 63^2 = 7938. int32 is fine.

# Initialize distance matrix
# To save memory and init time, we can compute on the fly if needed, 
# but 64MB is fine and speed is critical.
# However, initializing 16M entries in Python loops is slow.
# We can use NumPy broadcasting.

xs = np.arange(N)
ys = np.arange(N)
# Create meshgrid of coordinates
XX, YY = np.meshgrid(xs, ys, indexing='ij')
# Flatten to 1D arrays of coordinates
pts_x = XX.flatten()
pts_y = YY.flatten()

# Compute squared distances
# dist[i][j] = (pts_x[i] - pts_x[j])^2 + (pts_y[i] - pts_y[j])^2
# Using broadcasting: 
# diff_x = pts_x[:, None] - pts_x[None, :]
# This creates a 4096x4096 array.
# diff_x ** 2 + diff_y ** 2

# This might take a few seconds to compute, but it's a one-time cost.
# Let's check memory: 4096x4096 float32 is 64MB. float64 is 128MB.
# We can use int32.

def precompute_distances():
    # Using int32 to save memory
    dx = pts_x.astype(np.int32)[:, None] - pts_x.astype(np.int32)[None, :]
    dy = pts_y.astype(np.int32)[:, None] - pts_y.astype(np.int32)[None, :]
    dists = dx*dx + dy*dy
    return dists

DIST_MATRIX = precompute_distances()

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 10.0
    
    # Helper to get index from point
    def p_idx(p):
        return p[0] * N + p[1]
        
    # Helper to check if point p_idx can be added to set of indices S
    # S is a list of indices
    def can_add(p, S):
        if not S:
            return True
        
        # 1. Check for duplicate distances from p to points in S
        # d_p_s = DIST_MATRIX[p][s]
        # We need to ensure all d_p_s are distinct.
        
        # 2. Check for isosceles with pairs in S
        # For s1, s2 in S:
        # d(p, s1) == d(p, s2) -> caught by 1
        # d(p, s1) == d(s1, s2)
        # d(p, s2) == d(s1, s2)
        
        K = len(S)
        if K < 2:
             # Just check if p is not in S (assumed) and no dup dists (trivial for K=1)
             return True

        # Fast path: Check duplicate distances from p
        # Get distances
        # d_vec = DIST_MATRIX[p][S] 
        # This is a numpy array view/slice.
        # Check if len(set(d_vec)) == len(d_vec)
        # Converting to set of numpy scalars is slow.
        # Use np.unique? Or just a python set of ints?
        
        # Let's use a python set for the first check, it's fast for small K.
        seen_dists = set()
        for s in S:
            d = DIST_MATRIX[p][s]
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
        # Now check pairs
        # This is the O(K^2) part.
        # We can optimize by iterating S.
        # For i in range(K):
        #   s1 = S[i]
        #   d_p_s1 = DIST_MATRIX[p][s1]
        #   for j in range(i+1, K):
        #      s2 = S[j]
        #      d_s1_s2 = DIST_MATRIX[s1][s2]
        #      if d_p_s1 == d_s1_s2: return False
        #      d_p_s2 = DIST_MATRIX[p][s2]
        #      if d_p_s2 == d_s1_s2: return False
        #      # d_p_s1 == d_p_s2 already checked
        
        for i in range(K):
            s1 = S[i]
            d_p_s1 = DIST_MATRIX[p][s1]
            for j in range(i+1, K):
                s2 = S[j]
                d_s1_s2 = DIST_MATRIX[s1][s2]
                if d_p_s1 == d_s1_s2:
                    return False
                d_p_s2 = DIST_MATRIX[p][s2]
                if d_p_s2 == d_s1_s2:
                    return False
        return True

    # Greedy construction
    def greedy_fill(initial_set=None, time_limit=1.0):
        if initial_set is None:
            current = []
            current_set = set()
        else:
            current = list(initial_set)
            current_set = set(initial_set)
            
        # List of all points not in current_set
        # To avoid O(N^2) filtering each time, we can just shuffle all points and skip if in set.
        # Or maintain a list of available points.
        
        all_points = list(range(TOTAL_POINTS))
        random.shuffle(all_points)
        
        start = time.time()
        for p in all_points:
            if time.time() > time_limit:
                break
            if p in current_set:
                continue
            if can_add(p, current):
                current.append(p)
                current_set.add(p)
        return current

    # Iterated Greedy Strategy
    best_set = []
    best_size = 0
    
    # Initial Greedy
    current_set = greedy_fill(time_limit=5.0)
    best_set = list(current_set)
    best_size = len(current_set)
    
    # ILS Loop
    while time.time() < deadline:
        # Perturb: remove k random points
        if len(current_set) < 3:
            # Reset if too small
            current_set = greedy_fill(time_limit=2.0)
            continue
            
        k_remove = random.randint(1, min(5, len(current_set)))
        points_to_remove = random.sample(current_set, k_remove)
        
        # Create perturbed set
        perturbed = [p for p in current_set if p not in points_to_remove]
        
        # Greedy fill on perturbed set
        # Use a short time limit for this inner greedy to allow many restarts
        inner_time = min(5.0, max(0.5, (deadline - time.time()) * 0.1))
        new_set = greedy_fill(perturbed, time_limit=inner_time)
        
        if len(new_set) > best_size:
            best_size = len(new_set)
            best_set = list(new_set)
            current_set = new_set
        else:
            # Accept worse solution with some probability (Simulated Annealing style)
            # Or just keep exploring from new_set to avoid getting stuck in the same local optimum repeatedly?
            # Usually, if new is worse, we might revert to best or keep new.
            # Let's keep new to explore the neighborhood, but track best.
            current_set = new_set
            
    # Convert indices back to (x, y) tuples
    result = [(idx // N, idx % N) for idx in best_set]
    return result
