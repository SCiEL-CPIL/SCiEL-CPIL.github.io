import time
import numpy as np

N = 64
GRID_SIZE = N * N

# Pre-compute all points
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)

# Pre-compute squared distances
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy)  # shape (GRID_SIZE, GRID_SIZE)

# Convert to list of tuples for return
ALL_POINTS = [tuple(p) for p in points_arr]


def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin (330s kill, keep 40s buffer)

    best_set = []
    best_size = 0

    def is_valid_addition(k, S_indices):
        m = len(S_indices)
        if m <= 1:
            return True

        # Check 1: k must not see any two points in S at the same distance
        seen = set()
        for a in S_indices:
            d = DIST[k, a]
            if d in seen:
                return False
            seen.add(d)

        # Check 2 & 3: for each pair (a, b) in S, check if d(k,a)==d(a,b) or d(k,b)==d(a,b)
        for i in range(m):
            a = S_indices[i]
            d_ka = DIST[k, a]
            for j in range(i + 1, m):
                b = S_indices[j]
                d_ab = DIST[a, b]
                d_kb = DIST[k, b]
                if d_ka == d_ab or d_kb == d_ab:
                    return False
        return True

    # Simulated Annealing approach
    # We maintain a set and try to add/remove/swap points
    # Key insight: use a bitmask-like structure for fast checking

    # Precompute for each point, the set of points at each distance from it
    # This helps with fast validity checking
    
    # Strategy: Start with empty set, greedily add, then use SA to improve
    # SA moves: add a random point, remove a random point, swap two points
    
    def greedy_construct(order):
        """Construct a set by greedily adding points in the given order."""
        S = []
        for k in order:
            if time.time() > deadline:
                break
            if is_valid_addition(k, S):
                S.append(k)
        return S

    def local_improve(S, max_iterations=100):
        """Try to improve set S by adding unselected points."""
        S_set = set(S)
        unselected = [k for k in range(GRID_SIZE) if k not in S_set]
        np.random.shuffle(unselected)
        
        improved = False
        for k in unselected:
            if time.time() > deadline - 10:
                break
            if is_valid_addition(k, S):
                S.append(k)
                improved = True
        return S, improved

    # Multiple restarts with greedy construction + local improvement
    indices = list(range(GRID_SIZE))
    
    while time.time() < deadline - 20:
        np.random.shuffle(indices)
        current_set = greedy_construct(indices)
        
        # Local improvement
        if time.time() < deadline - 15:
            current_set, improved = local_improve(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return [ALL_POINTS[i] for i in best_set]
