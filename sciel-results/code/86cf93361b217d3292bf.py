import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N

# Precompute all points as a flat index and coordinate arrays
XS = np.arange(N, dtype=np.int32)
YS = np.arange(N, dtype=np.int32)
XX, YY = np.meshgrid(XS, YS, indexing='ij')
COORDS = np.column_stack([XX.ravel(), YY.ravel()])  # shape (4096, 2)
COORDS = COORDS.astype(np.int32)

# Precompute squared distance matrix (4096 x 4096) - might be large but manageable
# 4096 * 4096 * 4 bytes = 64 MB, acceptable
D2 = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    for j in range(i+1, NUM_POINTS):
        dx = COORDS[i, 0] - COORDS[j, 0]
        dy = COORDS[i, 1] - COORDS[j, 1]
        d = dx*dx + dy*dy
        D2[i, j] = d
        D2[j, i] = d

# Build conflict graph: two points are in conflict if adding them to a set
# would create an isosceles triangle. This is complex, so we use a different approach.

# Better approach: Use the greedy + local search with numpy acceleration.
# Key insight: use bitsets for fast conflict checking.

# For each point, precompute which other points are at the same distance
# This allows fast isosceles detection.

# Actually, let's use a cleaner approach:
# Represent the set S as a boolean array of size NUM_POINTS.
# To check if point p can be added:
#   For each existing point s in S, compute d = dist2(p, s)
#   All these d values must be distinct, AND none of them should equal
#   any dist2(s1, s2) for s1, s2 in S.

# Alternative: maintain for each point p a "conflict count" or use
# the fact that if we track all distances in S, we can check quickly.

# Let's use a different structural approach: Simulated Annealing / Local Search
# starting from a good initial greedy solution.

def is_valid_set(S):
    """Check if set S (list of point indices) has no isosceles triangles."""
    n = len(S)
    if n < 3:
        return True
    # Check all triples - O(n^3) but n is small
    # Actually this is too slow for validation. Use the distance-based check.
    dist_set = set()
    for i in range(n):
        for j in range(i+1, n):
            d = D2[S[i], S[j]]
            if d in dist_set:
                return False
            dist_set.add(d)
    return True

def greedy_construct(order):
    """Greedy construction given a fixed order of points."""
    S = []
    dist_set = set()
    in_set = np.zeros(NUM_POINTS, dtype=bool)
    
    for p in order:
        m = len(S)
        if m < 2:
            S.append(p)
            in_set[p] = True
            if m == 1:
                d = D2[p, S[0]]
                dist_set.add(d)
            continue
        
        # Check if p can be added
        valid = True
        new_dists = []
        for s in S:
            d = D2[p, s]
            if d in dist_set:
                valid = False
                break
            new_dists.append(d)
        
        if valid:
            # Check all new distances are distinct
            if len(set(new_dists)) != m:
                valid = False
        
        if valid:
            S.append(p)
            in_set[p] = True
            dist_set.update(new_dists)
    
    return S, dist_set

def local_search(S, dist_set, time_budget):
    """Local search: try to swap points out of S with points not in S."""
    start_time = time.time()
    S_set = set(S)
    in_set = np.zeros(NUM_POINTS, dtype=bool)
    for p in S:
        in_set[p] = True
    
    best_S = S[:]
    best_len = len(S)
    
    # Precompute: for each point not in S, check if it can replace any point in S
    # This is expensive, so we do it incrementally.
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try to add a point
        candidates = [p for p in range(NUM_POINTS) if not in_set[p]]
        random.shuffle(candidates)
        
        for p in candidates[:50]:  # Limit to 50 candidates per iteration for speed
            m = len(S)
            new_dists = [D2[p, s] for s in S]
            
            # Check validity
            if len(set(new_dists)) != m:
                continue
            if any(d in dist_set for d in new_dists):
                continue
            
            # Can add p
            S.append(p)
            in_set[p] = True
            dist_set.update(new_dists)
            improved = True
            if len(S) > best_len:
                best_len = len(S)
                best_S = S[:]
            break
        
        if not improved:
            # Try swaps: remove one from S, add one from outside
            S_copy = S[:]
            for i in range(len(S_copy)):
                if time.time() - start_time >= time_budget:
                    break
                removed = S_copy[i]
                S_tmp = [s for j, s in enumerate(S) if j != i]
                in_set[removed] = False
                # Recompute dist_set for S_tmp
                new_dist_set = set()
                for a in range(len(S_tmp)):
                    for b in range(a+1, len(S_tmp)):
                        new_dist_set.add(D2[S_tmp[a], S_tmp[b]])
                
                # Try to add a new point
                candidates = [p for p in range(NUM_POINTS) if not in_set[p] and p != removed]
                random.shuffle(candidates)
                
                added = False
                for p in candidates[:20]:
                    new_dists = [D2[p, s] for s in S_tmp]
                    if len(set(new_dists)) != len(S_tmp):
                        continue
                    if any(d in new_dist_set for d in new_dists):
                        continue
                    
                    # Swap successful
                    S = S_tmp + [p]
                    in_set[removed] = False
                    in_set[p] = True
                    dist_set = new_dist_set | set(new_dists)
                    added = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best_S = S[:]
                    break
                
                if not added:
                    # Undo removal
                    S = S_copy[:]
                    in_set[removed] = True
                    # Recompute dist_set
                    dist_set = set()
                    for a in range(len(S)):
                        for b in range(a+1, len(S)):
                            dist_set.add(D2[S[a], S[b]])
        
        if time.time() - start_time >= time_budget:
            break
        
        # If no improvement in a full pass, stop
        if not improved:
            break
    
    return best_S, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 310  # 20s margin
    
    best_S = []
    best_len = 0
    
    # Multiple restarts with different random orders
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        # Time for greedy phase
        greedy_time = min(5, (deadline - time.time()) * 0.3)
        if greedy_time < 1:
            break
        
        S, dist_set = greedy_construct(order)
        S_len = len(S)
        
        if S_len > best_len:
            best_S = S[:]
            best_len = S_len
        
        # Local search with remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        ls_time = min(remaining * 0.7, 60)
        if ls_time < 2:
            continue
        
        S_ls, ls_len = local_search(S, dist_set, ls_time)
        
        if ls_len > best_len:
            best_S = S_ls[:]
            best_len = ls_len
        
        # Check if we should continue
        if time.time() > deadline - 10:
            break
    
    # Convert indices to (x, y) points
    result = [(COORDS[i, 0], COORDS[i, 1]) for i in best_S]
    return result
