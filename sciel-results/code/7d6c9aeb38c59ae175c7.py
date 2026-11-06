import time
import random
import numpy as np

N = 64
GRID_SIZE = N * N

# Pre-compute points
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
ALL_POINTS = [tuple(p) for p in points_arr]

xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy)  # shape (GRID_SIZE, GRID_SIZE)

# For each point k, we precompute a sorted array of distances to all other points.
# We also precompute, for each distance value, the list of points at that distance.
# This allows us to quickly find "conflicting" points.
# However, a simpler and often faster approach for the "is k valid?" check 
# in a growing set S is to maintain a dictionary or set of "forbidden" conditions.

# Let's try a different structural approach: **Greedy with Lookahead / Backtracking**.
# Or even better: **Iterative Greedy with Conflict Resolution**.

# The key bottleneck in the current approach is the O(m) Python loop in `is_valid_addition`.
# Can we vectorize the entire check?

# Check 1: No two points in S have the same distance to k.
# dks = DIST[k, S_arr]
# if len(dks) != len(np.unique(dks)): return False

# Check 2: For no a in S, is d(k, a) equal to d(a, b) for any b in S.
# This is equivalent to: for all a, b in S, d(k, a) != d(a, b) and d(k, b) != d(a, b).
# Note that d(a, b) = d(b, a).
# So for a fixed k, we need to ensure that for all pairs (a, b) in S, 
# d(k, a) is not in the set of distances from a to S, and d(k, b) is not in the set of distances from b to S.

# Let's precompute for each point a, a dictionary: dist_to_points[a] = {d: [list of points at distance d from a]}
# This might use a lot of memory. GRID_SIZE = 4096. 
# Each point has at most ~4095 distances. Many are duplicates.
# Number of unique distances from a point is at most ~2*N*N = 8192, but usually much less.
# Let's precompute the unique distances for each point and the counts.
# Actually, for the check "is d in the distances from a to S?", we need to know if there exists b in S such that d(a, b) = d.

# Alternative: Maintain a global structure that counts, for each pair of distances (d1, d2), how many points a in S have d(k, a) = d1 and d(a, b) = d2 for some b. This seems complex.

# Let's stick to the vectorized check but optimize the second part.
# The second part: "for any a in S, is d(k, a) in {d(a, b) for b in S}?"
# We can precompute for each a, a boolean array `has_dist_at_a` of size (max_dist+1) which is True if there is at least one point at that distance from a.
# But S changes. So we need a dynamic count.
# `count_dist_from_a[d]` = number of points b in S such that d(a, b) = d.
# We can maintain a 2D array `dist_count[a, d]`? No, that's too big (4096 * 8192).

# Let's try a **different search strategy**: **Genetic Algorithm with Bitmask Representation**.
# Represent a set as a Python `int` (bitmask).
# Fitness: number of bits set, if valid. Else 0.
# Validity check: We can precompute a "conflict table".
# Two points a, b are "compatible" if ... wait, compatibility is not pairwise. It's a property of the triple.
# So pairwise compatibility is not sufficient.

# Let's go back to **Simulated Annealing** but with a **much faster check**.
# The current check is O(m) Python loop. 
# Can we replace the loop with a single NumPy operation?
# We have `dks` (distances from k to S).
# We have `S_arr` (indices of S).
# We want to check if for any i, `dks[i]` is in `DIST[S_arr[i], S_arr]`.
# `DIST[S_arr, S_arr]` is the m x m distance matrix for S.
# We want to know if `dks[i]` is in `DIST[S_arr[i], :]`.
# This is a row-wise membership check.

# We can do this:
# `D_S = DIST[np.ix_(S_arr, S_arr)]`
# `dks_col = dks[:, None]`
# `matches = (D_S == dks_col) & (np.arange(m)[:, None] != np.arange(m)[None, :])` # Exclude self? No, d(k, a) > 0, d(a, a) = 0, so no self-match.
# `if np.any(matches): return False`

# This creates an m x m boolean matrix. For m=50, that's 2500 elements. Very fast.
# For m=100, 10000 elements. Still fast.
# For m=200, 40000 elements. Still fast in NumPy.
# The bottleneck is `DIST[np.ix_(S_arr, S_arr)]` which is a fancy indexing operation.
# But it's vectorized.

# Let's implement this vectorized check.

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    in_set = np.zeros(GRID_SIZE, dtype=bool)
    S = []
    best_set = []
    best_size = 0
    
    def is_valid_addition_vectorized(k):
        m = len(S)
        if m < 2:
            return True
        
        S_arr = np.array(S, dtype=np.int32)
        
        # 1. Check duplicate distances from k
        dks = DIST[k, S_arr]
        if len(dks) != len(np.unique(dks)):
            return False
        
        # 2. Check if d(k, a) is in distances from a to any other point in S
        # D_S is the distance matrix for points in S
        D_S = DIST[np.ix_(S_arr, S_arr)]
        
        # dks[i] should not be in D_S[i, :]
        # We can check this by:
        # Create a boolean matrix where D_S == dks[:, None]
        # But we need to exclude the diagonal? No, d(k, a) > 0, D_S[i, i] = 0.
        # So we just check if any element in row i of D_S equals dks[i].
        
        # `matches = (D_S == dks[:, None]).any(axis=1)`
        # `if np.any(matches): return False`
        
        # But wait, D_S[i, j] = d(S[i], S[j]).
        # We want to check if dks[i] = d(k, S[i]) is equal to d(S[i], S[j]) for any j.
        # So we check if dks[i] is in D_S[i, :].
        
        # `row_matches = (D_S == dks[:, None]).any(axis=1)`
        # `if np.any(row_matches): return False`
        
        # This is correct.
        
        # One optimization: we can avoid creating the full D_S matrix if m is small, 
        # but for m > 20, vectorized is likely faster than a Python loop.
        
        # Let's use a hybrid approach: if m < 10, use loop. Else, use vectorized.
        # Actually, let's just use vectorized for simplicity and speed.
        
        # However, `DIST[np.ix_(S_arr, S_arr)]` might be slow for large m due to memory access patterns.
        # Let's try it.
        
        # Alternative: use `np.isin`? No, that's for 1D.
        
        # Let's use the vectorized check.
        if (D_S == dks[:, None]).any(axis=1).any():
            return False
        
        return True
    
    def greedy_init():
        nonlocal S, in_set
        S = []
        in_set[:] = False
        
        indices = list(range(GRID_SIZE))
        random.shuffle(indices)
        
        for k in indices:
            if time.time() > deadline - 5:
                break
            if is_valid_addition_vectorized(k):
                S.append(k)
                in_set[k] = True
    
    def local_search():
        nonlocal S, in_set
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add random points
            for _ in range(500):
                if time.time() > deadline - 5:
                    break
                k = random.randint(0, GRID_SIZE - 1)
                if not in_set[k] and is_valid_addition_vectorized(k):
                    S.append(k)
                    in_set[k] = True
                    improved = True
            
            # Try swaps: remove a random point, try to add a random point
            if len(S) > 0:
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    idx = random.randint(0, len(S) - 1)
                    a = S[idx]
                    S.pop(idx)
                    in_set[a] = False
                    
                    added = False
                    for _ in range(20):
                        k = random.randint(0, GRID_SIZE - 1)
                        if not in_set[k] and is_valid_addition_vectorized(k):
                            S.append(k)
                            in_set[k] = True
                            added = True
                            improved = True
                            break
                    
                    if not added:
                        S.append(a)
                        in_set[a] = True
    
    # Main loop
    while time.time() < deadline - 10:
        greedy_init()
        local_search()
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
            # Update the global in_set and S to be the best set for potential continuation?
            # No, we restart from scratch each time.
            # But we can use the best set as a seed for the next greedy?
            # Let's try a "warm start": start with the best set, then try to add more.
            # This might be better than random restarts.
            
            # Let's implement a warm start strategy.
            # After finding a new best, we can use it as the initial set for the next iteration.
            # But `greedy_init` resets S. We need a `warm_start` function.
    
    # Let's modify the main loop to use warm starts.
    # 1. First, do a random greedy to get a good initial set.
    # 2. Then, iteratively try to improve it.
    
    # Actually, the current loop already does random restarts.
    # Let's just keep it simple and see if the vectorized check is fast enough.
    
    if len(best_set) == 0:
        # Fallback: return a small valid set if we didn't find anything
        best_set = [0]
    
    return [ALL_POINTS[i] for i in best_set]

# To ensure the function is ready to be called, we can add a small warm-up or just rely on the main loop.
# The current code looks good. The key change is the vectorized validity check.

# One more optimization: Precompute `DIST` as `int32` to save memory and potentially speed up ops.
# `DIST = (dx * dx + dy * dy).astype(np.int32)`
# But `dx` and `dy` are `int64`. The product will be `int64`.
# Max distance is (63^2 + 63^2) = 7938. This fits in `int32` easily.
# Let's cast to `int32` to save memory.
# However, `np.ix_` and comparisons might be faster with `int64`? Unlikely.
# Let's keep it as `int64` for now to avoid any potential issues.

# Let's also add a check to ensure we don't timeout in the middle of a greedy pass.
# The `time.time()` checks are already in place.

# Final check: The task requires the function name `construct` and it should return a list of (x, y) tuples.
# This is satisfied.

# Let's run a quick sanity check in our head.
# `is_valid_addition_vectorized` uses `DIST[np.ix_(S_arr, S_arr)]`.
# If `S` is empty, `S_arr` is empty, `DIST[np.ix_([], [])]` is a 0x0 array.
# `dks` is empty. `len(dks) == len(np.unique(dks))` is 0 == 0, True.
# `(D_S == dks[:, None])` is 0x0. `.any(axis=1)` is 0x0. `.any()` is False.
# So it returns True. Correct.

# If `S` has 1 element, `m=1`. `dks` has 1 element.
# `D_S` is 1x1. `D_S[0, 0] = 0`.
# `dks[0]` is the distance from k to the single point in S. It's > 0.
# `D_S == dks[:, None]` is `[[0 == dks[0]]]` which is `[[False]]`.
# `.any(axis=1)` is `[False]`. `.any()` is False.
# So it returns True. Correct.

# The logic seems sound.

# One potential issue: `random.shuffle(indices)` on a list of 4096 elements is fast.
# The loop `for k in indices` iterates 4096 times.
# Each `is_valid_addition_vectorized` call is O(m^2) in the worst case (for the matrix creation).
# As `m` grows, the cost increases.
# If `m` reaches ~100, the matrix is 100x100 = 10,000 elements.
# 4096 * 10,000 = 40 million operations. This should be fast in NumPy (< 0.1s).
# If `m` reaches ~200, 40,000 elements. 4096 * 40,000 = 160 million operations. Still fast (< 0.5s).
# So the greedy phase should be very fast.
# The local search phase will be slower because it tries many swaps, but it's also bounded by the time limit.

# This should significantly outperform the O(m) Python loop version.

# Let's finalize the code.
