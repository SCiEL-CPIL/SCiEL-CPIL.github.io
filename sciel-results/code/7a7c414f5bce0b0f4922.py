import time
import random
import numpy as np

N = 64
GRID_SIZE = N * N
MAX_TIME = 300

# Pre-compute points
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
ALL_POINTS = [tuple(p) for p in points_arr]

xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy).astype(np.int32)  # shape (GRID_SIZE, GRID_SIZE)

# For the "density-driven" construction, we need to know how "conflicting" a point is.
# A simple heuristic: points with many neighbors at the same distance are "bad" centers for isosceles triangles.
# We can precompute a "badness" score for each point.
# badness[i] = number of pairs (j, k) such that d(i,j) == d(i,k).
# This is O(N^3) if done naively, but we can compute it per point in O(N^2 log N) or O(N^2) with counting.
# Let's precompute a "conflict_potential" for each point.
# Actually, a better heuristic for the greedy phase: 
# When adding point k to set S, the "cost" is the number of points in S that would become "blocked" or the number of new conflicts created.
# But for a fast initial construction, let's just use a deterministic ordering based on a precomputed "centrality" or "density" metric.
# Or, simpler: use a fixed deterministic order (e.g., spiral or sorted by distance to center) and greedily add if valid.
# The randomness in the previous approach led to variability. A deterministic spiral might find a consistent, good packing.

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME - 10  # 10s safety margin
    
    # --- Phase 1: Deterministic Greedy Construction ---
    # Create a spiral order starting from the center
    cx, cy = N // 2, N // 2
    # Generate indices in a spiral order
    # This is a heuristic to prioritize central points which might have more "room" or structure?
    # Actually, corner points have fewer neighbors, so they might be "safer" to add?
    # Let's try a deterministic order: sort by (x+y, x-x) or just use a fixed permutation.
    # Let's use a fixed random seed for the greedy order to ensure reproducibility but different from pure index order.
    # Or, better: use the "badness" metric to prioritize points that are "less likely" to form isosceles triangles.
    
    # Precompute "isosceles potential" for each point:
    # For each point i, count how many pairs (j, k) satisfy d(i,j) == d(i,k).
    # This is the number of isosceles triangles with vertex i.
    # Points with fewer such pairs are "better" vertices.
    # Let's compute this.
    
    # This computation is O(N^3) if naive, but we can do it per point in O(N^2).
    # Total O(N^3) = 4096^3 which is too slow.
    # We can approximate or just use a simple heuristic.
    # Let's just use a fixed spiral order. It's fast and deterministic.
    
    # Generate spiral indices
    def spiral_order(n):
        indices = []
        top, bottom, left, right = 0, n - 1, 0, n - 1
        while top <= bottom and left <= right:
            for i in range(left, right + 1):
                indices.append(top * n + i)
            top += 1
            for i in range(top, bottom + 1):
                indices.append(i * n + right)
            right -= 1
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    indices.append(bottom * n + i)
                bottom -= 1
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    indices.append(i * n + left)
                left += 1
        return indices
    
    spiral_indices = spiral_order(N)
    # Reverse it to start from outside? Or inside?
    # Let's try starting from outside (corners) as they have fewer constraints.
    # Actually, let's just use the spiral order as is.
    
    in_set = np.zeros(GRID_SIZE, dtype=bool)
    S = []
    
    def is_valid_addition(k, S_arr):
        m = len(S_arr)
        if m < 2:
            return True
        
        # 1. Check duplicate distances from k to S
        dks = DIST[k, S_arr]
        if len(dks) != len(np.unique(dks)):
            return False
        
        # 2. Check if d(k, a) is in distances from a to any other point in S
        # D_S is the distance matrix for points in S
        D_S = DIST[np.ix_(S_arr, S_arr)]
        
        # dks[i] should not be in D_S[i, :]
        if (D_S == dks[:, None]).any(axis=1).any():
            return False
        
        return True

    # Greedy construction
    for k in spiral_indices:
        if time.time() > deadline - 60:  # Leave 60s for refinement
            break
        if not in_set[k]:
            S_arr = np.array(S, dtype=np.int32) if S else np.array([], dtype=np.int32)
            if is_valid_addition(k, S_arr):
                S.append(k)
                in_set[k] = True
    
    best_set = S[:]
    best_size = len(S)
    
    # --- Phase 2: Simulated Annealing ---
    # We will perform swaps to try to increase the size.
    # State: S (list of indices), in_set (bool array)
    # Move: Remove a random point a from S, add a random point b not in S.
    # If |S| increases, accept.
    # If |S| stays same (neutral), accept with probability exp(-delta/T). Here delta=0, so always accept?
    # No, we want to escape local minima. Neutral moves allow wandering.
    # If |S| decreases, reject (we want to maximize size).
    
    # To make this efficient, we need a fast validity check for the new set.
    # Removing a point and adding a point changes the set.
    # We can't easily update the validity of all other points incrementally without complex bookkeeping.
    # So we will re-check validity of the *added* point against the *rest* of the set.
    # The rest of the set is already valid.
    
    S = best_set[:]
    in_set = np.zeros(GRID_SIZE, dtype=bool)
    for k in S:
        in_set[k] = True
    
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if len(S) == 0:
            break
            
        # Pick a random point to remove
        idx = random.randint(0, len(S) - 1)
        a = S[idx]
        S.pop(idx)
        in_set[a] = False
        
        # Try to add a random point
        added = False
        # Try a few candidates
        for _ in range(10):
            b = random.randint(0, GRID_SIZE - 1)
            if not in_set[b]:
                S_arr = np.array(S, dtype=np.int32) if S else np.array([], dtype=np.int32)
                if is_valid_addition(b, S_arr):
                    S.append(b)
                    in_set[b] = True
                    added = True
                    break
        
        if added:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        else:
            # Restore a
            S.append(a)
            in_set[a] = True
        
        T *= cooling_rate
        if T < 0.1:
            T = 0.1
            
    return [ALL_POINTS[i] for i in best_set]

# To ensure the function is ready to be called, we can add a small warm-up or just rely on the main loop.
# The current code looks good. The key change is the deterministic greedy start and SA refinement.
# One optimization: The `is_valid_addition` creates `S_arr` and `D_S` every time.
# In the SA loop, `S` changes slightly. We could cache `S_arr` but it's updated every step.
# For m ~ 10-20, this is fast enough.
# If m grows larger, we might need to optimize the check further, but 300s is a lot of time.
# The greedy phase is very fast (deterministic).
# The SA phase runs for the remaining time.

# Let's double check the `is_valid_addition` logic.
# It checks if k can be added to S.
# 1. No two points in S have the same distance to k.
# 2. For no a in S, is d(k, a) equal to d(a, b) for any b in S.
# This is correct.

# The spiral order might not be the best, but it's a structural change from random shuffle.
# It provides a consistent starting point for the SA.
# The SA allows escaping local minima by accepting neutral moves (swaps that don't change size).
# This is a different search method (SA vs Greedy+Local Search) and a different initialization (Spiral vs Random).

# Final check on the code structure.
# `construct` returns a list of tuples.
# Time limit is handled.
# NumPy is used.

# One potential issue: `np.array(S, dtype=np.int32)` inside the loop is expensive if called millions of times.
# We can maintain `S_arr` as a numpy array and update it.
# But `S` is a list. Converting list to array is O(m).
# In the SA loop, we do this 10 times per iteration.
# If we run 1 million iterations, that's 10 million conversions.
# For m=10, this is 10 million * 10 = 100 million operations. This might be slow in Python.
# Let's optimize the SA loop to avoid repeated array creation.

# Revised SA Loop Strategy:
# Maintain `S` as a list.
# Maintain `S_arr` as a numpy array.
# When removing a, update `S` and `S_arr`.
# When adding b, update `S` and `S_arr`.
# This is tricky because `S_arr` indices change if we remove from the middle.
# Easier: Just use the list for the validity check, but convert to array only when needed.
# Actually, `is_valid_addition` takes `S_arr`.
# We can pass `S` (list) and convert inside `is_valid_addition`? No, that's what we're doing now.
# Let's try to make `is_valid_addition` accept a list and convert once.
# Or, better: In the SA loop, we only check validity for the *new* point `b` against the *current* `S`.
# The current `S` is valid.
# We can pre-compute the distance matrix for `S`? No, it changes.
# Let's stick to the current implementation but optimize the conversion.
# `np.array(S, dtype=np.int32)` is quite fast in NumPy.
# Let's assume it's fast enough for 300s.

# If it's too slow, we can reduce the number of SA iterations or the number of candidates per step.
# But let's try this first.

# One more thing: The `is_valid_addition` function uses `DIST[np.ix_(S_arr, S_arr)]`.
# This creates a new array every time.
# For m=10, this is 100 elements.
# For m=100, this is 10,000 elements.
# This is fast.

# Let's finalize.

# To be safe, let's add a timeout check inside the SA loop more frequently.
# The current check is at the start of the while loop.
# If one iteration takes long, we might exceed the deadline.
# But each iteration is O(m^2) which is small.

# Let's run it.
