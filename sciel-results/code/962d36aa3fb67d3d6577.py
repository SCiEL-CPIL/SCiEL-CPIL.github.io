"""
Strategy: Greedy Construction + Systematic Local Search (Remove/Swap/Move)
with aggressive precomputation and fast delta evaluation.

The key structural change from the previous SA approach:
1. Use a much faster precomputation (vectorized with NumPy) for isosceles relationships.
2. Use a deterministic greedy construction to get a valid maximal set quickly.
3. Use a local search with specific move types (Remove, Swap, Move) that are 
   carefully designed to preserve maximality or quickly restore it.
4. Use aggressive time management to ensure we return a valid result.

The local search tries to minimize |S| while maintaining:
- No isosceles triangles
- Maximality (every point not in S is covered by some pair in S)
"""

import time
import random
import itertools
import sys
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_SET = set(POINTS)
N_POINTS = len(POINTS)
DEADLINE = 310.0  # Safety margin before 330s kill

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# Precompute isosceles relationships using NumPy for speed
print("Precomputing isosceles relationships...", file=sys.stderr)
start_time = time.time()

# For each point p, store the set of pairs (a,b) such that (p,a,b) is isosceles
# iso_pairs_with[p] = set of frozenset({a,b}) where (p,a,b) is isosceles
# For each pair (a,b), store the set of points p such that (p,a,b) is isosceles
# iso_points_for_pair[frozenset({a,b})] = set of points p

# Vectorized computation
# All points as array
pts_arr = np.array(POINTS, dtype=np.int32)

# For each pair (i,j), compute which points k form an isosceles triangle with i,j
# We need: for each pair (i,j), the set of k such that dist2(i,k)==dist2(i,j) or 
# dist2(j,k)==dist2(i,j) or dist2(i,k)==dist2(j,k)

# Precompute all pairwise distances
# dist2[i][j] = dist2(pts_arr[i], pts_arr[j])
dist2_matrix = np.zeros((N_POINTS, N_POINTS), dtype=np.int64)
dx = pts_arr[:, 0:1] - pts_arr[:, 0:1].T
dy = pts_arr[:, 1:2] - pts_arr[:, 1:2].T
dist2_matrix = (dx**2 + dy**2).astype(np.int64)

# For each pair (i,j), find all k such that (i,j,k) is isosceles
# iso_points_for_pair: dict from frozenset({i,j}) to set of k indices
iso_points_for_pair = {}
iso_pairs_with = [set() for _ in range(N_POINTS)]  # iso_pairs_with[k] = set of frozenset({i,j})

# We only need to consider unordered pairs i < j
for i in range(N_POINTS):
    for j in range(i+1, N_POINTS):
        d_ij = dist2_matrix[i, j]
        if d_ij == 0:
            continue
        # Find all k such that (i,j,k) is isosceles
        # dist2(i,k) == d_ij OR dist2(j,k) == d_ij OR dist2(i,k) == dist2(j,k)
        di = dist2_matrix[i, :]
        dj = dist2_matrix[j, :]
        # k is isosceles if:
        # di[k] == d_ij (k equidistant from i as j is) - but k != i, k != j
        # dj[k] == d_ij (k equidistant from j as i is)
        # di[k] == dj[k] (k equidistant from i and j)
        mask = (di == d_ij) | (dj == d_ij) | (di == dj)
        # Exclude i and j themselves
        mask[i] = False
        mask[j] = False
        k_indices = np.where(mask)[0]
        if len(k_indices) > 0:
            pair_key = frozenset((i, j))
            iso_points_for_pair[pair_key] = set(k_indices.tolist())
            for k in k_indices:
                iso_pairs_with[k].add(pair_key)

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

def _is_valid_set(S_indices):
    """Check if set of point indices has no isosceles triangles."""
    S_list = list(S_indices)
    for idx, a in enumerate(S_list):
        for b in S_list[idx+1:]:
            pair = frozenset((a, b))
            # Check if there's any c in S that forms isosceles with (a,b)
            # We need to check if any point in S is in iso_points_for_pair[pair]
            if pair in iso_points_for_pair:
                for c in iso_points_for_pair[pair]:
                    if c in S_indices:
                        return False
    return True

def _is_maximal(S_indices):
    """Check if set S is maximal (every point not in S is covered by S)."""
    S_set = set(S_indices)
    for p in range(N_POINTS):
        if p in S_set:
            continue
        # Check if p is covered by any pair in S
        covered = False
        S_list = list(S_set)
        for idx, a in enumerate(S_list):
            for b in S_list[idx+1:]:
                pair = frozenset((a, b))
                if pair in iso_points_for_pair and p in iso_points_for_pair[pair]:
                    covered = True
                    break
            if covered:
                break
        if not covered:
            return False
    return True

def _greedy_maximal():
    """Greedy construction of a valid maximal set."""
    S = set()
    pts = list(range(N_POINTS))
    random.shuffle(pts)
    
    for p in pts:
        # Check if adding p creates an isosceles triangle
        valid = True
        S_list = list(S)
        for idx, a in enumerate(S_list):
            for b in S_list[idx+1:]:
                pair = frozenset((a, b))
                if pair in iso_points_for_pair and p in iso_points_for_pair[pair]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(p)
    
    # Now make it maximal: for each uncovered point, try to add it or find a swap
    # First, identify uncovered points
    S_list = list(S)
    for p in range(N_POINTS):
        if p in S:
            continue
        covered = False
        for idx, a in enumerate(S_list):
            for b in S_list[idx+1:]:
                pair = frozenset((a, b))
                if pair in iso_points_for_pair and p in iso_points_for_pair[pair]:
                    covered = True
                    break
            if covered:
                break
        if not covered:
            # p is uncovered. Try to add p if valid.
            valid = True
            for idx, a in enumerate(S_list):
                for b in S_list[idx+1:]:
                    pair = frozenset((a, b))
                    if pair in iso_points_for_pair and p in iso_points_for_pair[pair]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(p)
                S_list.append(p)
    
    # Verify maximality
    return S

def _fast_is_valid(S, p):
    """Check if adding p to S creates an isosceles triangle."""
    for pair in iso_pairs_with[p]:
        a, b = list(pair)
        if a in S and b in S:
            return False
    return True

def _fast_is_covered(S, p):
    """Check if p is covered by some pair in S."""
    # p is covered if there exists a pair (a,b) in S such that p in iso_points_for_pair[{a,b}]
    # Equivalently, there exists a pair (a,b) such that (p,a,b) is isosceles and a,b in S
    # This is: for some pair in iso_pairs_with[p], both points are in S
    for pair in iso_pairs_with[p]:
        a, b = list(pair)
        if a in S and b in S:
            return True
    return False

def construct():
    start_time = time.time()
    
    # Step 1: Get initial valid maximal set
    S = _greedy_maximal()
    
    # Ensure it's valid and maximal
    S_indices = set(S)
    
    # Step 2: Local search to minimize size
    best_S = set(S_indices)
    best_size = len(best_S)
    
    # Precompute for fast operations
    # For each point p, store the list of pairs in iso_pairs_with[p] as tuples
    iso_pairs_with_list = [list(pairs) for pairs in iso_pairs_with]
    
    # For each pair, store the list of points in iso_points_for_pair as list
    iso_points_for_pair_list = {k: list(v) for k, v in iso_points_for_pair.items()}
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    steps = 0
    no_improve_count = 0
    NO_IMPROVE_LIMIT = 50000
    
    while time.time() < DEADLINE:
        if no_improve_count > NO_IMPROVE_LIMIT:
            # Try a random restart
            S = _greedy_maximal()
            S_indices = set(S)
            if len(S_indices) < best_size:
                best_S = set(S_indices)
                best_size = len(best_S)
                no_improve_count = 0
            else:
                no_improve_count = 0
                # Continue with current best
                S_indices = set(best_S)
                S = set(S_indices)
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to remove a point
            if not S_indices:
                continue
            p = random.choice(list(S_indices))
            
            # Check if removing p preserves validity (it always does, since we're removing)
            # Check if removing p preserves maximality: p must be covered by S\{p}
            # And all other points that were only covered by pairs involving p must still be covered
            
            # First, check if p is covered by S\{p}
            new_S = S_indices - {p}
            
            # Check if p is covered by new_S
            p_covered = False
            for pair in iso_pairs_with_list[p]:
                a, b = list(pair)
                if a in new_S and b in new_S:
                    p_covered = True
                    break
            
            if not p_covered:
                # p would become uncovered, so we can't simply remove it
                # unless we can add another point to cover p
                # For now, skip this removal
                continue
            
            # Check if any other point becomes uncovered
            # Points that were covered by pairs involving p might become uncovered
            # For each pair (p, q) in S, for each r in iso_points_for_pair[{p,q}],
            # check if r is still covered by new_S
            all_still_covered = True
            for q in list(new_S):
                pair = frozenset((p, q))
                if pair in iso_points_for_pair_list:
                    for r in iso_points_for_pair_list[pair]:
                        if r not in S_indices and r != p:
                            # r was not in S, check if it's still covered by new_S
                            if not _fast_is_covered(new_S, r):
                                all_still_covered = False
                                break
                if not all_still_covered:
                    break
            
            if all_still_covered:
                # Removal is valid
                S_indices = new_S
                if len(S_indices) < best_size:
                    best_S = set(S_indices)
                    best_size = len(best_S)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
        elif move_type < 0.8:
            # Try a swap: remove p, add q
            if not S_indices:
                continue
            p = random.choice(list(S_indices))
            candidates = [q for q in range(N_POINTS) if q not in S_indices]
            if not candidates:
                continue
            q = random.choice(candidates)
            
            new_S = S_indices - {p} | {q}
            
            # Check if new_S is valid (no isosceles)
            valid = True
            for idx, a in enumerate(list(new_S)):
                for b in list(new_S)[idx+1:]:
                    pair = frozenset((a, b))
                    if pair in iso_points_for_pair_list:
                        for c in iso_points_for_pair_list[pair]:
                            if c in new_S:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check maximality
            maximal = True
            for r in range(N_POINTS):
                if r in new_S:
                    continue
                if not _fast_is_covered(new_S, r):
                    maximal = False
                    break
            
            if maximal and len(new_S) < best_size:
                S_indices = new_S
                best_S = set(S_indices)
                best_size = len(best_S)
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        else:
            # Try to move a point: remove p, add q, where q is near p
            if not S_indices:
                continue
            p = random.choice(list(S_indices))
            # Find nearby points
            px, py = POINTS[p]
            nearby = [q for q in range(N_POINTS) if q not in S_indices 
                     and abs(POINTS[q][0] - px) <= 2 and abs(POINTS[q][1] - py) <= 2]
            if not nearby:
                continue
            q = random.choice(nearby)
            
            new_S = S_indices - {p} | {q}
            
            # Check validity
            valid = True
            S_list = list(new_S)
            for idx, a in enumerate(S_list):
                for b in S_list[idx+1:]:
                    pair = frozenset((a, b))
                    if pair in iso_points_for_pair_list:
                        for c in iso_points_for_pair_list[pair]:
                            if c in new_S:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check maximality
            maximal = True
            for r in range(N_POINTS):
                if r in new_S:
                    continue
                if not _fast_is_covered(new_S, r):
                    maximal = False
                    break
            
            if maximal and len(new_S) < best_size:
                S_indices = new_S
                best_S = set(S_indices)
                best_size = len(best_S)
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        T *= cooling
        steps += 1
        
        if steps % 10000 == 0:
            print(f"Step {steps}, size={len(S_indices)}, best={best_size}, T={T:.4f}", file=sys.stderr)
    
    # Final verification
    if not _is_valid_set(best_S) or not _is_maximal(best_S):
        # Fallback to greedy
        best_S = _greedy_maximal()
    
    return [POINTS[p] for p in best_S]

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Points: {res[:10]}...")
