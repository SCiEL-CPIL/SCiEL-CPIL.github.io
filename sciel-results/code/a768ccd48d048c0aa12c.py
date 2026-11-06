import time
import random
import itertools
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_SET = set(POINTS)
N_POINTS = len(POINTS)
DEADLINE = 310.0

# Precompute isosceles relationships
# For each point p, we want to know which pairs {a, b} form an isosceles triangle with p.
# iso_pairs_with[p] = set of frozenset({a, b})
print("Precomputing isosceles relationships...", file=sys.stderr)
start_time = time.time()

pts_arr = np.array(POINTS, dtype=np.int32)
dist2_matrix = np.zeros((N_POINTS, N_POINTS), dtype=np.int64)
dx = pts_arr[:, 0:1] - pts_arr[:, 0:1].T
dy = pts_arr[:, 1:2] - pts_arr[:, 1:2].T
dist2_matrix = (dx**2 + dy**2).astype(np.int64)

iso_pairs_with = [set() for _ in range(N_POINTS)]

for i in range(N_POINTS):
    for j in range(i+1, N_POINTS):
        d_ij = dist2_matrix[i, j]
        if d_ij == 0:
            continue
        di = dist2_matrix[i, :]
        dj = dist2_matrix[j, :]
        # p is the apex if dist(p, i) == dist(p, j)
        # i is the apex if dist(i, p) == dist(i, j) => dist(p, i) == d_ij
        # j is the apex if dist(j, p) == dist(j, i) => dist(p, j) == d_ij
        
        # Mask for p such that dist(p, i) == dist(p, j)
        mask_apex_p = (di == dj)
        mask_apex_i = (dj == d_ij)
        mask_apex_j = (di == d_ij)
        
        mask = mask_apex_p | mask_apex_i | mask_apex_j
        mask[i] = False
        mask[j] = False
        
        k_indices = np.where(mask)[0]
        if len(k_indices) > 0:
            pair_key = frozenset((i, j))
            for k in k_indices:
                iso_pairs_with[k].add(pair_key)

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

# Convert to lists for faster iteration
iso_pairs_with_list = [list(s) for s in iso_pairs_with]

def _is_valid_set(S_indices):
    """Check if set of point indices has no isosceles triangles."""
    S_set = set(S_indices)
    for p in S_set:
        for pair in iso_pairs_with_list[p]:
            a, b = list(pair)
            if a in S_set and b in S_set:
                return False
    return True

def _is_maximal(S_indices):
    """Check if set S is maximal."""
    S_set = set(S_indices)
    for p in range(N_POINTS):
        if p in S_set:
            continue
        covered = False
        for pair in iso_pairs_with_list[p]:
            a, b = list(pair)
            if a in S_set and b in S_set:
                covered = True
                break
        if not covered:
            return False
    return True

def _try_remove(S, p):
    """
    Try to remove p from S. Return new S if valid and maximal, else None.
    """
    new_S = S - {p}
    
    # 1. Check if p is covered by new_S
    p_covered = False
    for pair in iso_pairs_with_list[p]:
        a, b = list(pair)
        if a in new_S and b in new_S:
            p_covered = True
            break
    if not p_covered:
        return None
    
    # 2. Check if any other point q (not in new_S) is still covered by new_S
    # We only need to check points q that were covered by pairs involving p, 
    # because if q was covered by a pair not involving p, it's still covered.
    # However, checking all q is safer and not too expensive.
    for q in range(N_POINTS):
        if q in new_S:
            continue
        covered = False
        for pair in iso_pairs_with_list[q]:
            a, b = list(pair)
            if a in new_S and b in new_S:
                covered = True
                break
        if not covered:
            return None
            
    return new_S

def _construct_impl():
    # Step 1: Initial construction
    # We start with an empty set. We want to build a maximal set.
    # A point p is "active" if it is not covered by the current set S.
    # If S is empty, all points are active.
    # We want to select a subset S such that all points are covered by S, and S is valid.
    
    S = set()
    
    # We use a greedy approach to pick points.
    # A point p is a good candidate if it covers many currently active points.
    # Active points are those not in S and not covered by S.
    
    # Actually, let's think about the "Maximal" constraint.
    # S is maximal iff for every p not in S, p is covered by S.
    # This is equivalent to: The set of points NOT in S is a subset of the points covered by S.
    # Let U = All points.
    # Let C(S) = { p in U | p is covered by S }.
    # Maximal iff U \ S is a subset of C(S).
    # This implies S is a "dominating set" in the hypergraph where hyperedges are isosceles triangles?
    # Not exactly. The relation is "p is covered by pair {a,b} in S".
    
    # Let's use a simple greedy:
    # 1. Start with S = empty.
    # 2. While there exists a point p not in S that is not covered by S:
    #    - Such a p makes S non-maximal.
    #    - We must add some point to S to cover p? No, adding a point q to S might cover p.
    #    - Or we can add p itself? If we add p, p is in S, so it doesn't need to be covered.
    #    - But adding p might violate validity.
    
    # Better approach:
    # We want to find a minimum size S such that:
    # 1. S is valid (no isosceles triangles).
    # 2. For all p not in S, p is covered by S.
    
    # This is a hitting set / set cover problem with conflicts.
    
    # Greedy:
    # Start with S = empty.
    # Loop:
    #   Find all points p not in S that are NOT covered by S.
    #   If none, S is maximal. Stop.
    #   For each such p, we need to "fix" the non-maximality.
    #   Options:
    #     a) Add p to S. (If valid)
    #     b) Add some other point q to S such that q helps cover p?
    #        Actually, adding q to S only helps cover p if {q, r} in S forms an isosceles with p for some r in S.
    #        This is complex.
        
    # Simpler Greedy:
    # 1. Pick a random point p.
    # 2. If adding p to S keeps S valid, add it.
    # 3. Repeat until no more points can be added?
    #    This gives a maximal valid set (by validity), but not necessarily maximal by the "no extension" definition?
    #    Wait. "Maximal no-isosceles set" usually means:
    #    1. No isosceles triangles in S.
    #    2. For any p not in S, S U {p} contains an isosceles triangle.
    #    This is exactly what we are checking.
    #    If we greedily add points as long as validity is preserved, we reach a state where no more points can be added without violating validity.
    #    This implies that for any p not in S, S U {p} is invalid, i.e., contains an isosceles triangle.
    #    So a maximal valid set (by inclusion) is a maximal no-isosceles set.
    
    # So, we just need to find a MINIMAL (by size) MAXIMAL (by inclusion) valid set.
    
    # Strategy:
    # 1. Generate a random maximal valid set S0.
    #    - Start with S = empty.
    #    - Shuffle points.
    #    - For each p, if S U {p} is valid, add p.
    # 2. Local search to reduce size:
    #    - Try to remove points one by one.
    #    - If S \ {p} is still maximal (i.e., valid and covers all outside points), remove p.
    
    S = set()
    pts = list(range(N_POINTS))
    random.shuffle(pts)
    
    for p in pts:
        # Check if adding p is valid
        valid = True
        for pair in iso_pairs_with_list[p]:
            a, b = list(pair)
            if a in S and b in S:
                valid = False
                break
        if valid:
            S.add(p)
            
    return S

def construct():
    start_time = time.time()
    
    # Step 1: Get initial valid maximal set
    S = _construct_impl()
    
    best_S = set(S)
    best_size = len(best_S)
    
    # Step 2: Local search
    # We will repeatedly try to remove points.
    # To avoid getting stuck, we will also try random swaps if stuck.
    
    while time.time() < DEADLINE:
        improved = False
        # Randomize order of points to try
        points_to_try = list(S)
        random.shuffle(points_to_try)
        
        for p in points_to_try:
            if time.time() > DEADLINE:
                break
            if p not in S:
                continue
                
            new_S = _try_remove(S, p)
            if new_S is not None:
                S = new_S
                improved = True
                if len(S) < best_size:
                    best_S = set(S)
                    best_size = len(best_S)
        
        if not improved:
            # Stuck at a local minimum for removals.
            # Try a random swap to escape?
            # Or just restart with a new greedy?
            # Let's try a few random swaps.
            escaped = False
            for _ in range(50):
                if time.time() > DEADLINE:
                    break
                if len(S) < 2:
                    break
                p = random.choice(list(S))
                # Find a candidate q to swap with
                # q should not be in S
                candidates = [q for q in range(N_POINTS) if q not in S]
                if not candidates:
                    break
                q = random.choice(candidates)
                
                # Check if S' = S \ {p} U {q} is valid
                S_temp = S - {p} | {q}
                valid = True
                for pair in iso_pairs_with_list[q]:
                    a, b = list(pair)
                    if a in S_temp and b in S_temp:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check if S' is maximal
                maximal = True
                for r in range(N_POINTS):
                    if r in S_temp:
                        continue
                    covered = False
                    for pair in iso_pairs_with_list[r]:
                        a, b = list(pair)
                        if a in S_temp and b in S_temp:
                            covered = True
                            break
                    if not covered:
                        maximal = False
                        break
                
                if maximal:
                    # Swap accepted
                    S = S_temp
                    escaped = True
                    # Try to remove again immediately
                    # (The loop will restart)
                    break
            
            if not escaped:
                # Try a new greedy restart
                S_new = _construct_impl()
                if len(S_new) < len(S):
                    S = S_new
                    if len(S) < best_size:
                        best_S = set(S)
                        best_size = len(best_S)
                else:
                    # If no improvement after several restarts, break?
                    # For now, just keep trying, but maybe add a counter.
                    pass
                    
    # Final verification
    if not _is_valid_set(best_S) or not _is_maximal(best_S):
        # Fallback to greedy
        best_S = _construct_impl()
    
    return [POINTS[p] for p in best_S]

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Points: {res[:10]}...")
