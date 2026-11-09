import itertools
import time
import random
import sys

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _build_conflict_masks(all_points):
    """
    Build bitmasks for the conflict graph.
    conflict_mask[i] is a bitset of all points j that conflict with i.
    i and j conflict if there exists k such that {i, j, k} is isosceles.
    """
    n = len(all_points)
    # dist_matrix for fast lookup
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # conflict_mask[i] will be a Python int representing a bitset
    conflict_mask = [0] * n
    
    # For each pair (i, j), check if they conflict.
    # They conflict if there exists k such that dist(i,k) == dist(j,k) or dist(i,j) == dist(i,k) or dist(i,j) == dist(j,k).
    # Actually, the condition for {i,j,k} to be isosceles is:
    # d(i,j) == d(i,k) OR d(i,j) == d(j,k) OR d(i,k) == d(j,k)
    
    # Optimization: Group points by distance from each point?
    # For each i, create a dict: distance -> list of points j.
    # If j and k are in the same list for i, then d(i,j)==d(i,k), so {i,j,k} is isosceles.
    # Thus j and k conflict.
    # Also, if d(i,j) == d(j,k), then i and k conflict? No, that's specific to j.
    
    # Let's just iterate k for each pair (i,j) for simplicity first, as N=256 is small enough for O(N^3) precomputation if optimized?
    # 256^3 = 16 million. Python might be slow.
    # Better: For each i, group others by distance.
    # If j, k have same distance from i, then j, k conflict.
    # If d(i,j) == d(j,k), then i, k conflict.
    
    # Let's use the grouping method.
    # 1. For each i, find all pairs (j, k) with d(i,j) == d(i,k). Add edge (j, k).
    # 2. For each j, find all pairs (i, k) with d(j,i) == d(j,k). Add edge (i, k).
    #    This is the same as step 1 if we iterate over all centers.
    
    # So, iterating over all centers c:
    #   Group all other points by distance to c.
    #   For each group of points with same distance, all pairs in the group conflict.
    
    for c in range(n):
        dist_groups = {}
        for p in range(n):
            if p == c:
                continue
            d = dist_matrix[c][p]
            if d not in dist_groups:
                dist_groups[d] = []
            dist_groups[d].append(p)
        
        for points_in_group in dist_groups.values():
            # All pairs in this group conflict
            # We can add edges efficiently
            m = len(points_in_group)
            for idx_a in range(m):
                a = points_in_group[idx_a]
                # Add b to conflict_mask[a] and a to conflict_mask[b]
                # We can do this by setting bits
                for idx_b in range(idx_a + 1, m):
                    b = points_in_group[idx_b]
                    conflict_mask[a] |= (1 << b)
                    conflict_mask[b] |= (1 << a)
    
    # Note: This covers d(a,c) == d(b,c).
    # Does it cover d(a,b) == d(a,c)?
    # If d(a,b) == d(a,c), then in the group for center a, b and c are in the same group (distance d(a,b)).
    # So the pair (b, c) is added. This means b and c conflict.
    # Wait. If d(a,b) == d(a,c), the isosceles triangle is (a,b,c) with apex a.
    # The conflict is between b and c?
    # The constraint is: no 3 points form an isosceles triangle.
    # If we have b and c in our set, and we add a, we form an isosceles triangle.
    # So b and c are "conflicting" in the sense that they cannot both be in the set if a is also in the set?
    # No. The set S is no-isosceles if NO three points in S form an isosceles triangle.
    # If b, c, a are all in S, and d(a,b)==d(a,c), then S is invalid.
    # So, if we choose a, we cannot choose both b and c.
    # This is a 3-uniform constraint.
    # However, the "conflict graph" approach for MMIS usually implies a 2-uniform constraint.
    # Can we reduce this?
    # If we define conflict(i, j) = True if there exists ANY k such that {i, j, k} is isosceles.
    # Then if i and j are in S, they conflict. But wait.
    # If i and j conflict, it means there is SOME k.
    # If k is NOT in S, then {i, j, k} is not a subset of S, so S is still valid.
    # So a 2-uniform conflict graph is TOO RESTRICTIVE.
    # If i and j conflict only via k, and k is not in S, i and j can coexist.
    
    # Therefore, we CANNOT use a simple 2-uniform conflict graph for MMIS directly unless the conflict is intrinsic to the pair regardless of the third point.
    # Is that true?
    # If d(i,j) == d(i,k), the conflict depends on k.
    # So we must use the 3-uniform hypergraph or a different formulation.
    
    # Back to the original `_can_add` logic which is correct.
    # The speedup must come from better data structures and search strategy.
    
    return conflict_mask, dist_matrix

def _can_add(x, chosen_mask, dist_matrix):
    """
    Check if x can be added to the set represented by chosen_mask.
    chosen_mask is a bitset.
    """
    chosen_indices = [i for i in range(len(dist_matrix)) if (chosen_mask >> i) & 1]
    if not chosen_indices:
        return True
    
    # Check 1: Distances from x to all chosen must be distinct
    seen_distances = set()
    for p in chosen_indices:
        d = dist_matrix[x][p]
        if d in seen_distances:
            return False
        seen_distances.add(d)
    
    # Check 2: For each p in chosen, d(x,p) must not equal d(p,q) for any q in chosen, q!=p
    for p in chosen_indices:
        d_xp = dist_matrix[x][p]
        # Check if any other point q in chosen has d(p,q) == d_xp
        # We can precompute for each p, a map of distance -> bitmask of points at that distance
        # Let's assume dist_matrix is passed, we just iterate.
        for q in chosen_indices:
            if q == p:
                continue
            if dist_matrix[p][q] == d_xp:
                return False
    return True

def _is_maximal(chosen_mask, all_mask, dist_matrix):
    """
    Check if chosen_mask is maximal.
    """
    n = len(dist_matrix)
    for x in range(n):
        if (chosen_mask >> x) & 1:
            continue
        if _can_add(x, chosen_mask, dist_matrix):
            return False
    return True

def _greedy_construct(all_mask, dist_matrix, rng, heuristic='max_degree'):
    """
    Greedy construction.
    """
    n = len(dist_matrix)
    chosen_mask = 0
    available_mask = all_mask
    
    # Precompute degrees for heuristic?
    # For 'max_degree', we pick the point that conflicts with the most available points?
    # Conflict is not 2-uniform, so "degree" is tricky.
    # Let's just use random order or a simple heuristic: pick point that minimizes future options?
    # Or just random.
    
    order = [i for i in range(n) if (available_mask >> i) & 1]
    rng.shuffle(order)
    
    # To make it faster, we can check validity.
    for p in order:
        if (available_mask >> p) & 1: # Still available
            if _can_add(p, chosen_mask, dist_matrix):
                chosen_mask |= (1 << p)
                # Update available?
                # We don't have a simple "remove neighbors" step because conflicts are 3-uniform.
                # We just rely on _can_add in future steps.
    
    return chosen_mask

def _local_search_reduce(chosen_mask, all_mask, dist_matrix, rng, deadline):
    """
    Local search: try to remove 2 points and add 1 to reduce size.
    """
    s_mask = chosen_mask
    n = len(dist_matrix)
    improved = True
    
    # Get list of points in set
    s_list = [i for i in range(n) if (s_mask >> i) & 1]
    
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs to remove
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if time.time() > deadline:
                    break
                
                p, q = s_list[i], s_list[j]
                # Remove p and q
                base_mask = s_mask & ~(1 << p) & ~(1 << q)
                
                # Try to add a point r
                # We can prioritize points that are "close" or random
                candidates = [r for r in range(n) if not ((base_mask >> r) & 1)]
                rng.shuffle(candidates)
                
                for r in candidates:
                    if not _can_add(r, base_mask, dist_matrix):
                        continue
                    
                    candidate_mask = base_mask | (1 << r)
                    if _is_maximal(candidate_mask, all_mask, dist_matrix):
                        s_mask = candidate_mask
                        s_list = [i for i in range(n) if (s_mask >> i) & 1]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return s_mask

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    n = len(all_points)
    all_mask = (1 << n) - 1
    
    # Precompute dist_matrix
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    best_mask = None
    best_size = float('inf')
    rng = random.Random(42)
    
    while time.time() < _DEADLINE:
        # Greedy construction
        chosen_mask = _greedy_construct(all_mask, dist_matrix, rng)
        
        # Local search reduce
        pruned_mask = _local_search_reduce(chosen_mask, all_mask, dist_matrix, rng, _DEADLINE)
        
        # Update best
        size = bin(pruned_mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = pruned_mask
    
    if best_mask is None:
        best_mask = _greedy_construct(all_mask, dist_matrix, rng)
        
    # Convert mask to list of points
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(all_points[i])
            
    return result

# EVOLVE-BLOCK-END
