"""Improved search for minimum maximal no-isosceles set using bitset MIS formulation."""
N = 16
import time
import random
import itertools
import numpy as np


def _build_conflict_graph():
    """Build conflict graph: edge between i,j if they form isosceles with some k."""
    n = N * N
    points = list(itertools.product(range(N), repeat=2))
    # For each pair (i,j), check if there exists k such that (i,j,k) is isosceles
    # Precompute: for each pair (i,j), the set of k that form isosceles with i,j
    # Two points i,j conflict if there EXISTS some k (not i,j) making isosceles
    # Actually, the constraint is: no THREE points form isosceles.
    # So the conflict is: a triple (i,j,k) is bad if isosceles.
    # For MIS formulation: we need to select points such that NO triple is isosceles.
    # This is NOT a standard graph problem (it's a 3-uniform hypergraph independent set).
    # 
    # Alternative: model as - for each point p, it conflicts with pair (a,b) if (p,a,b) is isosceles.
    # We can't directly use MIS on a graph.
    #
    # Better approach: Use the fact that if we fix two points, the third is constrained.
    # Let's use a different formulation:
    # A set S is "no-isosceles" iff for all distinct a,b,c in S, they don't form isosceles.
    # 
    # For branch-and-bound: we build the set incrementally. At each step, we decide
    # to include or exclude a point. The pruning is: if adding point p to current set
    # would create an isosceles with any two existing points, p cannot be added.
    #
    # Key insight for speed: precompute for each point p and each pair (a,b) whether
    # (p,a,b) is isosceles. Store as bitsets.
    
    # For each point p, precompute: conflict_pairs[p] = set of (a,b) pairs that with p form isosceles
    # Actually more useful: for each point p, and current set S, p can be added iff
    # for all a,b in S, (p,a,b) is NOT isosceles.
    # 
    # Let's precompute: iso_mask[p] = bitset of all points q such that... hmm, this is pair-dependent.
    # 
    # Better: precompute for each ordered pair (a,b), the set of points p where (p,a,b) is isosceles.
    # iso_with_pair[a][b] = set of p (as bitset)
    
    n_pts = n
    # Precompute distance squared between all pairs
    d2 = np.zeros((n_pts, n_pts), dtype=np.int32)
    for i in range(n_pts):
        for j in range(n_pts):
            d2[i, j] = (points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2
    
    # For each point p, compute which other points are at each distance
    # iso_with_pair[p] = for each pair (a,b) not involving p, whether (p,a,b) is isosceles
    # This is too large to precompute fully.
    
    # Alternative: for each point p, precompute dist_from_p = d2[p, :] 
    # Then (p,a,b) is isosceles iff d2[p,a]==d2[p,b] or d2[p,a]==d2[a,b] or d2[p,b]==d2[a,b]
    # The last two don't involve p directly in a simple way.
    
    # Let me precompute: for each p, and for each a != p, the set of b such that (p,a,b) is isosceles.
    # iso_conflict[p][a] = bitset of b where (p,a,b) is isosceles (b != p, b != a)
    
    iso_conflict = np.zeros((n_pts, n_pts), dtype=object)
    for p in range(n_pts):
        for a in range(n_pts):
            if p == a:
                iso_conflict[p, a] = 0
                continue
            mask = 0
            for b in range(n_pts):
                if b == p or b == a:
                    continue
                dp_a = d2[p, a]
                dp_b = d2[p, b]
                da_b = d2[a, b]
                if dp_a == dp_b or dp_a == da_b or dp_b == da_b:
                    mask |= (1 << b)
            iso_conflict[p, a] = mask
    
    return points, d2, iso_conflict


def _can_add_bitset(p, chosen_mask, iso_conflict, d2):
    """Check if point p can be added to set represented by chosen_mask."""
    # For each a in chosen, check if any b in chosen makes (p,a,b) isosceles
    # (p,a,b) is isosceles iff p is in iso_with_pair[a][b] OR b is in iso_with_pair[a][p] OR a is in iso_with_pair[b][p]
    # Using iso_conflict: (p,a,b) is iso iff b is in iso_conflict[p][a] OR a is in iso_conflict[p][b] OR p is in iso_conflict[a][b]
    # We need: for all a,b in chosen, (p,a,b) is NOT isosceles.
    # Equivalently: for all a in chosen, (chosen_mask & iso_conflict[p][a]) == 0 AND (a not in iso_conflict[p][?])
    # Wait: (p,a,b) is iso iff:
    #   d(p,a)==d(p,b) -> b in iso_conflict[p][a] (the d(p,a)==d(p,b) case)
    #   d(p,a)==d(a,b) -> this means a is equidistant from p and b... 
    #   Let me re-derive.
    # (p,a,b) is isosceles iff d(p,a)==d(p,b) OR d(p,a)==d(a,b) OR d(p,b)==d(a,b)
    # 
    # For fixed p and a:
    #   Case 1: d(p,a)==d(p,b) -> b is at same distance from p as a. 
    #   Case 2: d(p,a)==d(a,b) -> b is at distance d(p,a) from a.
    #   Case 3: d(p,b)==d(a,b) -> b is at distance d(a,b) from a, which equals d(p,b)... circular.
    #
    # Let me just use the precomputed iso_conflict[p][a] which contains all b where (p,a,b) is isosceles.
    # So: p can be added iff for all a in chosen: (chosen_mask & iso_conflict[p][a]) == 0
    # AND also need to check the other direction: for all a,b in chosen, p is not in iso_conflict[a][b]
    # But iso_conflict[a][b] contains p iff (a,b,p) is isosceles, which is the same as (p,a,b) is isosceles.
    # And (p,a,b) is isosceles iff b is in iso_conflict[p][a] (by symmetry of the condition).
    # So checking (chosen_mask & iso_conflict[p][a]) == 0 for all a in chosen should suffice.
    
    chosen_indices = [i for i in range(len(iso_conflict)) if (chosen_mask >> i) & 1]
    for a in chosen_indices:
        if (chosen_mask & iso_conflict[p][a]) != 0:
            return False
    return True


def _is_maximal_bitset(chosen_mask, all_mask, iso_conflict, d2):
    """Check if no unchosen point can be added."""
    n_pts = len(iso_conflict)
    for p in range(n_pts):
        if (chosen_mask >> p) & 1:
            continue
        if _can_add_bitset(p, chosen_mask, iso_conflict, d2):
            return False
    return True


def _greedy_construct(order, iso_conflict, d2):
    """Greedy construction following given order."""
    chosen_mask = 0
    for p in order:
        if not (chosen_mask >> p) & 1:
            if _can_add_bitset(p, chosen_mask, iso_conflict, d2):
                chosen_mask |= (1 << p)
    return chosen_mask


def _make_maximal(chosen_mask, all_mask, iso_conflict, d2):
    """Add points until maximal."""
    changed = True
    while changed:
        changed = False
        for p in range(all_mask.bit_length()):
            if (chosen_mask >> p) & 1:
                continue
            if _can_add_bitset(p, chosen_mask, iso_conflict, d2):
                chosen_mask |= (1 << p)
                changed = True
    return chosen_mask


def _branch_and_bound(points, d2, iso_conflict, all_mask, best_mask, deadline):
    """
    Branch and bound to find smaller maximal no-isosceles set.
    State: (chosen_mask, remaining_mask, excluded_mask)
    We try to find a set smaller than best.
    """
    n_pts = N * N
    best_len = bin(best_mask).count('1')
    best_result = [best_mask]
    nodes = [0]
    
    # Order points by some heuristic (e.g., degree in conflict structure)
    # Points that conflict with many others should be considered first
    point_order = list(range(n_pts))
    # Sort by "constraint power" - points that are in many iso_conflict sets
    def constraint_score(p):
        score = 0
        for a in range(n_pts):
            if a != p:
                score += bin(iso_conflict[p][a]).count('1')
        return score
    
    point_order.sort(key=lambda p: -constraint_score(p))
    
    def _search(chosen_mask, remaining_mask, depth):
        if time.time() > deadline:
            return
        nodes[0] += 1
        if nodes[0] % 10000 == 0 and nodes[0] > 0:
            pass  # could log
        
        chosen_count = bin(chosen_mask).count('1')
        remaining_count = bin(remaining_mask).count('1')
        
        # Pruning: even if we add all remaining, can we beat best?
        # We need the final set to be maximal AND smaller than best_len
        # Lower bound on final size: chosen_count (can't remove chosen)
        # Upper bound: chosen_count + remaining_count
        if chosen_count >= best_result[0] and bin(best_result[0]).count('1') <= best_len:
            return
        
        # If no remaining points, check if maximal
        if remaining_mask == 0:
            if _is_maximal_bitset(chosen_mask, all_mask, iso_conflict, d2):
                if bin(chosen_mask).count('1') < bin(best_result[0]).count('1'):
                    best_result[0] = chosen_mask
            return
        
        # Get next point to decide
        # Find lowest set bit in remaining_mask
        p = (remaining_mask & -remaining_mask).bit_length() - 1
        new_remaining = remaining_mask & ~(1 << p)
        
        # Option 1: include p if possible
        if _can_add_bitset(p, chosen_mask, iso_conflict, d2):
            _search(chosen_mask | (1 << p), new_remaining, depth + 1)
        
        # Option 2: exclude p
        _search(chosen_mask, new_remaining, depth + 1)
    
    _search(0, all_mask, 0)
    return best_result[0]


def _local_search_reduce(chosen_mask, all_mask, iso_conflict, d2, max_iters=5000, deadline=None):
    """Try removing each point and re-maximizing."""
    current = chosen_mask
    best = chosen_mask
    best_len = bin(current).count('1')
    n_pts = N * N
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for idx in range(n_pts):
            if deadline is not None and time.time() > deadline:
                break
            if not (current >> idx) & 1:
                continue
            # Try removing point idx
            trial = current & ~(1 << idx)
            # Check validity: no isosceles in remaining set
            valid = True
            chosen_indices = [i for i in range(n_pts) if (trial >> i) & 1]
            for i in range(len(chosen_indices)):
                for j in range(i + 1, len(chosen_indices)):
                    if (trial & iso_conflict[chosen_indices[i]][chosen_indices[j]]) != 0:
                        # Some point in trial conflicts with pair (i,j)
                        # But we need to check if the conflicting point is IN trial
                        conflict_pts = trial & iso_conflict[chosen_indices[i]][chosen_indices[j]]
                        if conflict_pts != 0:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                continue
            
            # Make maximal
            extended = _make_maximal(trial, all_mask, iso_conflict, d2)
            if bin(extended).count('1') < bin(current).count('1'):
                current = extended
                improved = True
                break
        
        if not improved:
            break
        
        if bin(current).count('1') < best_len:
            best = current
            best_len = bin(current).count('1')
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s limit
    
    all_points = list(itertools.product(range(N), repeat=2))
    n_pts = N * N
    all_mask = (1 << n_pts) - 1
    
    # Build conflict structure
    points, d2, iso_conflict = _build_conflict_graph()
    
    best_mask = None
    best_len = float('inf')
    
    # Phase 1: Quick greedy constructions
    orderings = [
        ("row_major", list(range(n_pts))),
        ("col_major", [i for i in range(n_pts) if True]),  # placeholder
        ("diag", sorted(range(n_pts), key=lambda i: (points[i][0] + points[i][1], points[i][0]))),
        ("checker", [i for i in range(n_pts) if (points[i][0] + points[i][1]) % 2 == 0] +
                   [i for i in range(n_pts) if (points[i][0] + points[i][1]) % 2 == 1]),
        ("anti_diag", sorted(range(n_pts), key=lambda i: (abs(points[i][0] - points[i][1]), points[i][0]))),
    ]
    
    for name, order in orderings:
        if time.time() > deadline:
            break
        chosen = _greedy_construct(order, iso_conflict, d2)
        extended = _make_maximal(chosen, all_mask, iso_conflict, d2)
        ext_len = bin(extended).count('1')
        if ext_len < best_len:
            best_mask = extended
            best_len = ext_len
    
    # Phase 2: Local search from best
    if best_mask is not None and time.time() < deadline - 30:
        local_best = _local_search_reduce(best_mask, all_mask, iso_conflict, d2,
                                           max_iters=5000, deadline=deadline - 30)
        local_len = bin(local_best).count('1')
        if local_len < best_len:
            best_mask = local_best
            best_len = local_len
    
    # Phase 3: Branch and bound (limited)
    if best_mask is not None and time.time() < deadline - 20:
        bt_result = _branch_and_bound(points, d2, iso_conflict, all_mask, best_mask, deadline - 20)
        bt_len = bin(bt_result).count('1')
        if bt_len < best_len:
            best_mask = bt_result
            best_len = bt_len
    
    # Phase 4: More local search
    if best_mask is not None and time.time() < deadline - 5:
        final = _local_search_reduce(best_mask, all_mask, iso_conflict, d2,
                                      max_iters=3000, deadline=deadline - 5)
        final_len = bin(final).count('1')
        if final_len < best_len:
            best_mask = final
            best_len = final_len
    
    # Convert back to list of points
    if best_mask is None:
        # Fallback: simple greedy
        best_mask = _make_maximal(0, all_mask, iso_conflict, d2)
    
    result = [points[i] for i in range(n_pts) if (best_mask >> i) & 1]
    return result
