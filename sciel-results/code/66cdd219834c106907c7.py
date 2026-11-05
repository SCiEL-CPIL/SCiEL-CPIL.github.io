import random
import time
import itertools
import math
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# For each point p, compute which other points q are "blocked" by p.
# p blocks q if there exists some r such that {p, q, r} is isosceles.
# Equivalently: q cannot be added to a set S if p in S and there exists r in S with {p,q,r} isosceles.
# But for maximality: q is blocked by S if for all valid extensions, adding q would create isosceles.
# Actually: q is NOT addable to S iff exists u,v in S such that {q,u,v} is isosceles.
# So q is blocked by S iff the set of "bad pairs" for q intersects S in at least one pair.

# Let's define: for each point p, BLOCKED_BY[p] = set of points q such that p and q together block some third point.
# More precisely: p "covers" q in the blocking sense if p is in S and q cannot be added.
# q cannot be added to S iff exists u,v in S with {q,u,v} isosceles.
# So if p in S, p helps block q if there exists u in S\{p} such that {q,p,u} is isosceles.

# Let's precompute: for each pair (p, q) with p != q, 
#   is_blocked_pair[p][q] = True if there exists r such that {p,q,r} is isosceles
# Then: q is blocked by S iff exists p in S such that is_blocked_pair[p][q] AND exists another constraint...

# Actually, the correct formulation:
# S is a no-isosceles set iff for all distinct p,q,r in S, {p,q,r} is NOT isosceles.
# S is maximal iff for all q not in S, there exist u,v in S such that {q,u,v} IS isosceles.

# Let's precompute for each q, the set of pairs (u,v) with u<v in [0,N_PTS) such that {q,u,v} is isosceles.
# BAD_PAIRS[q] = set of (u,v) with u < v, {q,u,v} is isosceles.

BAD_PAIRS = [set() for _ in range(N_PTS)]

def _precompute_bad_pairs():
    i_idx, j_idx = np.triu_indices(N_PTS, k=1)
    for p in range(N_PTS):
        dp = dist2[p, :]
        dp_i = dp[i_idx]
        dp_j = dp[j_idx]
        dij = dist2[i_idx, j_idx]
        cond = (dp_i == dp_j) | (dp_i == dij) | (dp_j == dij)
        bad_indices = np.where(cond)[0]
        for k in bad_indices:
            u, v = i_idx[k], j_idx[k]
            if u > v:
                u, v = v, u
            BAD_PAIRS[p].add((u, v))

_start_time = time.time()
_precompute_bad_pairs()

# For efficient checking, convert BAD_PAIRS to a more efficient structure.
# For each point q, we want to quickly check if any pair in BAD_PAIRS[q] is a subset of S.
# We'll keep it as a set of tuples for now.

def _can_add(p, S_set):
    """Check if p can be added to S_set (represented as a set of indices)."""
    for u, v in BAD_PAIRS[p]:
        if u in S_set and v in S_set:
            return False
    return True

def _is_maximal(S_set):
    """Check if S_set is maximal."""
    for p in range(N_PTS):
        if p not in S_set:
            if _can_add(p, S_set):
                return False
    return True

def _is_valid_no_isosceles(S_set):
    """Check that S_set is a valid no-isosceles set."""
    S_list = list(S_set)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a, b = S_list[i], S_list[j]
            for k in range(j+1, len(S_list)):
                c = S_list[k]
                # Check if {a,b,c} is isosceles
                da = dist2[a, b]
                db = dist2[b, c]
                dc = dist2[a, c]
                if da == db or db == dc or da == dc:
                    return False
    return True

def _backtrack_search(deadline, max_size=None):
    """
    Systematic backtracking search for minimum maximal no-isosceles set.
    Uses constraint propagation and pruning.
    """
    best_set = None
    best_size = max_size if max_size else float('inf')
    
    # Order points by some heuristic to improve pruning
    # Points that block many others should be considered first
    point_order = list(range(N_PTS))
    
    # Compute "blocking power" of each point: how many other points it helps block
    blocking_power = [0] * N_PTS
    for p in range(N_PTS):
        for q in range(N_PTS):
            if p == q:
                continue
            # p and q form a "blocking pair" if there exists r such that {p,q,r} is isosceles
            # Actually, we need: does p help block q?
            # q is blocked by S if exists u,v in S with {q,u,v} isosceles.
            # If p in S, p contributes to blocking q if exists u in S\{p} with {q,p,u} isosceles.
            # This is complex. Let's use a simpler heuristic: degree in the conflict graph.
            pass
    
    # For each point p, count how many other points q have p in some bad pair with another point
    # Simplified: use the number of bad pairs involving p
    for p in range(N_PTS):
        blocking_power[p] = len(BAD_PAIRS[p])
    
    # Sort by blocking power descending (points that are in more bad pairs first)
    point_order.sort(key=lambda x: -blocking_power[x])
    
    # State: current set S (as a set), and we decide for each point in point_order whether to include it
    # But we need to ensure maximality at the end.
    
    # Alternative approach: use iterative deepening or branch and bound.
    # Let's use a recursive backtracking with pruning.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(idx, S_set, remaining):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if idx == len(point_order):
            # Check if S_set is maximal
            if _is_maximal(S_set) and _is_valid_no_isosceles(S_set):
                if len(S_set) < best_size:
                    best_size = len(S_set)
                    best_set = set(S_set)
            return
        
        # Pruning: if current size + minimum additional needed > best_size, prune
        # Minimum additional needed: at least 0, but we need maximality
        # Better pruning: if len(S_set) >= best_size, prune
        if len(S_set) >= best_size:
            return
        
        p = point_order[idx]
        
        # Option 1: Don't include p
        # Check if this is still feasible (i.e., we can still achieve maximality)
        # For now, just try both options
        
        # Option 2: Include p (if compatible)
        if _can_add(p, S_set):
            S_set.add(p)
            backtrack(idx + 1, S_set, remaining - 1)
            S_set.discard(p)
        
        # Option 1: Don't include p
        backtrack(idx + 1, S_set, remaining)
    
    backtrack(0, set(), N_PTS)
    
    return list(best_set) if best_set else None

def _local_search_improved(S_initial, deadline):
    """
    Improved local search using 1-move and 2-move swaps.
    """
    S_set = set(S_initial)
    
    # Verify initial set is valid and maximal
    if not _is_valid_no_isosceles(S_set):
        # Repair: remove points that cause isosceles triangles
        changed = True
        while changed:
            changed = False
            S_list = list(S_set)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    for k in range(j+1, len(S_list)):
                        c = S_list[k]
                        da = dist2[a, b]
                        db = dist2[b, c]
                        dc = dist2[a, c]
                        if da == db or db == dc or da == dc:
                            # Remove one of the three
                            S_set.discard(c)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                continue
            break
    
    # Make it maximal
    while True:
        addable = [p for p in range(N_PTS) if p not in S_set and _can_add(p, S_set)]
        if not addable:
            break
        p = random.choice(addable)
        S_set.add(p)
    
    best_set = set(S_set)
    best_size = len(S_set)
    
    while time.time() < deadline:
        if not S_set:
            break
            
        # Try 1-move: remove u, add v
        u = random.choice(list(S_set))
        candidates_v = [v for v in range(N_PTS) if v not in S_set]
        if not candidates_v:
            break
        v = random.choice(candidates_v)
        
        S_minus_u = S_set - {u}
        S_new = S_minus_u | {v}
        
        # Check if v is compatible with S_minus_u
        valid = True
        for w, x in BAD_PAIRS[v]:
            if w in S_minus_u and x in S_minus_u:
                valid = False
                break
        
        if valid and _is_maximal(S_new):
            S_set = S_new
            if len(S_set) < best_size:
                best_size = len(S_set)
                best_set = set(S_set)
    
    return list(best_set)

def _greedy_with_pruning(deadline):
    """
    Greedy construction with aggressive pruning.
    At each step, choose the point that blocks the most remaining addable points.
    """
    S = []
    S_set = set()
    C = set(range(N_PTS))
    
    while C:
        if time.time() > deadline:
            break
        
        best_p = -1
        best_coverage = -1
        best_new_C = None
        
        # Try all candidates, pick the one that maximizes coverage
        for p in C:
            if not _can_add(p, S_set):
                continue
            
            coverage = 0
            new_C = set()
            
            for q in C:
                if q == p:
                    continue
                # Check if q is still addable with S_set ∪ {p}
                if _can_add(q, S_set | {p}):
                    new_C.add(q)
                else:
                    coverage += 1
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_p = p
                best_new_C = new_C
        
        if best_p == -1:
            break
        
        S.append(best_p)
        S_set.add(best_p)
        C = best_new_C
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Quick greedy to get a starting point
    S_init = _greedy_with_pruning(deadline - 200.0)
    if S_init:
        S_opt = _local_search_improved(S_init, deadline - 100.0)
        if S_opt:
            best_size = len(S_opt)
            best_set = S_opt
    
    # Phase 2: Multiple restarts with local search
    restarts = 0
    while time.time() < deadline - 20.0 and restarts < 50:
        restarts += 1
        random.seed()  # Re-seed for diversity
        
        S_init = _greedy_with_pruning(deadline - 20.0)
        if S_init:
            S_opt = _local_search_improved(S_init, deadline - 10.0)
            if S_opt and len(S_opt) < best_size:
                best_size = len(S_opt)
                best_set = S_opt
    
    # Phase 3: If time remains, try backtracking for small sizes
    if time.time() < deadline - 30.0 and best_set is not None:
        # Try to find something smaller than best_size
        bt_result = _backtrack_search(deadline - 10.0, max_size=best_size)
        if bt_result and len(bt_result) < best_size:
            best_size = len(bt_result)
            best_set = bt_result
    
    if best_set is None:
        # Fallback
        best_set = _greedy_with_pruning(deadline)
    
    return [ALL_POINTS[i] for i in best_set]
