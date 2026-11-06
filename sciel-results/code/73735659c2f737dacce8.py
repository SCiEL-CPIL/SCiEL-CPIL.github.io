import time
import random
import itertools
from collections import defaultdict

N = 16

def _dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist_sq(a, b)
    d2 = _dist_sq(a, c)
    d3 = _dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _precompute_conflicts():
    """Precompute for each pair of points, which other points form isosceles with them."""
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_idx = {p: i for i, p in enumerate(all_points)}
    n = len(all_points)
    
    # conflict[pair_idx] = set of points that form isosceles with this pair
    # pair_idx = i * n + j for i < j
    conflict = {}
    for i in range(n):
        for j in range(i + 1, n):
            blocked = set()
            for k in range(n):
                if k == i or k == j:
                    continue
                if _is_isosceles(all_points[k], all_points[i], all_points[j]):
                    blocked.add(k)
            conflict[(i, j)] = blocked
    
    return all_points, point_idx, conflict

def _is_valid_set(S, point_idx, all_points):
    """Check if S is a valid no-isosceles set."""
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_isosceles(S[i], S[j], S[k]):
                    return False
    return True

def _is_maximal(S, point_idx, all_points, conflict):
    """Check if S is maximal (no point can be added)."""
    S_idx = set(point_idx[p] for p in S)
    n = len(all_points)
    for p in all_points:
        if p in S:
            continue
        pi = point_idx[p]
        # Check if p can be added: for every pair in S, p should not form isosceles
        can_add = True
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                si = point_idx[S[i]]
                sj = point_idx[S[j]]
                pair_key = (min(si, sj), max(si, sj))
                if pi in conflict.get(pair_key, set()):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True

def _remove_redundant(S, point_idx, all_points, conflict):
    """Remove points from S that aren't needed for maximality.
    A point p in S is redundant if removing it still leaves S maximal."""
    S = list(S)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            S_test = S[:i] + S[i+1:]
            if _is_maximal(S_test, point_idx, all_points, conflict):
                S = S_test
                changed = True
                break
    return S

def _local_search(S, point_idx, all_points, conflict, time_limit):
    """Try to reduce set size via swap moves."""
    S = list(S)
    start_time = time.time()
    
    while time.time() < time_limit:
        improved = False
        for i in range(len(S)):
            # Try removing S[i] and adding a different point
            S_without = [S[j] for j in range(len(S)) if j != i]
            # Find points that could replace S[i]
            for p in all_points:
                if p in S_without:
                    continue
                S_candidate = S_without + [p]
                if not _is_valid_set(S_candidate, point_idx, all_points):
                    continue
                if len(S_candidate) < len(S):
                    S = S_candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return S

def construct():
    all_points, point_idx, conflict = _precompute_conflicts()
    
    best_set = None
    best_size = float('inf')
    
    start_time = time.time()
    deadline = start_time + 280
    
    # Greedy construction with blocking degree heuristic
    def greedy_build():
        S = []
        remaining = set(range(len(all_points)))
        
        while remaining:
            # Find valid candidates
            valid = []
            for pi in remaining:
                p = all_points[pi]
                compatible = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        si = point_idx[S[i]]
                        sj = point_idx[S[j]]
                        pair_key = (min(si, sj), max(si, sj))
                        if pi in conflict.get(pair_key, set()):
                            compatible = False
                            break
                    if not compatible:
                        break
                if compatible:
                    valid.append(pi)
            
            if not valid:
                break
            
            # Heuristic: pick point with max blocking degree
            best_v = None
            best_deg = -1
            best_ties = []
            
            for v in valid:
                deg = 0
                for u in remaining:
                    if u == v:
                        continue
                    # Check if u is blocked by v given S
                    for i in range(len(S)):
                        si = point_idx[S[i]]
                        pair_key = (min(si, v), max(si, v))
                        if u in conflict.get(pair_key, set()):
                            deg += 1
                            break
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                    best_ties = [v]
                elif deg == best_deg:
                    best_ties.append(v)
            
            chosen = random.choice(best_ties)
            S.append(all_points[chosen])
            remaining.remove(chosen)
            
            # Update remaining
            new_remaining = set()
            for q in remaining:
                compatible = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        si = point_idx[S[i]]
                        sj = point_idx[S[j]]
                        pair_key = (min(si, sj), max(si, sj))
                        if q in conflict.get(pair_key, set()):
                            compatible = False
                            break
                    if not compatible:
                        break
                if compatible:
                    new_remaining.add(q)
            remaining = new_remaining
        
        return S
    
    # Run multiple greedy builds
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        S = greedy_build()
        
        # Remove redundant points
        S = _remove_redundant(S, point_idx, all_points, conflict)
        
        # Local search
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            S = _local_search(S, point_idx, all_points, conflict, time.time() + remaining_time * 0.3)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
        
        if best_size <= 8:  # Early exit if we find a very small set
            break
    
    if best_set is None:
        # Fallback: return all points (trivially maximal)
        best_set = all_points
    
    return best_set
