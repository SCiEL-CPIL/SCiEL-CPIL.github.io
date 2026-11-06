import itertools
import time
import random
import numpy as np

N = 16
V = N * N

def _precompute_conflicts():
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # dist[i,j] = squared distance
    dist = np.zeros((V, V), dtype=np.int32)
    for i in range(V):
        diff = pts[i] - pts
        dist[i] = np.sum(diff**2, axis=1)
        
    # conflict[i][j] is a set of k such that (i,j,k) is isosceles
    # We store as a list of sets for easy manipulation in DFS, 
    # but for speed in checking, we might want bitsets.
    # For now, let's use sets for the DFS logic as V=256 is small.
    
    conflict = [[set() for _ in range(V)] for __ in range(V)]
    for i in range(V):
        for j in range(i+1, V):
            d_ij = dist[i, j]
            d_ik = dist[i]
            d_jk = dist[j]
            mask = (d_ik == d_ij) | (d_jk == d_ij) | (d_ik == d_jk)
            mask[i] = False
            mask[j] = False
            ks = np.where(mask)[0]
            for k in ks:
                conflict[i][j].add(k)
                conflict[j][i].add(k)
                
    return points, conflict

_points, _conflict = None, None

def _init():
    global _points, _conflict
    if _points is None:
        _points, _conflict = _precompute_conflicts()

def _is_valid(S, conflict):
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            for k in S_list:
                if k != S_list[i] and k != S_list[j] and k in conflict[S_list[i]][S_list[j]]:
                    return False
    return True

def _get_blocked(S, conflict):
    blocked = set()
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            blocked |= conflict[S_list[i]][S_list[j]]
    return blocked

def _is_maximal(S, conflict):
    blocked = _get_blocked(S, conflict)
    # All points in the grid must be in S or blocked
    # Actually, the definition is "no further grid point can be added".
    # This means for all p not in S, p is blocked by some pair in S.
    # Points in S are trivially "not addable" because they are already there.
    # So we check if (V \ S) is a subset of blocked.
    S_set = set(S)
    for p in range(V):
        if p not in S_set and p not in blocked:
            return False
    return True

def _dfs_find_set(k, conflict, deadline):
    """
    Try to find a valid, maximal set of size k.
    Returns the set if found, else None.
    """
    points = list(range(V))
    random.shuffle(points) # Randomize order for diversity
    
    def dfs(S, blocked, idx):
        if time.time() > deadline:
            return None
        if len(S) == k:
            if _is_valid(S, conflict) and _is_maximal(S, conflict):
                return list(S)
            return None
            
        # Pruning: if even adding all remaining points cannot cover the grid?
        # This is hard to estimate quickly.
        
        # If current S is invalid, prune
        if not _is_valid(S, conflict):
            return None
            
        # Try adding points from idx onwards
        for i in range(idx, len(points)):
            p = points[i]
            if p in S:
                continue
                
            new_S = S + [p]
            
            # Check validity of new_S
            # Only need to check triangles involving p
            valid = True
            for a in S:
                for b in S:
                    if a == b: continue
                    if p in conflict[a][b]:
                        valid = False
                        break
                if not valid: break
            if not valid:
                continue
                
            # Update blocked
            new_blocked = set(blocked)
            for a in S:
                new_blocked |= conflict[p][a]
                
            res = dfs(new_S, new_blocked, i+1)
            if res is not None:
                return res
                
        return None
        
    return dfs([], set(), 0)

def construct():
    _init()
    points = _points
    conflict = _conflict
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Try increasing sizes
    for k in range(1, 11):
        # Multiple random trials for each k
        trials = 50
        for t in range(trials):
            if time.time() > deadline:
                break
            # Shuffle points globally for this trial
            random.seed(t * 100 + k)
            res = _dfs_find_set(k, conflict, deadline)
            if res is not None:
                return [points[i] for i in res]
                
    # Fallback: Greedy
    S = []
    U = set(range(V))
    while U:
        blocked = _get_blocked(S, conflict)
        candidates = [p for p in U if p not in blocked]
        if not candidates:
            break
        # Pick candidate that blocks most
        best_c = None
        max_block = -1
        for p in candidates:
            new_block = set()
            for a in S:
                new_block |= conflict[p][a]
            new_block &= U
            if len(new_block) > max_block:
                max_block = len(new_block)
                best_c = p
        if best_c is None:
            break
        S.append(best_c)
        U.remove(best_c)
        
    return [points[i] for i in S]
