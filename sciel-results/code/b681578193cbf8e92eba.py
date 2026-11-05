"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools
import numpy as np

_GLOBAL_POINTS = None
_GLOBAL_BLOCKS = None

def _precompute_blocks():
    global _GLOBAL_POINTS, _GLOBAL_BLOCKS
    if _GLOBAL_POINTS is not None:
        return _GLOBAL_POINTS, _GLOBAL_BLOCKS
        
    points = list(itertools.product(range(N), repeat=2))
    n_points = len(points)
    
    dist_sq = np.zeros((n_points, n_points), dtype=np.int32)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d = dx * dx + dy * dy
            dist_sq[i, j] = d
            dist_sq[j, i] = d
            
    blocks = [[] for _ in range(n_points * n_points)]
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d_ab = dist_sq[i, j]
            for k in range(n_points):
                if k == i or k == j:
                    continue
                d_ka = dist_sq[k, i]
                d_kb = dist_sq[k, j]
                if d_ka == d_kb or d_ka == d_ab or d_kb == d_ab:
                    blocks[i * n_points + j].append(k)
                    blocks[j * n_points + i].append(k)
                    
    _GLOBAL_POINTS = points
    _GLOBAL_BLOCKS = blocks
    return _GLOBAL_POINTS, _GLOBAL_BLOCKS

def _greedy_set_cover():
    points, blocks = _precompute_blocks()
    n_points = len(points)
    
    blocked = [False] * n_points
    S = []
    S_set = set()
    
    while True:
        valid_candidates = []
        for p in range(n_points):
            if p not in S_set and not blocked[p]:
                valid_candidates.append(p)
                
        if not valid_candidates:
            break
            
        best_p = None
        max_new_blocks = -1
        
        for p in valid_candidates:
            new_blocks = 0
            for a in S:
                if a == p: continue
                for q in blocks[p * n_points + a]:
                    if not blocked[q]:
                        new_blocks += 1
            if new_blocks > max_new_blocks:
                max_new_blocks = new_blocks
                best_p = p
                
        if best_p is None:
            best_p = valid_candidates[0]
            
        S.append(best_p)
        S_set.add(best_p)
        
        for a in S:
            if a == best_p: continue
            for q in blocks[best_p * n_points + a]:
                blocked[q] = True
                
    return S

def construct():
    # Run deterministic greedy set cover
    result = _greedy_set_cover()
    return result
# EVOLVE-BLOCK-END
