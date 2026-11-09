"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from math import gcd, isqrt
import time

def construct():
    start_time = time.time()
    timeout = 100  # seconds
    
    # Search for 6 points with small integer coordinates
    max_coord = 20
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))
    
    n = len(candidates)
    
    # Precompute rational distance (perfect square d2)
    rational_dist = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dx = candidates[i][0] - candidates[j][0]
            dy = candidates[i][1] - candidates[j][1]
            d2 = dx*dx + dy*dy
            s = isqrt(d2)
            if s * s == d2:
                rational_dist[i][j] = True
                rational_dist[j][i] = True
    
    def is_collinear(p1, p2, p3):
        return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0
    
    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        r1 = x1*x1 + y1*y1; r2 = x2*x2 + y2*y2
        r3 = x3*x3 + y3*y3; r4 = x4*x4 + y4*y4
        det = (x1*(y2*(r3-r4) - y3*(r2-r4) + y4*(r2-r3))
             - y1*(x2*(r3-r4) - x3*(r2-r4) + x4*(r2-r3))
             + r1*(x2*y3 - x3*y2 - x2*y4 + x4*y2 + x3*y4 - x4*y3)
             - (x2*y3*r4 - x3*y2*r4 + x2*y4*r3 - x4*y2*r3 - x3*y4*r2 + x4*y3*r2))
        return det == 0
    
    # Backtracking
    idx_00 = candidates.index((0,0))
    best_set = None
    
    def backtrack(current_set, start_idx):
        nonlocal best_set
        if best_set is not None:
            return
        if time.time() - start_time > timeout:
            return
        if len(current_set) == 6:
            for i in range(3):
                for j in range(i+1, 3):
                    for k in range(j+1, 3):
                        if is_collinear(candidates[current_set[i]], 
                                         candidates[current_set[j]], 
                                         candidates[current_set[k]]):
                            return
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        for l in range(k+1, 4):
                            if is_concyclic(candidates[current_set[i]], 
                                             candidates[current_set[j]], 
                                             candidates[current_set[k]], 
                                             candidates[current_set[l]]):
                                return
            best_set = current_set[:]
            return
        
        for idx in range(start_idx, n):
            if time.time() - start_time > timeout:
                return
            if idx in current_set:
                continue
            ok = True
            for ci in current_set:
                if not rational_dist[ci][idx]:
                    ok = False
                    break
            if not ok:
                continue
            if len(current_set) >= 2:
                for i in range(len(current_set)):
                    for j in range(i+1, len(current_set)):
                        if is_collinear(candidates[current_set[i]], 
                                         candidates[current_set[j]], 
                                         candidates[idx]):
                            ok = False
                            break
                    if not ok:
                        break
            if not ok:
                continue
            if len(current_set) >= 3:
                for i in range(len(current_set)):
                    for j in range(i+1, len(current_set)):
                        for k in range(j+1, len(current_set)):
                            if is_concyclic(candidates[current_set[i]], 
                                             candidates[current_set[j]], 
                                             candidates[current_set[k]], 
                                             candidates[idx]):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
            if not ok:
                continue
            
            current_set.append(idx)
            backtrack(current_set, idx + 1)
            current_set.pop()
            if best_set is not None:
                return
    
    backtrack([idx_00], idx_00 + 1)
    
    if best_set is not None:
        points = [candidates[i] for i in best_set]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
    
    # Fallback: 5-point set
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
