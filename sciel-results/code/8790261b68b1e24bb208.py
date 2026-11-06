"""Integral point sets in general position."""
import math
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def _is_integer_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        r = int(math.isqrt(d2))
        return r*r == d2
    
    def _no_three_collinear(pts):
        for i,j,k in combinations(range(len(pts)), 3):
            x1,y1 = pts[i]; x2,y2 = pts[j]; x3,y3 = pts[k]
            if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                return False
        return True
    
    def _no_four_concyclic(pts):
        for i,j,k,l in combinations(range(len(pts)), 4):
            x1,y1 = pts[i]; x2,y2 = pts[j]; x3,y3 = pts[k]; x4,y4 = pts[l]
            # Four points concyclic iff the determinant is zero
            A = [[x1, y1, x1*x1+y1*y1, 1],
                 [x2, y2, x2*x2+y2*y2, 1],
                 [x3, y3, x3*x3+y3*y3, 1],
                 [x4, y4, x4*x4+y4*y4, 1]]
            det = (A[0][0]*(A[1][1]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - A[1][2]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) + A[1][3]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]))
                 - A[0][1]*(A[1][0]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - A[1][2]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + A[1][3]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]))
                 + A[0][2]*(A[1][0]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) - A[1][1]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + A[1][3]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]))
                 - A[0][3]*(A[1][0]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]) - A[1][1]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]) + A[1][2]*(A[2][0]*A[3][1]-A[2][1]*A[3][0])))
            if det == 0:
                return False
        return True
    
    def _valid_set(pts):
        if len(pts) < 3:
            return False
        for i,j in combinations(range(len(pts)), 2):
            if not _is_integer_dist(pts[i], pts[j]):
                return False
        if not _no_three_collinear(pts):
            return False
        if len(pts) >= 4 and not _no_four_concyclic(pts):
            return False
        return True
    
    # Known 5-point integral set
    base5 = [(0,0), (-56,33), (-16,63), (0,33), (16,63)]
    best_points = base5
    
    # Try to find a 6-point set by systematic search
    # Strategy: search for points (x,y) with small integer coords
    # such that all pairwise distances are integers and no three collinear
    
    # Precompute: for each candidate point, which other points have integer distance
    search_range = 30
    candidates = [(x, y) for x in range(-search_range, search_range+1) 
                  for y in range(-search_range, search_range+1)]
    
    # Build adjacency: can[i][j] = True if dist(candidates[i], candidates[j]) is integer
    n = len(candidates)
    can = [[False]*n for _ in range(n)]
    for i in range(n):
        xi, yi = candidates[i]
        for j in range(i+1, n):
            xj, yj = candidates[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx*dx + dy*dy
            r = int(math.isqrt(d2))
            if r*r == d2:
                can[i][j] = can[j][i] = True
    
    # Try to find a 6-clique in this graph with no three collinear and no four concyclic
    # Use backtracking
    found_6 = False
    best_6 = None
    
    # Sort candidates to prefer points that form good configurations
    # Try greedy + backtracking approach
    for start_i in range(n):
        if time.time() > deadline - 15:
            break
        x0, y0 = candidates[start_i]
        # Find neighbors of start_i
        neighbors = [j for j in range(start_i+1, n) if can[start_i][j]]
        
        for i1 in neighbors:
            if time.time() > deadline - 15:
                break
            x1, y1 = candidates[i1]
            # Check not collinear with origin (trivially not since 2 points)
            # Find neighbors of i1 that are also neighbors of start_i
            neighbors2 = [j for j in range(i1+1, n) if can[start_i][j] and can[i1][j]]
            
            for i2 in neighbors2:
                if time.time() > deadline - 15:
                    break
                x2, y2 = candidates[i2]
                # Check no three collinear
                if (x1-x0)*(y2-y0) == (y1-y0)*(x2-x0):
                    continue
                # Find neighbors of i2 that connect to all previous
                neighbors3 = [j for j in range(i2+1, n) 
                             if can[start_i][j] and can[i1][j] and can[i2][j]]
                
                for i3 in neighbors3:
                    if time.time() > deadline - 15:
                        break
                    x3, y3 = candidates[i3]
                    # Check no three collinear with any pair
                    collinear = False
                    for a in [(x0,y0),(x1,y1),(x2,y2)]:
                        if (x3-a[0])*(y3-y0) == (y3-y0)*(x3-x0) and a != (x0,y0):
                            pass  # This check is wrong, let me redo
                    # Proper collinearity check
                    pts4 = [(x0,y0),(x1,y1),(x2,y2),(x3,y3)]
                    if not _no_three_collinear(pts4):
                        continue
                    # Find neighbors of i3 that connect to all previous
                    neighbors4 = [j for j in range(i3+1, n)
                                 if can[start_i][j] and can[i1][j] and can[i2][j] and can[i3][j]]
                    
                    for i4 in neighbors4:
                        if time.time() > deadline - 15:
                            break
                        x4, y4 = candidates[i4]
                        pts5 = [(x0,y0),(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                        if not _no_three_collinear(pts5):
                            continue
                        # Check no four concyclic
                        if not _no_four_concyclic(pts5):
                            continue
                        # Find neighbors of i4 that connect to all previous
                        neighbors5 = [j for j in range(i4+1, n)
                                     if can[start_i][j] and can[i1][j] and can[i2][j] and can[i3][j] and can[i4][j]]
                        
                        for i5 in neighbors5:
                            if time.time() > deadline - 15:
                                break
                            x5, y5 = candidates[i5]
                            pts6 = [(x0,y0),(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5)]
                            if _no_three_collinear(pts6) and _no_four_concyclic(pts6):
                                best_6 = pts6
                                found_6 = True
                                break
                        if found_6:
                            break
                    if found_6:
                        break
                if found_6:
                    break
            if found_6:
                break
        if found_6:
            break
    
    if found_6 and best_6 is not None:
        best_points = best_6
    
    # Verify the best set
    assert _valid_set(best_points), "Final set failed validation!"
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
