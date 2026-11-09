import itertools
import math
import random
import sys

def construct():
    # Grid size
    N = 15 # 31x31 = 961 points. Manageable.
    points_grid = [(x, y) for x in range(-N, N+1) for y in range(-N, N+1)]
    n = len(points_grid)
    
    # Precompute valid edges (integer distance)
    valid_edge = [[False]*n for _ in range(n)]
    for i in range(n):
        xi, yi = points_grid[i]
        for j in range(i+1, n):
            xj, yj = points_grid[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx*dx + dy*dy
            if d2 == 0: continue
            d = int(math.isqrt(d2))
            if d*d == d2:
                valid_edge[i][j] = True
                valid_edge[j][i] = True

    # Helper functions
    def is_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p1[1] - p3[1]) == (p1[1] - p2[1]) * (p1[0] - p3[0])

    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        d1 = x1*x1 + y1*y1
        d2 = x2*x2 + y2*y2
        d3 = x3*x3 + y3*y3
        d4 = x4*x4 + y4*y4
        
        # det4
        a, b, c, d = x1, y1, d1, 1
        e, f, g, h = x2, y2, d2, 1
        i, j, k, l = x3, y3, d3, 1
        m, n, o, p = x4, y4, d4, 1
        
        return (a*(f*(k*p - l*o) - g*(j*p - l*n) + h*(j*o - k*n))
              - b*(e*(k*p - l*o) - g*(i*p - l*m) + h*(i*o - k*m))
              + c*(e*(j*p - l*n) - f*(i*p - l*m) + h*(i*n - j*m))
              - d*(e*(j*o - k*n) - f*(i*o - k*m) + g*(i*n - j*m))) == 0

    best_set = []
    
    # Bron-Kerbosch style with pivot? Or simple recursive with available set.
    # We want max clique in the graph defined by valid_edge,
    # subject to geometric constraints.
    
    # Since geometric constraints are "clique-independent" (depend on subsets),
    # we can't just use standard max clique.
    # But we can use the graph to prune.
    
    # Let's use a simple backtracking with an 'available' set.
    # To avoid duplicates, we enforce an ordering: indices in the set must be increasing?
    # No, that might miss sets. But if we iterate candidates in a fixed order,
    # and only add candidates with index > last_added_index, we find each set once.
    
    def backtrack(current_indices, current_points, available_indices):
        nonlocal best_set
        if len(current_points) > len(best_set):
            best_set = current_points[:]
        
        if len(best_set) >= 6:
            return
            
        # Pruning: if current size + len(available_indices) <= len(best_set), return
        if len(current_points) + len(available_indices) <= len(best_set):
            return
            
        for idx in available_indices:
            # Check geometric constraints with current points
            p_new = points_grid[idx]
            valid = True
            
            # Collinearity
            for i in range(len(current_points)):
                for j in range(i+1, len(current_points)):
                    if is_collinear(current_points[i], current_points[j], p_new):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Concyclicity
            for i in range(len(current_points)):
                for j in range(i+1, len(current_points)):
                    for k in range(j+1, len(current_points)):
                        if is_concyclic(current_points[i], current_points[j], current_points[k], p_new):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Recurse
                # New available: those in available_indices that are > idx (to enforce order) 
                # AND connected to idx
                new_available = [i for i in available_indices if i > idx and valid_edge[idx][i]]
                current_indices.append(idx)
                current_points.append(p_new)
                backtrack(current_indices, current_points, new_available)
                current_indices.pop()
                current_points.pop()
                
                if len(best_set) >= 6:
                    return

    # Initial available: all indices
    # To improve speed, sort available by degree?
    # The ordering `i > idx` requires us to pass a sorted list.
    # Let's just use all indices 0..n-1 initially.
    
    backtrack([], [], list(range(n)))
    
    # Fallback
    if len(best_set) < 5:
        seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_set = seed_points

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
