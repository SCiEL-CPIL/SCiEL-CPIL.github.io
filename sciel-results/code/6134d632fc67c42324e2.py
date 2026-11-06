# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Known valid 5-point set
    base_points_list = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (-3, 4), (0, 5), (3, 4), (-4, 3)],
        [(0, 0), (4, 3), (0, 5), (-4, 3), (3, 4)],
    ]
    
    # Verification function
    def is_valid(pts):
        n = len(pts)
        # Check distinct
        for i in range(n):
            for j in range(i+1, n):
                if pts[i] == pts[j]:
                    return False
        
        # Check rational distances
        for i in range(n):
            for j in range(i+1, n):
                dx = Fraction(pts[i][0]) - Fraction(pts[j][0])
                dy = Fraction(pts[i][1]) - Fraction(pts[j][1])
                dist_sq = dx*dx + dy*dy
                if dist_sq.denominator != 1:
                    return False
                num = dist_sq.numerator
                if num < 0:
                    return False
                root = int(num**0.5)
                if root*root != num:
                    return False
        
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    if (p2[1]-p1[1])*(p3[0]-p1[0]) == (p3[1]-p1[1])*(p2[0]-p1[0]):
                        return False
        
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        
                        dx2 = Fraction(x2-x1)
                        dy2 = Fraction(y2-y1)
                        rhs2 = Fraction(-(x2*x2+y2*y2 - x1*x1-y1*y1))
                        
                        dx3 = Fraction(x3-x1)
                        dy3 = Fraction(y3-y1)
                        rhs3 = Fraction(-(x3*x3+y3*y3 - x1*x1-y1*y1))
                        
                        det = dx2*dy3 - dx3*dy2
                        if det == 0:
                            return False
                            
                        A = (rhs2*dy3 - rhs3*dy2) / det
                        B = (dx2*rhs3 - dx3*rhs2) / det
                        C = Fraction(-(x1*x1+y1*y1)) - A*x1 - B*y1
                        
                        val = x4*x4 + y4*y4 + A*x4 + B*y4 + C
                        if val == 0:
                            return False
                            
        return True
    
    # First verify base sets are valid
    valid_base_sets = []
    for bp in base_points_list:
        if is_valid(bp):
            valid_base_sets.append(bp)
    
    if not valid_base_sets:
        # Fallback: return a simple valid 3-point set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}
    
    best_points = valid_base_sets[0]  # Default to first valid 5-point set
    best_score = len(valid_base_sets[0])
    
    # For each valid base set, try to find a 6th point
    for base_points in valid_base_sets:
        if time.time() > deadline - 5:
            break
        
        base_set = set(base_points)
        
        # Generate candidate points in spiral order for better coverage
        def spiral_candidates(max_radius):
            candidates = []
            for r in range(0, max_radius + 1):
                if r == 0:
                    candidates.append((0, 0))
                    continue
                # Points on the square boundary at radius r
                for x in range(-r, r + 1):
                    if abs(x) == r:
                        for y in range(-r, r + 1):
                            if abs(y) == r:
                                candidates.append((x, y))
                # Deduplicate by keeping only boundary points
            # Remove duplicates while preserving order
            seen = set()
            unique = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            return unique
        
        # Try increasing radii
        found_sixth = False
        max_radius = 50
        
        while not found_sixth and time.time() < deadline - 5:
            candidates = spiral_candidates(max_radius)
            
            for (x, y) in candidates:
                if time.time() > deadline - 5:
                    break
                
                # Skip if this point is already in base_points
                if (x, y) in base_set:
                    continue
                
                # Quick check: distances to all 5 base points must be rational (perfect squares)
                valid_candidate = True
                for (xi, yi) in base_points:
                    dx = x - xi
                    dy = y - yi
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < 0:
                        valid_candidate = False
                        break
                    root = math.isqrt(dist_sq)
                    if root * root != dist_sq:
                        valid_candidate = False
                        break
                
                if not valid_candidate:
                    continue
                
                # Check collinearity with existing points
                collinear_found = False
                for i in range(len(base_points)):
                    for j in range(i+1, len(base_points)):
                        p1 = (x, y)
                        p2 = base_points[i]
                        p3 = base_points[j]
                        if (p2[1]-p1[1])*(p3[0]-p1[0]) == (p3[1]-p1[1])*(p2[0]-p1[0]):
                            collinear_found = True
                            break
                    if collinear_found:
                        break
                
                if collinear_found:
                    continue
                
                # Check concyclicity: no 4 points (including new) should be concyclic
                concyclic_found = False
                for i in range(len(base_points)):
                    for j in range(i+1, len(base_points)):
                        for k in range(j+1, len(base_points)):
                            p1 = base_points[i]
                            p2 = base_points[j]
                            p3 = base_points[k]
                            p4 = (x, y)
                            
                            x1, y1 = p1
                            x2, y2 = p2
                            x3, y3 = p3
                            x4, y4 = p4
                            
                            dx2 = x2 - x1
                            dy2 = y2 - y1
                            rhs2 = -(x2*x2+y2*y2 - x1*x1-y1*y1)
                            
                            dx3 = x3 - x1
                            dy3 = y3 - y1
                            rhs3 = -(x3*x3+y3*y3 - x1*x1-y1*y1)
                            
                            det = dx2*dy3 - dx3*dy2
                            if det == 0:
                                # Collinear, skip (already checked)
                                continue
                                
                            A_num = rhs2*dy3 - rhs3*dy2
                            B_num = dx2*rhs3 - dx3*rhs2
                            C_num = -(x1*x1+y1*y1)*det - A_num*x1 - B_num*y1
                            
                            check = (x4*x4+y4*y4)*det + A_num*x4 + B_num*y4 + C_num
                            if check == 0:
                                concyclic_found = True
                                break
                        if concyclic_found:
                            break
                    if concyclic_found:
                        break
                
                if concyclic_found:
                    continue
                
                # All checks passed! We found a valid 6th point
                candidate_set = base_points + [(x, y)]
                if is_valid(candidate_set):
                    if len(candidate_set) > best_score:
                        best_points = candidate_set
                        best_score = len(candidate_set)
                    found_sixth = True
                    break
            
            if found_sixth:
                break
            
            max_radius += 20
    
    # Final verification
    if is_valid(best_points):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
    
    # Fallback
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in valid_base_sets[0]]}
# EVOLVE-BLOCK-END
