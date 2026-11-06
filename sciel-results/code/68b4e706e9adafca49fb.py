import time
import math
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit

    # Helper to check if a set of points is valid
    # Points are tuples of Fractions: (x, y)
    def is_valid(points):
        n = len(points)
        # Check distinct
        if len(set(points)) != n:
            return False
        
        # Check no three collinear - use integer arithmetic where possible
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # det = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
                    dx1 = p2[0] - p1[0]
                    dy1 = p2[1] - p1[1]
                    dx2 = p3[0] - p1[0]
                    dy2 = p3[1] - p1[1]
                    det = dx1 * dy2 - dx2 * dy1
                    if det == 0:
                        return False
        
        # Check no four concyclic
        # Use the determinant of the circle equation
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1 = points[i]
                        p2 = points[j]
                        p3 = points[k]
                        p4 = points[l]
                        # Circle through p1, p2, p3: x^2 + y^2 + Dx + Ey + F = 0
                        # For p4 to be concyclic, it must satisfy the same equation
                        # Use the 4x4 determinant:
                        # |x1^2+y1^2  x1  y1  1|
                        # |x2^2+y2^2  x2  y2  1| = 0
                        # |x3^2+y3^2  x3  y3  1|
                        # |x4^2+y4^2  x4  y4  1|
                        
                        # Compute using Fraction arithmetic
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        
                        a1 = x1*x1 + y1*y1
                        a2 = x2*x2 + y2*y2
                        a3 = x3*x3 + y3*y3
                        a4 = x4*x4 + y4*y4
                        
                        # 4x4 determinant
                        # Use the formula directly
                        det_val = (
                            a1 * (x2*(y3*1 - y4*1) - y2*(x3*1 - x4*1) + 1*(x3*y4 - x4*y3))
                            - x1 * (a2*(y3*1 - y4*1) - y2*(a3*1 - a4*1) + 1*(a3*y4 - a4*y3))
                            + y1 * (a2*(x3*1 - x4*1) - a3*(x2*1 - x4*1) + a4*(x2*y3 - x3*y2))
                            - 1 * (a2*(x3*y4 - x4*y3) - x3*(a2*y4 - a4*y2) + x4*(a2*y3 - a3*y2))
                        )
                        if det_val == 0:
                            return False
        return True

    # Strategy: Use a known 5-point set and try to find a 6th point.
    # The seed points are:
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # These are integer points. Distances are integers.
    # We will try to find a 6th integer point (x,y) such that:
    # 1. Distances to all 5 are integers.
    # 2. No three collinear, no four concyclic.
    
    seed_pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_points = [(Fraction(x), Fraction(y)) for x,y in seed_pts]
    
    # Verify seed is valid (should be)
    if not is_valid(best_points):
        # Fallback: try a simpler set? The task says seed is valid.
        pass

    # Search for 6th point
    # We search integer coordinates in a range.
    # If we find one, we try to find a 7th.
    
    found_6 = False
    # Use a larger range if possible, but keep it efficient.
    # 2000x2000 = 4,000,000 checks. Each check is 5 distance checks + 1 validity check (which is expensive).
    # The distance check is cheap (integer arithmetic). The validity check is only run if distances are integers.
    # So most of the time is spent in the distance check loop.
    
    # Let's estimate: 4M points * 5 dist checks = 20M operations. Fast.
    # The bottleneck is is_valid() which is O(N^4) for concyclic check.
    # For N=6, it's C(6,3)=20 collinear checks + C(6,4)=15 concyclic checks.
    # Each concyclic check involves 4x4 determinant of Fractions.
    # This might be slow if we call it for many candidates.
    # But candidates with integer distances to 5 specific points are rare.
    
    # Let's try a range of -2000 to 2000.
    # If we find a 6th point, we can stop searching for 6th and start searching for 7th.
    
    # Optimization: Check distinctness first.
    seed_set = set(seed_pts)
    
    search_range = 2000
    
    for x in range(-search_range, search_range + 1):
        if time.time() - start_time > timeout:
            break
        for y in range(-search_range, search_range + 1):
            if time.time() - start_time > timeout:
                break
            new_pt = (x, y)
            if new_pt in seed_set:
                continue
            
            valid_dist = True
            for p in seed_pts:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx*dx + dy*dy
                root = math.isqrt(d2)
                if root*root != d2:
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            # Quick collinearity pre-check with integer arithmetic
            # Check if new point is collinear with any pair of seed points
            collinear = False
            for i in range(len(seed_pts)):
                for j in range(i+1, len(seed_pts)):
                    p1 = seed_pts[i]
                    p2 = seed_pts[j]
                    # Check if (x,y) is collinear with p1 and p2
                    dx1 = p2[0] - p1[0]
                    dy1 = p2[1] - p1[1]
                    dx2 = x - p1[0]
                    dy2 = y - p1[1]
                    if dx1 * dy2 - dx2 * dy1 == 0:
                        collinear = True
                        break
                if collinear:
                    break
            
            if collinear:
                continue
            
            # Candidate found. Check general position.
            new_set = [(Fraction(px), Fraction(py)) for px, py in seed_pts] + [(Fraction(x), Fraction(y))]
            if is_valid(new_set):
                best_points = new_set
                found_6 = True
                break
        if found_6:
            break

    # If we found a 6th point, try to find a 7th
    if found_6:
        # Search for 7th point
        # Now we have 6 points.
        # Search range can be smaller or larger?
        # Let's try -2000 to 2000 again.
        found_7 = False
        # Convert best_points back to integers for fast distance check?
        # They are Fractions but they are integers.
        int_pts = [(int(p[0]), int(p[1])) for p in best_points]
        int_set = set(int_pts)
        
        search_range_7 = 2000
        
        for x in range(-search_range_7, search_range_7 + 1):
            if time.time() - start_time > timeout:
                break
            for y in range(-search_range_7, search_range_7 + 1):
                if time.time() - start_time > timeout:
                    break
                new_pt = (x, y)
                if new_pt in int_set:
                    continue
                
                valid_dist = True
                for p in int_pts:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx*dx + dy*dy
                    root = math.isqrt(d2)
                    if root*root != d2:
                        valid_dist = False
                        break
                
                if not valid_dist:
                    continue
                
                # Quick collinearity pre-check with integer arithmetic
                collinear = False
                for i in range(len(int_pts)):
                    for j in range(i+1, len(int_pts)):
                        p1 = int_pts[i]
                        p2 = int_pts[j]
                        dx1 = p2[0] - p1[0]
                        dy1 = p2[1] - p1[1]
                        dx2 = x - p1[0]
                        dy2 = y - p1[1]
                        if dx1 * dy2 - dx2 * dy1 == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                
                if collinear:
                    continue
                
                # Candidate found. Check general position.
                new_set = best_points + [(Fraction(x), Fraction(y))]
                if is_valid(new_set):
                    best_points = new_set
                    found_7 = True
                    break
            if found_7:
                break

    # If we found a 7th point, try to find an 8th
    if found_7:
        found_8 = False
        int_pts = [(int(p[0]), int(p[1])) for p in best_points]
        int_set = set(int_pts)
        
        search_range_8 = 1500  # Slightly smaller range to ensure we finish in time
        
        for x in range(-search_range_8, search_range_8 + 1):
            if time.time() - start_time > timeout:
                break
            for y in range(-search_range_8, search_range_8 + 1):
                if time.time() - start_time > timeout:
                    break
                new_pt = (x, y)
                if new_pt in int_set:
                    continue
                
                valid_dist = True
                for p in int_pts:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx*dx + dy*dy
                    root = math.isqrt(d2)
                    if root*root != d2:
                        valid_dist = False
                        break
                
                if not valid_dist:
                    continue
                
                # Quick collinearity pre-check
                collinear = False
                for i in range(len(int_pts)):
                    for j in range(i+1, len(int_pts)):
                        p1 = int_pts[i]
                        p2 = int_pts[j]
                        dx1 = p2[0] - p1[0]
                        dy1 = p2[1] - p1[1]
                        dx2 = x - p1[0]
                        dy2 = y - p1[1]
                        if dx1 * dy2 - dx2 * dy1 == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                
                if collinear:
                    continue
                
                # Candidate found. Check general position.
                new_set = best_points + [(Fraction(x), Fraction(y))]
                if is_valid(new_set):
                    best_points = new_set
                    found_8 = True
                    break
            if found_8:
                break

    # Format output
    D = 1
    points_out = []
    for p in best_points:
        x, y = p
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': points_out}
