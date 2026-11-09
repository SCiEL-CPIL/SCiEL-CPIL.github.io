"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _check_rational_dist(p1, p2, D):
    """Check if squared distance is a perfect square rational."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    num = dx * dx + D * dy * dy
    # For D=1 and integer coords, num is integer.
    # Distance is rational iff num is a perfect square.
    return _is_square(num)

def _check_no_three_collinear(points, D):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Cross product in (x, y) space (since D is constant factor)
                val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if val == 0:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Circle equation: x^2 + D*y^2 + Ax + By + C = 0
                    # Solve for A, B, C using p1, p2, p3 and check p4.
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    
                    # System:
                    # x1*A + y1*B + C = -(x1^2 + D*y1^2)
                    # x2*A + y2*B + C = -(x2^2 + D*y2^2)
                    # x3*A + y3*B + C = -(x3^2 + D*y3^2)
                    
                    # Determinant of coefficient matrix
                    det = x1*(y2 - y3) - y1*(x2 - x3) + (x2*y3 - x3*y2)
                    if det == 0:
                        # Collinear, already checked
                        continue
                    
                    # Cramer's Rule for A
                    # Replace col 1 with RHS
                    rhs1 = -(x1*x1 + D*y1*y1)
                    rhs2 = -(x2*x2 + D*y2*y2)
                    rhs3 = -(x3*x3 + D*y3*y3)
                    
                    det_A = rhs1*(y2 - y3) - y1*(rhs2 - rhs3) + (rhs2*y3 - rhs3*y2)
                    # A = det_A / det
                    
                    # Replace col 2 with RHS
                    det_B = x1*(rhs2 - rhs3) - rhs1*(x2 - x3) + (x2*rhs3 - x3*rhs2)
                    # B = det_B / det
                    
                    # Replace col 3 with RHS
                    det_C = x1*(y2*rhs3 - y3*rhs2) - y1*(x2*rhs3 - x3*rhs2) + rhs1*(x2*y3 - x3*y2)
                    # C = det_C / det
                    
                    # Check if p4 satisfies: x4*A + y4*B + C = -(x4^2 + D*y4^2)
                    # x4*det_A + y4*det_B + det_C == -det * (x4^2 + D*y4^2)
                    lhs = x4*det_A + y4*det_B + det_C
                    rhs_check = -det * (x4*x4 + D*y4*y4)
                    
                    if lhs == rhs_check:
                        return False
    return True

def _is_valid(points, D):
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if not _check_rational_dist(points[i], points[j], D):
                return False
    if not _check_no_three_collinear(points, D):
        return False
    if n >= 4:
        if not _check_no_four_concyclic(points, D):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    D = 1
    # Known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = base_points
    
    # Search for 6th point
    # Candidates must be at integer distance from all 5 base points
    max_coord = 1000
    found_6th = None
    
    # To speed up, we can iterate x, y and check distances
    # Optimization: Check distance to first point (0,0) first.
    # x^2 + y^2 must be a square.
    
    # We will iterate through a grid.
    # To avoid checking all 4 million points, we can be smarter?
    # For now, 2000x2000 = 4M points.
    # Distance check is 5 is_square calls.
    # 4M * 5 = 20M operations. Should be < 1s in Python? Maybe 5-10s.
    # Let's limit to 500 first, then expand if needed.
    
    for radius in [100, 200, 500, 1000]:
        if time.time() > deadline - 20:
            break
            
        for x in range(-radius, radius + 1):
            if time.time() > deadline - 20:
                break
            # Check dist to (0,0)
            if not _is_square(x*x): # Wait, (0,0) to (x,y) is x^2+y^2
                # We need x^2 + y^2 to be square.
                # So we can't just check x.
                pass
            
            for y in range(-radius, radius + 1):
                if time.time() > deadline - 20:
                    break
                
                cand = (x, y)
                if cand in base_points:
                    continue
                
                # Check dist to all base points
                valid_dist = True
                for bx, by in base_points:
                    if not _is_square((x - bx)**2 + (y - by)**2):
                        valid_dist = False
                        break
                
                if valid_dist:
                    # Check general position
                    new_set = base_points + [cand]
                    if _check_no_three_collinear(new_set, D) and _check_no_four_concyclic(new_set, D):
                        found_6th = cand
                        break
            if found_6th:
                break
        if found_6th:
            break
    
    if found_6th:
        best_points = base_points + [found_6th]
        
        # Try to find 7th point
        # Must be at integer distance to all 6 points
        # This is even rarer.
        # We can try a smaller search space or skip if time is tight.
        if time.time() < deadline - 30:
            found_7th = None
            for radius in [100, 200, 500]:
                if time.time() > deadline - 30:
                    break
                for x in range(-radius, radius + 1):
                    if time.time() > deadline - 30:
                        break
                    for y in range(-radius, radius + 1):
                        if time.time() > deadline - 30:
                            break
                        cand = (x, y)
                        if cand in best_points:
                            continue
                        valid_dist = True
                        for bx, by in best_points:
                            if not _is_square((x - bx)**2 + (y - by)**2):
                                valid_dist = False
                                break
                        if valid_dist:
                            new_set = best_points + [cand]
                            if _check_no_three_collinear(new_set, D) and _check_no_four_concyclic(new_set, D):
                                found_7th = cand
                                break
                    if found_7th:
                        break
                if found_7th:
                    break
            
            if found_7th:
                best_points = best_points + [found_7th]

    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
