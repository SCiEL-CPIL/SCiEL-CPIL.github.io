import time
import math
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_collinear(p1, p2, p3):
        # (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1)
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])

    def is_concyclic(p1, p2, p3, p4):
        # Check if 4 points are concyclic
        # Using determinant of circle equation
        # x^2+y^2+Dx+Ey+F = 0
        # det of:
        # [x_i, y_i, x_i^2+y_i^2, 1] for i=1..4
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        a1 = x1*x1 + y1*y1
        a2 = x2*x2 + y2*y2
        a3 = x3*x3 + y3*y3
        a4 = x4*x4 + y4*y4
        
        # Subtract row 1 from others to simplify determinant calculation
        # Actually, just compute the 4x4 determinant directly or use the formula
        # The determinant is non-zero if not concyclic.
        # det = | x1 y1 a1 1 |
        #       | x2 y2 a2 1 |
        #       | x3 y3 a3 1 |
        #       | x4 y4 a4 1 |
        
        # We can compute this by expanding or using the property that 
        # the points are concyclic iff the determinant is 0.
        # For speed, we can check if the circle through p1,p2,p3 passes through p4.
        # Equation of circle: x^2+y^2+Dx+Ey+F=0
        # D, E, F can be found from p1,p2,p3.
        # Then check if p4 satisfies it.
        
        # To avoid division by zero issues, use determinant.
        # Let's use the standard formula for 4 points concyclic.
        # It is equivalent to checking if the power of p4 w.r.t circle(p1,p2,p3) is 0.
        
        # A faster check:
        # The determinant is:
        # (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) - 
        # (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) + 
        # (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
        
        det = (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) \
            - (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) \
            + (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
            
        return det == 0

    def validate_set(points):
        n = len(points)
        if n < 3:
            return False
        # Check all pairwise distances are integer
        for i in range(n):
            for j in range(i+1, n):
                d = dist_sq(points[i], points[j])
                if not is_square(d):
                    return False
        # Check no three collinear
        for i, j, k in combinations(range(n), 3):
            if is_collinear(points[i], points[j], points[k]):
                return False
        # Check no four concyclic
        for i, j, k, l in combinations(range(n), 4):
            if is_concyclic(points[i], points[j], points[k], points[l]):
                return False
        return True

    # Start with a known 5-point set
    # This set is verified to have integer distances and be in general position
    seed = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not validate_set(seed):
        # Fallback to a smaller valid set if seed is invalid (should not happen)
        seed = [(0,0), (3,4), (0, 12)]
        # Check if this is valid
        if not validate_set(seed):
             # Try another
             seed = [(0,0), (3,4), (7,24)]
             if not validate_set(seed):
                 return {'D': 1, 'points': [[[0,1],[0,1]]]} # Minimal valid 1 point? No, needs 3.
                 # Let's just return the first 3 of the original seed if it fails?
                 # The original seed is likely valid.
    
    best_set = list(seed)
    
    # Try to extend the set
    # We will search for new points in an expanding grid
    # To make it faster, we can precompute some things or use heuristics.
    
    def can_add_point(p, existing):
        # p is (x, y)
        x, y = p
        # Check distances
        for q in existing:
            d = dist_sq(p, q)
            if not is_square(d):
                return False
        # Check collinearity
        for i, j in combinations(range(len(existing)), 2):
            if is_collinear(p, existing[i], existing[j]):
                return False
        # Check concyclicity
        if len(existing) >= 3:
            for i, j, k in combinations(range(len(existing)), 3):
                if is_concyclic(p, existing[i], existing[j], existing[k]):
                    return False
        return True

    # Search for new points using circle intersection
    # For each pair of existing points, find rational points at integer distances from both.
    
    def find_candidates_from_pair(p1, p2, max_r=200):
        # p1, p2 are (x, y) integers
        # d is integer distance
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        d_sq = dx*dx + dy*dy
        d = math.isqrt(d_sq)
        if d*d != d_sq:
            return []
        
        candidates = []
        # We look for r1, r2 such that the intersection of circles C(p1, r1) and C(p2, r2)
        # has rational coordinates.
        # The condition is that the area of the triangle formed by p1, p2, p is rational?
        # No, the coordinates must be rational.
        # Let's just iterate r1, r2 and check if the resulting x, y are rational.
        
        for r1 in range(1, max_r + 1):
            for r2 in range(1, max_r + 1):
                # Check triangle inequality
                if r1 + r2 < d or r1 + d < r2 or r2 + d < r1:
                    continue
                
                # Calculate intersection
                # x = p1_x + (r1^2 - r2^2 + d^2) / (2d) * (dx/d) ? No.
                # Let u = (r1^2 - r2^2 + d^2) / (2d). This is the projection of p1-p onto p1-p2?
                # Actually, the distance from p1 to the midpoint of the chord is u.
                # The height h = sqrt(r1^2 - u^2).
                # u must be rational. h must be rational.
                # u = (r1^2 - r2^2 + d^2) / (2d). Since r1, r2, d are integers, u is rational.
                # h^2 = r1^2 - u^2. We need h to be rational, so h^2 must be a square of a rational.
                # h^2 = r1^2 - ((r1^2 - r2^2 + d^2) / (2d))^2
                # Let num = r1^2 - r2^2 + d^2. u = num / (2d).
                # h^2 = r1^2 - num^2 / (4d^2) = (4d^2 r1^2 - num^2) / (4d^2).
                # We need 4d^2 r1^2 - num^2 to be a perfect square, say k^2.
                # Then h = k / (2d).
                
                num = r1*r1 - r2*r2 + d*d
                # h^2 * 4d^2 = 4d^2 r1^2 - num^2
                val = 4*d*d*r1*r1 - num*num
                if val < 0:
                    continue
                k = math.isqrt(val)
                if k*k != val:
                    continue
                
                # h = k / (2d)
                # The point P is at distance u from p1 along the line p1-p2, and distance h perpendicular.
                # p1 = (x1, y1), p2 = (x2, y2).
                # Vector v = p2 - p1 = (dx, dy). Length d.
                # Unit vector along v: (dx/d, dy/d).
                # Perpendicular unit vector: (-dy/d, dx/d) or (dy/d, -dx/d).
                # P = p1 + u * (dx/d, dy/d) +/- h * (-dy/d, dx/d)
                # P_x = x1 + u*dx/d +/- h*(-dy/d)
                # P_x = x1 + (num/(2d))*dx/d +/- (k/(2d))*(-dy/d)
                # P_x = x1 + (num*dx - k*dy) / (2d^2)  (for one sign)
                # P_x = x1 + (num*dx + k*dy) / (2d^2)  (for other sign)
                
                # Since x1, num, dx, k, dy, d are integers, P_x is rational.
                # We need P_x to be an integer? No, the task allows rational coordinates.
                # But the task says "rational x, y".
                # However, the output format is [x_num, x_den].
                # My current code assumes integers. I need to handle rationals.
                
                # Let's stick to integer coordinates for simplicity first, or check if denominator divides numerator.
                # If we find a rational point, we can scale the whole set? No, D is fixed.
                # The task allows rational coordinates.
                # I will modify the representation to use Fractions or tuples (num, den).
                
                # For now, let's just check if the coordinates are integers to fit the existing `can_add_point`.
                # If I want to support rationals, I need to rewrite the validation logic to use Fractions.
                
                # Let's try to find integer points first.
                # Condition for integer coordinates:
                # (num*dx - k*dy) must be divisible by 2d^2.
                # This is a strong condition.
                
                # Let's just collect the rational candidates and check them.
                # I will create a helper to convert rational to tuple (num, den) reduced.
                
                def to_frac(num, den):
                    if den < 0:
                        num = -num
                        den = -den
                    g = math.gcd(abs(num), den)
                    if g > 0:
                        num //= g
                        den //= g
                    return (num, den)
                
                # Candidate 1
                num_x1 = num*dx - k*dy
                num_y1 = num*dy + k*dx # Wait, perp is (-dy, dx). So y comp is h*dx/d.
                # P_y = y1 + u*dy/d +/- h*(dx/d)
                # P_y = y1 + (num*dy +/- k*dx) / (2d^2)
                
                den_x = 2*d*d
                den_y = 2*d*d
                
                # Check if integer
                if num_x1 % den_x == 0 and num_y1 % den_y == 0:
                    px = p1[0] + num_x1 // den_x
                    py = p1[1] + num_y1 // den_y
                    candidates.append((px, py))
                
                # Candidate 2 (other sign)
                num_x2 = num*dx + k*dy
                num_y2 = num*dy - k*dx
                if num_x2 % den_x == 0 and num_y2 % den_y == 0:
                    px = p1[0] + num_x2 // den_x
                    py = p1[1] + num_y2 // den_y
                    if (px, py) not in candidates:
                        candidates.append((px, py))
                        
        return candidates

    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        # Try to find a new point
        # Iterate through pairs
        n = len(best_set)
        for i in range(n):
            for j in range(i+1, n):
                if time.time() > deadline - 2:
                    break
                candidates = find_candidates_from_pair(best_set[i], best_set[j], 100)
                for p in candidates:
                    if p in best_set:
                        continue
                    if can_add_point(p, best_set):
                        best_set.append(p)
                        added_any = True
                        break
                if added_any:
                    break
            if added_any:
                break
            
    # Final verification
    if not validate_set(best_set):
        # If the extended set is invalid, fall back to the seed
        if validate_set(seed):
            best_set = seed
        else:
            # Try to find any valid set of 3 points
            # (0,0), (3,4), (0, 12) ?
            # Distances: 5, 12, sqrt(3^2 + 8^2) = sqrt(73) -> Not integer.
            # (0,0), (3,4), (4, 12) ?
            # d(0,0)-(4,12) = sqrt(16+144)=sqrt(160) No.
            # (0,0), (3,4), (7, 24) ?
            # d(0,0)-(7,24) = 25.
            # d(3,4)-(7,24) = sqrt(16 + 400) = sqrt(416) No.
            # Let's just use the seed. It is known to be valid.
            best_set = seed

    # Format output
    D = 1
    points_formatted = []
    for (x, y) in best_set:
        # x and y are integers
        points_formatted.append([[x, 1], [y, 1]])

    return {'D': D, 'points': points_formatted}
