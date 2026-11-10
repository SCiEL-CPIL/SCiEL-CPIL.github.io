import time
import random
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for evaluation overhead
    
    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def check_validity(points, D):
        """
        points: list of (x_num, x_den, y_num, y_den)
        D: int
        Returns True if all distances are rational, no 3 collinear, no 4 concyclic.
        """
        n = len(points)
        if n < 3:
            return True
        
        # 1. Check pairwise distances are rational
        for i in range(n):
            xi, xd_i, yi, yd_i = points[i]
            for j in range(i + 1, n):
                xj, xd_j, yj, yd_j = points[j]
                
                # dx = xi/xd_i - xj/xd_j
                dx_num = xi * xd_j - xj * xd_i
                dx_den = xd_i * xd_j
                
                # dy = yi/yd_i - yj/yd_j
                dy_num = yi * yd_j - yj * yd_i
                dy_den = yd_i * yd_j
                
                # dist_sq = dx^2 + D * dy^2
                # dist_sq = (dx_num/dx_den)^2 + D * (dy_num/dy_den)^2
                # = (dx_num^2 * dy_den^2 + D * dy_num^2 * dx_den^2) / (dx_den^2 * dy_den^2)
                
                num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
                den = dx_den * dx_den * dy_den * dy_den
                
                # For dist to be rational, dist_sq must be a rational square.
                # Since num/den is positive rational, we check if num * den is a perfect square integer?
                # No, num/den = (k/l)^2 => num*l^2 = den*k^2 => num*den = (k*l)^2.
                # So num * den must be a perfect square.
                if num == 0:
                    continue # Distinct points, so this shouldn't happen if points are distinct
                if not is_square(num * den):
                    return False

        # 2. Check no three collinear
        for i in range(n):
            xi, xd_i, yi, yd_i = points[i]
            for j in range(i + 1, n):
                xj, xd_j, yj, yd_j = points[j]
                # Slope m_ij
                # dy/dx = (yj/xd_j - yi/yd_i) / (xj/xd_j - xi/xd_i) ... wait, coords are (x, y*sqrt(D))
                # Collinearity in (x, y*sqrt(D)) plane.
                # Determinant: (xj-xi)*(yk-yi) - (yk-yi)*(xk-xi) = 0?
                # Points are P=(x, Y) where Y=y*sqrt(D).
                # Collinear if (xj-xi)*(yk-yi) - (yk-yi)*(xk-xi) is not quite right.
                # Slope m = (Yj-Yi)/(xj-xi).
                # Yj-Yi = (yj-yi)*sqrt(D).
                # So m = (yj-yi)*sqrt(D) / (xj-xi).
                # For 3 points to be collinear, slopes must be equal.
                # m_ij = m_ik => (yj-yi)*sqrt(D)/(xj-xi) = (yk-yi)*sqrt(D)/(xk-xi)
                # => (yj-yi)/(xj-xi) = (yk-yi)/(xk-xi)
                # This is equivalent to collinearity of (x, y) points in standard plane!
                # Because sqrt(D) is a common factor for all y-differences.
                # So we just check if (xi, yi), (xj, yj), (xk, yk) are collinear in Q^2.
                for k in range(j + 1, n):
                    xk, xd_k, yk, yd_k = points[k]
                    # Cross product of vectors (xj-xi, yj-yi) and (xk-xi, yk-yi)
                    # (xj/xd_j - xi/xd_i) * (yk/yd_k - yi/yd_i) - (yj/yd_j - yi/yd_i) * (xk/xd_k - xi/xd_i)
                    # This is messy. Use determinant:
                    # | xi yi 1 |
                    # | xj yj 1 | = 0
                    # | xk yk 1 |
                    # xi(yj-yk) - yi(xj-xk) + (xj*yk - xk*yj) = 0
                    # Multiply by common denominators to clear fractions.
                    # Let X_i = xi, Y_i = yi (as Fractions)
                    # det = X_i*(Y_j-Y_k) - Y_i*(X_j-X_k) + (X_j*Y_k - X_k*Y_j)
                    Xi = Fraction(xi, xd_i)
                    Yi = Fraction(yi, yd_i)
                    Xj = Fraction(xj, xd_j)
                    Yj = Fraction(yj, yd_j)
                    Xk = Fraction(xk, xd_k)
                    Yk = Fraction(yk, yd_k)
                    
                    det = Xi * (Yj - Yk) - Yi * (Xj - Xk) + (Xj * Yk - Xk * Yj)
                    if det == 0:
                        return False

        # 3. Check no four concyclic
        # Four points P1, P2, P3, P4 are concyclic if they lie on a circle.
        # In the metric space with distance d(P,Q) = sqrt((x1-x2)^2 + D(y1-y2)^2),
        # the condition for concyclicity is different from Euclidean plane.
        # However, the problem states "no four concyclic" in the context of the geometry defined by the points.
        # Usually, this refers to the Euclidean geometry of the embedding space?
        # The points are in R^2: (x, y*sqrt(D)).
        # So we check concyclicity in R^2.
        # P1(x1, Y1), P2(x2, Y2), P3(x3, Y3), P4(x4, Y4) where Yi = yi*sqrt(D).
        # Circle equation: x^2 + Y^2 + Ax + BY + C = 0.
        # Substitute: x^2 + D*y^2 + Ax + B*sqrt(D)*y + C = 0.
        # For this to hold for rational x,y, the coefficients of 1 and sqrt(D) must match separately?
        # x^2 + D*y^2 + C + A*x + (B*sqrt(D))*y = 0.
        # Rational part: x^2 + D*y^2 + C + A*x = 0.
        # Irrational part: B*y = 0 => B=0 (if y != 0).
        # If B=0, the circle is symmetric with respect to x-axis in the Y-coordinate?
        # Actually, if B=0, the equation is x^2 + D*y^2 + A*x + C = 0.
        # This is an ellipse in (x,y) plane, but a circle in (x, Y) plane.
        # So we need to check if there exist A, B, C such that the equation holds for all 4 points.
        # This is a linear system in A, B, C for each point:
        # A*xi + B*Yi + C = -(xi^2 + Yi^2)
        # Since Yi = yi*sqrt(D), we have:
        # A*xi + B*sqrt(D)*yi + C = -(xi^2 + D*yi^2)
        # Separating rational and irrational parts:
        # A*xi + C = -(xi^2 + D*yi^2)  (1)
        # B*yi = 0                       (2)
        
        # From (2), if yi != 0, then B must be 0.
        # If all yi are 0, points are on x-axis, collinear (already handled).
        # So assume at least one yi != 0, then B=0.
        # Then we need to check if there exist A, C such that A*xi + C = -(xi^2 + D*yi^2) for all 4 points.
        # This is a system of 4 equations for 2 unknowns.
        # We can check if the first 3 points determine A, C, and if the 4th satisfies it.
        # Or simpler: The points (xi, xi^2 + D*yi^2) must be collinear?
        # Let Zi = xi^2 + D*yi^2.
        # Equation: A*xi + C = -Zi => Zi = -A*xi - C.
        # So Zi is a linear function of xi.
        # Thus, the points (xi, Zi) must be collinear in the (x, z) plane.
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        # Check if (xi, Zi), (xj, Zj), (xk, Zk), (xl, Zl) are concyclic in R^2.
                        # This reduces to checking if (xi, Zi) are collinear for all 4?
                        # Wait. The condition B=0 came from "irrational part B*yi = 0".
                        # This holds if yi is not 0.
                        # If yi=0 for some points, it's more complex.
                        # But generally, if B=0, the condition is that (xi, Zi) are collinear.
                        # Let's check collinearity of (xi, Zi) for the 4 points.
                        Xi = Fraction(xi, xd_i)
                        Yi = Fraction(yi, yd_i)
                        Zi = Xi * Xi + D * Yi * Yi
                        
                        Xj = Fraction(xj, xd_j)
                        Yj = Fraction(yj, yd_j)
                        Zj = Xj * Xj + D * Yj * Yj
                        
                        Xk = Fraction(xk, xd_k)
                        Yk = Fraction(yk, yd_k)
                        Zk = Xk * Xk + D * Yk * Yk
                        
                        Xl = Fraction(xl, xd_l)
                        Yl = Fraction(yl, yd_l)
                        Zl = Xl * Xl + D * Yl * Yl
                        
                        # Check if (Xi, Zi), (Xj, Zj), (Xk, Zk) are collinear.
                        # If not, they define a unique circle (with B=0). Check if (Xl, Zl) is on it.
                        # If yes, they are concyclic.
                        # Collinearity of (X,Z):
                        # (Zj-Zi)/(Xj-Xi) == (Zk-Zi)/(Xk-Xi)
                        # (Zj-Zi)*(Xk-Xi) == (Zk-Zi)*(Xj-Xi)
                        
                        # To avoid division by zero, use cross product.
                        # det = (Xj-Xi)*(Zk-Zi) - (Zj-Zi)*(Xk-Xi)
                        det1 = (Xj - Xi) * (Zk - Zi) - (Zj - Zi) * (Xk - Xi)
                        if det1 != 0:
                            # Points i,j,k are not collinear in (X,Z) plane.
                            # They define a line? No, they define a circle in (x,Y) plane.
                            # Wait, the condition for concyclicity with B=0 is that (Xi, Zi) are collinear?
                            # No. The equation is Zi = -A*Xi - C. This is a LINE in (X,Z) plane.
                            # So if (Xi, Zi) are collinear, then B=0 works.
                            # If (Xi, Zi) are NOT collinear, then B cannot be 0?
                            # If B != 0, then we need B*yi = 0 for all i, which implies yi=0 for all i.
                            # If yi=0 for all i, they are on x-axis, collinear (handled).
                            # So if any yi != 0, B MUST be 0.
                            # Therefore, concyclicity is equivalent to (Xi, Zi) being collinear.
                            # So if det1 != 0, they are NOT concyclic (with B=0).
                            # But wait, if det1 != 0, does it mean they are concyclic with B != 0?
                            # No, because B != 0 requires yi=0.
                            # So if det1 != 0, they are NOT concyclic.
                            # If det1 == 0, they are collinear in (X,Z), so they ARE concyclic (with B=0).
                            # So we just need to check if ANY 4 points have (Xi, Zi) collinear.
                            # Actually, if (Xi, Zi) are collinear for 3 points, the 4th must also be on that line to be concyclic.
                            # So we check if (Xi, Zi), (Xj, Zj), (Xk, Zk), (Xl, Zl) are all collinear.
                            pass # Logic: if they are collinear in (X,Z), return False.
                        else:
                            # i,j,k are collinear in (X,Z). Check if l is on the same line.
                            det2 = (Xj - Xi) * (Zl - Zi) - (Zj - Zi) * (Xl - Xi)
                            if det2 == 0:
                                return False # All 4 are concyclic

        return True

    best_points = None
    best_D = 1
    best_size = 0
    
    # Fallback: The known 5-point set
    fallback = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = [[x, 1, y, 1] for x, y in fallback]
    best_D = 1
    best_size = 5
    
    # Search for better sets
    # Try small squarefree D
    D_candidates = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95, 97]
    
    # Candidate pool for coordinates
    # We use integers for simplicity in search, then convert to Fractions
    coord_range = range(-20, 21)
    
    for D in D_candidates:
        if time.time() - start_time > timeout:
            break
            
        # Generate candidate points (x, y) with x,y integers
        # We want to find a set of points in this field.
        # Greedy insertion.
        # Start with a random valid set of 3 points.
        
        # To speed up, we pre-filter pairs with rational distances.
        # But for small range, we can just check on the fly.
        
        # Let's try to find a set of size 6.
        # Random restarts.
        for _ in range(50): # 50 restarts per D
            if time.time() - start_time > timeout:
                break
                
            # Pick 3 random distinct points
            candidates = [(x, y) for x in coord_range for y in coord_range]
            random.shuffle(candidates)
            
            current_set = []
            for p in candidates:
                if len(current_set) >= 6:
                    break
                # Check if adding p to current_set is valid
                # We need to check distances, collinearity, concyclicity.
                # For speed, we only check against existing points.
                valid = True
                for q in current_set:
                    # Check distance p-q
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    dist_sq = dx * dx + D * dy * dy
                    if not is_square(dist_sq):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check collinearity with all pairs in current_set
                if len(current_set) >= 2:
                    for i in range(len(current_set)):
                        for j in range(i + 1, len(current_set)):
                            p1 = current_set[i]
                            p2 = current_set[j]
                            # Cross product of (p-p1) and (p2-p1)
                            # (px-p1x)*(py-p1y) - (py-p1y)*(px-p1x) ... wait
                            # (px-p1x)*(p2y-p1y) - (py-p1y)*(p2x-p1x)
                            det = (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])
                            if det == 0:
                                valid = False
                                break
                        if not valid:
                            break
                
                if not valid:
                    continue
                
                # Check concyclicity with all triples in current_set
                if len(current_set) >= 3:
                    for i in range(len(current_set)):
                        for j in range(i + 1, len(current_set)):
                            for k in range(j + 1, len(current_set)):
                                p1 = current_set[i]
                                p2 = current_set[j]
                                p3 = current_set[k]
                                # Check if p, p1, p2, p3 are concyclic.
                                # This requires the (X, Z) collinearity check.
                                # Z = X^2 + D*Y^2
                                Zp = p[0]**2 + D * p[1]**2
                                Z1 = p1[0]**2 + D * p1[1]**2
                                Z2 = p2[0]**2 + D * p2[1]**2
                                Z3 = p3[0]**2 + D * p3[1]**2
                                
                                # Check if (Xp, Zp), (X1, Z1), (X2, Z2), (X3, Z3) are collinear.
                                # We can check if (X1, Z1), (X2, Z2), (X3, Z3) are collinear.
                                # If not, they define a unique line (circle). Check if (Xp, Zp) is on it.
                                # If yes, concyclic.
                                # If (X1, Z1), (X2, Z2), (X3, Z3) are collinear, then (Xp, Zp) must also be on that line.
                                
                                # Slope 12 = (Z2-Z1)/(X2-X1)
                                # Slope 13 = (Z3-Z1)/(X3-X1)
                                # If X2==X1, vertical.
                                
                                # Use cross product for collinearity of 3 points (X1,Z1), (X2,Z2), (X3,Z3)
                                det123 = (X2 - X1) * (Z3 - Z1) - (Z2 - Z1) * (X3 - X1)
                                
                                if det123 != 0:
                                    # Not collinear. Check if p is on the circle defined by 1,2,3.
                                    # The circle passes through 1,2,3.
                                    # Equation in (X,Z): Z = A*X + B? No, that's line.
                                    # Wait, my previous deduction was: Concyclic <=> (X,Z) collinear.
                                    # This deduction assumed B=0.
                                    # If (X1,Z1), (X2,Z2), (X3,Z3) are NOT collinear, then B cannot be 0?
                                    # If B != 0, then yi=0 for all i.
                                    # So if any yi != 0, B MUST be 0.
                                    # So if (X1,Z1), (X2,Z2), (X3,Z3) are NOT collinear, then NO circle (with B=0) passes through them.
                                    # And no circle with B!=0 passes through them (unless yi=0).
                                    # So if any yi != 0, and (X,Z) not collinear, they are NOT concyclic.
                                    # So we only need to check if they ARE collinear.
                                    pass
                                else:
                                    # They are collinear in (X,Z). Check if p is on this line.
                                    det12p = (X2 - X1) * (Zp - Z1) - (Z2 - Z1) * (Xp - X1)
                                    if det12p == 0:
                                        valid = False
                                        break
                            if not valid:
                                break
                
                if valid:
                    current_set.append(p)
                    
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_D = D
                best_points = [[x, 1, y, 1] for x, y in current_set]
                # Optional: try to extend further? Greedy is fast.

    # Convert best_points to required format
    # best_points is list of [x_num, x_den, y_num, y_den]
    # Required: [[[x_num, x_den], [y_num, y_den]], ...]
    result_points = []
    for p in best_points:
        x_num, x_den, y_num, y_den = p
        result_points.append([[x_num, x_den], [y_num, y_den]])
    
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
