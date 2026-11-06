"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import gcd

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n < 1:
        return False
    # Check if n is squarefree
    if n == 1:
        return True
    # Trial division up to sqrt(n)
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _check_no_three_collinear(points):
    """Check that no three points are collinear using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Points are (x_i, y_i*sqrt(D)), (x_j, y_j*sqrt(D)), (x_k, y_k*sqrt(D))
                # Collinearity condition: (x_j - x_i)(y_k*sqrt(D) - y_i*sqrt(D)) == (x_k - x_i)(y_j*sqrt(D) - y_i*sqrt(D))
                # This simplifies to: (x_j - x_i)(y_k - y_i) == (x_k - x_i)(y_j - y_i)
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                x_k, y_k = points[k]
                if (x_j - x_i) * (y_k - y_i) == (x_k - x_i) * (y_j - y_i):
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check that no four points are concyclic using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Four points (x_a, y_a*sqrt(D)), etc. are concyclic iff the determinant is zero
                    # Circle equation: x^2 + y^2* D + A*x + B*y*sqrt(D) + C = 0
                    # For point (x_i, y_i*sqrt(D)): x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
                    # This must hold for 4 points. We can eliminate B by considering the system.
                    # Actually, let's use the standard concyclicity test.
                    # Four points (x1,y1), (x2,y2), (x3,y3), (x4,y4) in the plane are concyclic iff:
                    # |x1^2+y1^2  x1  y1  1|
                    # |x2^2+y2^2  x2  y2  1| = 0
                    # |x3^2+y3^2  x3  y3  1|
                    # |x4^2+y4^2  x4  y4  1|
                    # But here our points are (x_i, y_i*sqrt(D)) in the Euclidean plane.
                    # So the squared distance from origin is x_i^2 + D*y_i^2.
                    # The concyclicity condition for points (x_i, y_i*sqrt(D)) is:
                    # The determinant of the matrix with rows [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1] is zero.
                    # However, since y_i*sqrt(D) involves sqrt(D), we need to handle this carefully.
                    # Let's write the circle equation as: x^2 + D*y^2 + A*x + B*y*sqrt(D) + C = 0
                    # For each point i: x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
                    # This is a linear system in A, B*sqrt(D), C. Let's denote B' = B*sqrt(D).
                    # Then: x_i^2 + D*y_i^2 + A*x_i + B'*y_i + C = 0
                    # We have 4 equations and 3 unknowns (A, B', C), so the system is consistent iff the 4x4 determinant is zero.
                    # The matrix is:
                    # [x_i^2 + D*y_i^2, x_i, y_i, 1] for i in {i,j,k,l}
                    # Wait, this is not quite right because B' is a single coefficient.
                    # Let me reconsider. The general circle equation in the (x, y*sqrt(D)) plane is:
                    # x^2 + (y*sqrt(D))^2 + A*x + B*(y*sqrt(D)) + C = 0
                    # x^2 + D*y^2 + A*x + B*sqrt(D)*y + C = 0
                    # Let C' = C, A' = A, B' = B*sqrt(D). Then:
                    # x^2 + D*y^2 + A'*x + B'*y + C' = 0
                    # For each point (x_i, y_i): x_i^2 + D*y_i^2 + A'*x_i + B'*y_i + C' = 0
                    # This is linear in A', B', C'. For 4 points, the system has a non-trivial solution iff the determinant is zero.
                    # The determinant is of the matrix:
                    # [x_i^2 + D*y_i^2, x_i, y_i, 1]
                    # [x_j^2 + D*y_j^2, x_j, y_j, 1]
                    # [x_k^2 + D*y_k^2, x_k, y_k, 1]
                    # [x_l^2 + D*y_l^2, x_l, y_l, 1]
                    # We need this determinant to be non-zero for all quadruples.
                    # Compute the determinant using exact rational arithmetic.
                    a1, b1 = points[i]
                    a2, b2 = points[j]
                    a3, b3 = points[k]
                    a4, b4 = points[l]
                    # Compute x^2 + D*y^2 for each point
                    v1 = a1 * a1 + D * b1 * b1
                    v2 = a2 * a2 + D * b2 * b2
                    v3 = a3 * a3 + D * b3 * b3
                    v4 = a4 * a4 + D * b4 * b4
                    # Compute the 4x4 determinant
                    # M = [[v1, a1, b1, 1], [v2, a2, b2, 1], [v3, a3, b3, 1], [v4, a4, b4, 1]]
                    # det(M) = v1*(a2*(b3*1 - 1*b4) - a3*(b2*1 - 1*b4) + a4*(b2*1 - b3*1))
                    #         - a1*(v2*(b3*1 - 1*b4) - v3*(b2*1 - 1*b4) + v4*(b2*1 - b3*1))
                    #         + b1*(v2*(a3*1 - 1*a4) - v3*(a2*1 - 1*a4) + v4*(a2*1 - a3*1))
                    #         - 1*(v2*(a3*b4 - b3*a4) - v3*(a2*b4 - b2*a4) + v4*(a2*b3 - b2*a3))
                    # Let's compute this step by step.
                    term1 = v1 * (a2 * (b3 - b4) - a3 * (b2 - b4) + a4 * (b2 - b3))
                    term2 = -a1 * (v2 * (b3 - b4) - v3 * (b2 - b4) + v4 * (b2 - b3))
                    term3 = b1 * (v2 * (a3 - a4) - v3 * (a2 - a4) + v4 * (a2 - a3))
                    term4 = -(v2 * (a3 * b4 - b3 * a4) - v3 * (a2 * b4 - b2 * a4) + v4 * (a2 * b3 - b2 * a3))
                    det = term1 + term2 + term3 + term4
                    if det == 0:
                        return False
    return True

def _check_all_pairwise_distances_rational(points, D):
    """Check that all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x_i, y_i = points[i]
            x_j, y_j = points[j]
            # Distance squared = (x_i - x_j)^2 + D*(y_i - y_j)^2
            dx = x_i - x_j
            dy = y_i - y_j
            dist_sq = dx * dx + D * dy * dy
            # Check if dist_sq is a perfect square of a rational number
            # dist_sq is a Fraction. We need to check if it's a square.
            num = dist_sq.numerator
            den = dist_sq.denominator
            # Check if num and den are perfect squares
            if num < 0:
                return False
            # Check if num is a perfect square
            if num == 0:
                return False  # Points must be distinct
            # Check perfect square
            r = int(num ** 0.5)
            if r * r != num and (r + 1) * (r + 1) != num:
                # More careful check for large numbers
                r = int(num ** 0.5)
                if r * r != num:
                    r += 1
                    if r * r != num:
                        return False
            r2 = int(den ** 0.5)
            if r2 * r2 != den:
                r2 += 1
                if r2 * r2 != den:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s margin
    
    # Start with the known 5-point configuration
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63) with D=1
    base_points = [(Fraction(0), Fraction(0)),
                   (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)),
                   (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]
    D = 1
    
    # Verify base configuration
    if not _check_no_three_collinear(base_points):
        return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in base_points]}
    if not _check_no_four_concyclic(base_points, D):
        return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in base_points]}
    if not _check_all_pairwise_distances_rational(base_points, D):
        return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in base_points]}
    
    best_points = base_points[:]
    
    # Try to extend by adding more points
    # We'll search for additional points with rational coordinates
    # that maintain all the constraints
    
    # Candidate points to try - we'll generate candidates systematically
    # Try points with small integer coordinates first
    
    def try_add_point(current_points, D, candidate):
        """Try adding a candidate point to the current set."""
        new_points = current_points + [candidate]
        if not _check_no_three_collinear(new_points):
            return None
        if not _check_no_four_concyclic(new_points, D):
            return None
        if not _check_all_pairwise_distances_rational(new_points, D):
            return None
        return new_points
    
    # Generate candidate points
    # Try integer points in a reasonable range
    max_coord = 200
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((Fraction(x), Fraction(y)))
    
    # Sort candidates by some heuristic - maybe distance from existing points
    # Try to add points one at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for candidate in candidates:
            if time.time() > deadline:
                break
            # Skip if candidate is already in the set
            if any(candidate[0] == p[0] and candidate[1] == p[1] for p in best_points):
                continue
            new_points = try_add_point(best_points, D, candidate)
            if new_points is not None:
                best_points = new_points
                improved = True
                break
    
    # Format the result
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}

# EVOLVE-BLOCK-END
