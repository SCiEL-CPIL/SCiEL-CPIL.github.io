import math
from fractions import Fraction
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _is_square_frac(f):
    """Check if Fraction f is a perfect square of a rational."""
    if f < 0:
        return False
    n, d = f.numerator, f.denominator
    rn = math.isqrt(n)
    rd = math.isqrt(d)
    return rn * rn == n and rd * rd == d

def _sqrt_frac(f):
    """Compute sqrt of a Fraction that is known to be a perfect square."""
    n, d = f.numerator, f.denominator
    rn = math.isqrt(n)
    rd = math.isqrt(d)
    return Fraction(rn, rd)

def _dist_sq(p1, p2, D):
    """Squared distance between two points (x,y) in the (x, y*sqrt(D)) metric."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + Fraction(D) * dy * dy

def _is_rational_dist(p1, p2, D):
    d2 = _dist_sq(p1, p2, D)
    return _is_square_frac(d2)

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Determinant for concyclicity
                    # | x  y  x^2+y^2  1 |
                    # | x  y  x^2+y^2  1 | = 0
                    # | x  y  x^2+y^2  1 |
                    # | x  y  x^2+y^2  1 |
                    M = [
                        [p1[0], p1[1], p1[0]**2 + p1[1]**2, Fraction(1)],
                        [p2[0], p2[1], p2[0]**2 + p2[1]**2, Fraction(1)],
                        [p3[0], p3[1], p3[0]**2 + p3[1]**2, Fraction(1)],
                        [p4[0], p4[1], p4[0]**2 + p4[1]**2, Fraction(1)],
                    ]
                    # Compute 4x4 determinant
                    def det4(M):
                        return (M[0][0]*(M[1][1]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) + M[1][3]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]))
                             - M[0][1]*(M[1][0]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]))
                             + M[0][2]*(M[1][0]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) - M[1][1]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][1]-M[2][1]*M[3][0]))
                             - M[0][3]*(M[1][0]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]) - M[1][1]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]) + M[1][2]*(M[2][0]*M[3][1]-M[2][1]*M[3][0])))
                    if det4(M) == 0:
                        return False
    return True

def _validate(points, D):
    if len(points) < 3:
        return False
    if len(set(points)) != len(points):
        return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if not _is_rational_dist(points[i], points[j], D):
                return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def _try_extend(points, D, deadline, max_candidates=200):
    """Try to add one more point to the given set."""
    start = time.time()
    P = list(points)
    n = len(P)
    
    # For each pair of existing points, find intersection of circles
    # with rational radii, and check if the intersection point is valid
    best = None
    
    # Precompute all pairwise distances (squared) for reference
    pair_dists = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_dists[(i, j)] = _dist_sq(P[i], P[j], D)
    
    # Try different radius combinations
    # Use small rational radii: p/q where p, q are small
    radius_candidates = []
    for num in range(1, 30):
        for den in range(1, 10):
            r = Fraction(num, den)
            r2 = r * r
            radius_candidates.append((r2, r))
    
    # Sort by radius for early termination
    radius_candidates.sort(key=lambda x: x[0])
    
    # Try pairs of reference points
    ref_pairs = list(itertools.combinations(range(n), 2))
    
    for (i, j) in ref_pairs:
        if time.time() > deadline - 5:
            break
        P_i = P[i]
        P_j = P[j]
        d2_ij = pair_dists[(i, j)]
        
        if not _is_square_frac(d2_ij):
            continue
        d_ij = _sqrt_frac(d2_ij)
        
        # For each pair of radii
        for (r1_sq, r1) in radius_candidates:
            if time.time() > deadline - 5:
                break
            for (r2_sq, r2) in radius_candidates:
                if time.time() > deadline - 5:
                    break
                
                # Check triangle inequality roughly
                if r1 + r2 < d_ij or r1 + d_ij < r2 or r2 + d_ij < r1:
                    continue
                if r1 + r2 == d_ij or r1 + d_ij == r2 or r2 + d_ij == r1:
                    continue  # Degenerate, skip for now
                
                # Intersection of two circles
                # x = x_i + (r1^2 - r2^2 + d^2) / (2*d)
                x_offset = (r1_sq - r2_sq + d2_ij) / (2 * d_ij)
                x = P_i[0] + x_offset
                
                # (y - y_i)^2 = r1^2 - x_offset^2
                y_sq = r1_sq - x_offset * x_offset
                if y_sq < 0:
                    continue
                if not _is_square_frac(y_sq):
                    continue
                y_offset = _sqrt_frac(y_sq)
                
                candidates = [(x, P_i[1] + y_offset), (x, P_i[1] - y_offset)]
                
                for qx, qy in candidates:
                    Q = (qx, qy)
                    if Q in P:
                        continue
                    
                    # Check rational distance to all existing points
                    valid = True
                    for pk in P:
                        if not _is_rational_dist(Q, pk, D):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check no three collinear with new point
                    new_set = P + [Q]
                    if not _no_three_collinear(new_set):
                        continue
                    
                    # Check no four concyclic with new point
                    if not _no_four_concyclic(new_set):
                        continue
                    
                    best = new_set
                    return best
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_points = None
    best_D = None
    
    # Known 5-point set with D=1
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Verify these have rational distances
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    points = [(Fraction(x), Fraction(y)) for x, y in known_5]
    
    if _validate(points, D):
        best_points = points
        best_D = D
    
    # Try to extend to 6 points
    if best_points is not None:
        extended = _try_extend(best_points, best_D, deadline)
        if extended:
            best_points = extended
            best_D = best_D
    
    # Try other D values with smaller starting sets
    # Try D=2, D=3, etc. with simple configurations
    for D_test in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
        if time.time() > deadline - 10:
            break
        
        # Try to find a 3-point set for this D
        # Start with (0,0) and try to find two more points
        base = [(Fraction(0), Fraction(0))]
        
        # Try points (x, 0) and (0, y) type configurations
        found_3 = None
        for x1_num in range(1, 20):
            if time.time() > deadline - 10:
                break
            for y2_num in range(1, 20):
                if time.time() > deadline - 10:
                    break
                # Point 1: (x1, 0), Point 2: (0, y2)
                p1 = (Fraction(x1_num), Fraction(0))
                p2 = (Fraction(0), Fraction(y2_num))
                
                # Check dist(base, p1): x1^2 must be square -> x1_num^2 is square, ok
                # Check dist(base, p2): D*y2^2 must be square
                d2 = Fraction(D_test) * y2_num * y2_num
                if not _is_square_frac(d2):
                    continue
                # Check dist(p1, p2): x1^2 + D*y2^2 must be square
                d12 = x1_num * x1_num + D_test * y2_num * y2_num
                if not _is_square_frac(Fraction(d12)):
                    continue
                
                candidate_3 = [base[0], p1, p2]
                if _validate(candidate_3, D_test):
                    found_3 = candidate_3
                    break
            if found_3:
                break
        
        if found_3 is None:
            continue
        
        # Try to extend to 4, 5, 6 points
        current = found_3
        for _ in range(3):  # Try to add up to 3 more points
            if time.time() > deadline - 10:
                break
            extended = _try_extend(current, D_test, deadline)
            if extended:
                current = extended
            else:
                break
        
        if len(current) > (len(best_points) if best_points else 0):
            best_points = current
            best_D = D_test
    
    # Format output
    if best_points is None:
        # Fallback: return the known 5-point set
        best_points = [(Fraction(x), Fraction(y)) for x, y in known_5]
        best_D = 1
    
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
