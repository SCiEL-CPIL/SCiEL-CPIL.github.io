"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import gcd
from math import isqrt

# EVOLVE-BLOCK-START

def _is_perfect_square(n):
    """Check if n is a perfect square using integer arithmetic."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_no_three_collinear(points):
    """Check that no three points are collinear using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
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
                    a1, b1 = points[i]
                    a2, b2 = points[j]
                    a3, b3 = points[k]
                    a4, b4 = points[l]
                    
                    v1 = a1 * a1 + D * b1 * b1
                    v2 = a2 * a2 + D * b2 * b2
                    v3 = a3 * a3 + D * b3 * b3
                    v4 = a4 * a4 + D * b4 * b4
                    
                    # Determinant of [[v1, a1, b1, 1], [v2, a2, b2, 1], [v3, a3, b3, 1], [v4, a4, b4, 1]]
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
            dx = x_i - x_j
            dy = y_i - y_j
            dist_sq = dx * dx + D * dy * dy
            if dist_sq <= 0:
                return False
            num = dist_sq.numerator
            den = dist_sq.denominator
            if not _is_perfect_square(num):
                return False
            if not _is_perfect_square(den):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s margin
    
    def format_result(points, D):
        result_points = []
        for x, y in points:
            result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        return {'D': D, 'points': result_points}
    
    # Strategy: Use regular hexagon vertices in Q(√3)
    # Regular hexagon centered at origin with radius r has vertices:
    # (r, 0), (r/2, r√3/2), (-r/2, r√3/2), (-r, 0), (-r/2, -r√3/2), (r/2, -r√3/2)
    # In our form (x, y√D) with D=3: (x, y√3)
    # So vertices are: (r, 0), (r/2, r/2), (-r/2, r/2), (-r, 0), (-r/2, -r/2), (r/2, -r/2)
    # Wait, that's not quite right. Let me recalculate.
    # Vertex (r/2, r√3/2) means x = r/2, y√3 = r√3/2, so y = r/2.
    # So the six vertices in (x, y) form are:
    # (r, 0), (r/2, r/2), (-r/2, r/2), (-r, 0), (-r/2, -r/2), (r/2, -r/2)
    
    # Let's use r = 2 to keep numbers small
    # Vertices: (2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)
    hex_points = [(Fraction(2), Fraction(0)),
                  (Fraction(1), Fraction(1)),
                  (Fraction(-1), Fraction(1)),
                  (Fraction(-2), Fraction(0)),
                  (Fraction(-1), Fraction(-1)),
                  (Fraction(1), Fraction(-1))]
    D_hex = 3
    
    # Verify hexagon
    hex_valid = (_check_no_three_collinear(hex_points) and
                 _check_no_four_concyclic(hex_points, D_hex) and
                 _check_all_pairwise_distances_rational(hex_points, D_hex))
    
    if hex_valid:
        best_points = hex_points
        best_D = D_hex
    else:
        # Fallback to 5-point configuration with D=1
        base_points = [(Fraction(0), Fraction(0)),
                       (Fraction(-56), Fraction(33)),
                       (Fraction(-16), Fraction(63)),
                       (Fraction(0), Fraction(33)),
                       (Fraction(16), Fraction(63))]
        D_base = 1
        
        if (_check_no_three_collinear(base_points) and
            _check_no_four_concyclic(base_points, D_base) and
            _check_all_pairwise_distances_rational(base_points, D_base)):
            best_points = base_points
            best_D = D_base
        else:
            # Extreme fallback: 3 points
            best_points = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
            best_D = 1
    
    # Try to extend to 7 points
    # Search for a 7th point
    if len(best_points) == 6 and time.time() < deadline - 10:
        # For the hexagon, we can try to add a point
        # The hexagon has high symmetry, so we need to be careful
        # Let's try integer points in a reasonable range
        
        x_min = -5
        x_max = 5
        y_min = -5
        y_max = 5
        
        found_seven = False
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if time.time() > deadline - 5:
                    break
                candidate = (Fraction(x), Fraction(y))
                
                # Skip if already in set
                if any(candidate[0] == p[0] and candidate[1] == p[1] for p in best_points):
                    continue
                
                # Check distances to all existing points
                valid = True
                for p in best_points:
                    dx = p[0] - candidate[0]
                    dy = p[1] - candidate[1]
                    dist_sq = dx * dx + best_D * dy * dy
                    if dist_sq <= 0:
                        valid = False
                        break
                    if not _is_perfect_square(dist_sq.numerator):
                        valid = False
                        break
                    if not _is_perfect_square(dist_sq.denominator):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check collinearity with all pairs
                new_points = best_points + [candidate]
                if not _check_no_three_collinear(new_points):
                    continue
                
                # Check concyclicity
                if not _check_no_four_concyclic(new_points, best_D):
                    continue
                
                # If we get here, it's a valid 7-point set
                best_points = new_points
                found_seven = True
                break
            if found_seven or time.time() > deadline - 5:
                break
    
    # Try to extend to 8 points if we have 7
    if len(best_points) == 7 and time.time() < deadline - 10:
        x_min = -8
        x_max = 8
        y_min = -8
        y_max = 8
        
        found_eight = False
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if time.time() > deadline - 5:
                    break
                candidate = (Fraction(x), Fraction(y))
                
                # Skip if already in set
                if any(candidate[0] == p[0] and candidate[1] == p[1] for p in best_points):
                    continue
                
                # Check distances to all existing points
                valid = True
                for p in best_points:
                    dx = p[0] - candidate[0]
                    dy = p[1] - candidate[1]
                    dist_sq = dx * dx + best_D * dy * dy
                    if dist_sq <= 0:
                        valid = False
                        break
                    if not _is_perfect_square(dist_sq.numerator):
                        valid = False
                        break
                    if not _is_perfect_square(dist_sq.denominator):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check collinearity with all pairs
                new_points = best_points + [candidate]
                if not _check_no_three_collinear(new_points):
                    continue
                
                # Check concyclicity
                if not _check_no_four_concyclic(new_points, best_D):
                    continue
                
                # If we get here, it's a valid 8-point set
                best_points = new_points
                found_eight = True
                break
            if found_eight or time.time() > deadline - 5:
                break
    
    # If we only have 5 points, try to extend to 6
    if len(best_points) == 5 and time.time() < deadline - 10:
        x_min = -60
        x_max = 20
        y_min = -10
        y_max = 80
        
        found_six = False
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if time.time() > deadline - 5:
                    break
                candidate = (Fraction(x), Fraction(y))
                
                # Skip if already in set
                if any(candidate[0] == p[0] and candidate[1] == p[1] for p in best_points):
                    continue
                
                # Check distances to all existing points
                valid = True
                for p in best_points:
                    dx = p[0] - candidate[0]
                    dy = p[1] - candidate[1]
                    dist_sq = dx * dx + best_D * dy * dy
                    if dist_sq <= 0:
                        valid = False
                        break
                    if not _is_perfect_square(dist_sq.numerator):
                        valid = False
                        break
                    if not _is_perfect_square(dist_sq.denominator):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check collinearity with all pairs
                new_points = best_points + [candidate]
                if not _check_no_three_collinear(new_points):
                    continue
                
                # Check concyclicity
                if not _check_no_four_concyclic(new_points, best_D):
                    continue
                
                # If we get here, it's a valid 6-point set
                best_points = new_points
                found_six = True
                break
            if found_six or time.time() > deadline - 5:
                break
    
    return format_result(best_points, best_D)

# EVOLVE-BLOCK-END
