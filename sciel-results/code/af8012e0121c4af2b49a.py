"""Combined program: hexagon attempt + 5-point D=1 base with fast power-of-point extension."""
import time
from fractions import Fraction
from math import isqrt

# EVOLVE-BLOCK-START

def _is_square_int(n):
    """Fast perfect-square test for non-negative integers."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_square_frac(fr):
    """Check if a Fraction is a perfect square of a rational."""
    if fr <= 0:
        return False
    return _is_square_int(fr.numerator) and _is_square_int(fr.denominator)

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) == (xk - xi) * (yj - yi):
                    return False
    return True

def _check_no_four_concyclic(points, D):
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
                    term1 = v1 * (a2 * (b3 - b4) - a3 * (b2 - b4) + a4 * (b2 - b3))
                    term2 = -a1 * (v2 * (b3 - b4) - v3 * (b2 - b4) + v4 * (b2 - b3))
                    term3 = b1 * (v2 * (a3 - a4) - v3 * (a2 - a4) + v4 * (a2 - a3))
                    term4 = -(v2 * (a3 * b4 - b3 * a4) - v3 * (a2 * b4 - b2 * a4) + v4 * (a2 * b3 - b2 * a3))
                    if term1 + term2 + term3 + term4 == 0:
                        return False
    return True

def _check_all_pairwise_distances_rational(points, D):
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + D * dy * dy
            if dist_sq <= 0:
                return False
            if not _is_square_frac(dist_sq):
                return False
    return True

def _is_valid(points, D):
    return (_check_no_three_collinear(points) and
            _check_no_four_concyclic(points, D) and
            _check_all_pairwise_distances_rational(points, D))

def _try_extend(points, D, x_range, y_range, time_limit):
    """Try to add one point to `points` from the given grid range."""
    n = len(points)
    # Precompute line coefficients for all pairs (for collinearity check with new point)
    line_coeffs = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            A = yi - yj
            B = xj - xi
            C = xj * yi - xi * yj
            line_coeffs.append((A, B, C))

    # Precompute power-of-point for all triples
    triple_powers = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                vi = xi * xi + D * yi * yi
                vj = xj * xj + D * yj * yj
                vk = xk * xk + D * yk * yk
                a11 = xj - xi
                a12 = yj - yi
                a21 = xk - xi
                a22 = yk - yi
                b1 = -(vj - vi)
                b2 = -(vk - vi)
                det = a11 * a22 - a12 * a21
                if det == 0:
                    continue
                Ac = (b1 * a22 - a12 * b2) / det
                Bc = (a11 * b2 - b1 * a21) / det
                Cc = -(vi + Ac * xi + Bc * yi)
                triple_powers.append((Ac, Bc, Cc))

    existing_set = set()
    for p in points:
        existing_set.add((p[0], p[1]))

    for x in x_range:
        for y in y_range:
            if time.time() > time_limit:
                return None
            cand = (Fraction(x), Fraction(y))
            if cand in existing_set:
                continue
            cx, cy = cand
            # Check distances to all existing points
            valid = True
            for p in points:
                dx = p[0] - cx
                dy = p[1] - cy
                dist_sq = dx * dx + D * dy * dy
                if dist_sq <= 0 or not _is_square_frac(dist_sq):
                    valid = False
                    break
            if not valid:
                continue
            # Check collinearity: new point against all precomputed lines
            for (A, B, C) in line_coeffs:
                if A * cx + B * cy + C == 0:
                    valid = False
                    break
            if not valid:
                continue
            # Check concyclicity: new point against all precomputed triple circles
            vcx = cx * cx + D * cy * cy
            for (Ac, Bc, Cc) in triple_powers:
                if vcx + Ac * cx + Bc * cy + Cc == 0:
                    valid = False
                    break
            if not valid:
                continue
            return cand

    return None

def _search_extension(current, D, deadline, max_target=8):
    """Adaptive grid search to extend point set."""
    current = list(current)
    
    while len(current) < max_target and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining < 3:
            break
        
        # Adaptive grid: larger as we have more points (harder to extend)
        base_extent = 15 + 8 * len(current)
        
        # Try multiple grid configurations
        grids = [
            (range(-base_extent, base_extent + 1), range(-base_extent, base_extent + 1)),
            (range(-base_extent, base_extent + 1), range(-base_extent, base_extent + 1)),  # same, for symmetry
            (range(-base_extent - 5, base_extent + 6), range(-base_extent - 3, base_extent + 4)),
            (range(-base_extent + 3, base_extent + 4), range(-base_extent + 7, base_extent + 8)),
            (range(-base_extent - 10, base_extent + 11), range(-base_extent - 8, base_extent + 9)),
        ]
        
        # Allocate time across grids
        time_per_grid = max(2, remaining * 0.8 / len(grids))
        
        found = False
        for x_range, y_range in grids:
            if time.time() > deadline - 5:
                break
            t_limit = min(deadline - 3, time.time() + time_per_grid)
            new_pt = _try_extend(current, D, x_range, y_range, t_limit)
            if new_pt is not None:
                current.append(new_pt)
                found = True
                break
        
        if not found:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def format_result(points, D):
        result_points = []
        for x, y in points:
            result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        return {'D': D, 'points': result_points}

    # ---- Try 6-point hexagon with D=3 ----
    hex_points = [(Fraction(2), Fraction(0)),
                  (Fraction(1), Fraction(1)),
                  (Fraction(-1), Fraction(1)),
                  (Fraction(-2), Fraction(0)),
                  (Fraction(-1), Fraction(-1)),
                  (Fraction(1), Fraction(-1))]
    D_hex = 3

    best_points = None
    best_D = 1

    if _is_valid(hex_points, D_hex):
        best_points = hex_points
        best_D = D_hex
    else:
        # ---- 5-point D=1 base (integer points) ----
        base = [(Fraction(0), Fraction(0)),
                (Fraction(-56), Fraction(33)),
                (Fraction(-16), Fraction(63)),
                (Fraction(0), Fraction(33)),
                (Fraction(16), Fraction(63))]
        if _is_valid(base, 1):
            best_points = base
            best_D = 1
        else:
            # Fallback: 3 points
            best_points = [(Fraction(0), Fraction(0)),
                           (Fraction(1), Fraction(0)),
                           (Fraction(0), Fraction(1))]
            best_D = 1

    # ---- Extension search ----
    current = list(best_points)
    D_current = best_D
    
    # Try to extend with current D
    extended = _search_extension(current, D_current, deadline, max_target=8)
    if len(extended) > len(best_points):
        best_points = extended
        best_D = D_current
    
    # If we didn't get 6+ points, try alternative D values with small bases
    if len(best_points) < 6 and time.time() < deadline - 20:
        alt_Ds = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]
        for D_alt in alt_Ds:
            if time.time() > deadline - 10:
                break
            # Try small integer grids for this D
            small_base = [(Fraction(0), Fraction(0)),
                          (Fraction(1), Fraction(0)),
                          (Fraction(0), Fraction(1)),
                          (Fraction(1), Fraction(1))]
            if _is_valid(small_base, D_alt):
                extended_alt = _search_extension(small_base, D_alt, deadline, max_target=8)
                if len(extended_alt) > len(best_points):
                    best_points = extended_alt
                    best_D = D_alt
    
    return format_result(best_points, best_D)

# EVOLVE-BLOCK-END
