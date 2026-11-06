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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def format_result(points, D):
        result_points = []
        for x, y in points:
            result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        return {'D': D, 'points': result_points}

    # ---- Try 6-point hexagon with D=3 ----
    # Vertices of regular hexagon in (x, y*sqrt(3)) form:
    # (2,0),(1,1),(-1,1),(-2,0),(-1,-1),(1,-1)
    # NOTE: These are concyclic (all on circle x^2+3y^2=4), so this will fail.
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

    # ---- Extension search for D=1 (fast integer path) ----
    if best_D == 1 and best_points is not None:
        # Convert to integer tuples for speed
        int_points = [(p[0].numerator, p[0].denominator, p[1].numerator, p[1].denominator) for p in best_points]
        # For D=1 with integer points, distances are sqrt(dx^2+dy^2)
        # Use a unified search with Fraction points but fast distance check

        def try_extend(points, D, x_range, y_range, time_limit):
            """Try to add one point to `points` from the given grid range."""
            n = len(points)
            # Precompute line coefficients for all pairs (for collinearity check with new point)
            # Line through (xi,yi) and (xj,yj): A*x + B*y + C = 0 where A=yi-yj, B=xj-xi, C=xj*yi-xi*yj
            line_coeffs = []
            for i in range(n):
                xi, yi = points[i]
                for j in range(i + 1, n):
                    xj, yj = points[j]
                    A = yi - yj
                    B = xj - xi
                    C = xj * yi - xi * yj
                    line_coeffs.append((A, B, C, i, j))

            # Precompute power-of-point for all triples
            # Pow(P; circle through A,B,C) = v_P - A_coeff*P - B_coeff*P - C_const
            # where circle: x^2 + D*y^2 + A_c*x + B_c*y + C_c = 0
            # For triple (i,j,k): solve for A_c, B_c, C_c
            # v_i + A_c*x_i + B_c*y_i + C_c = 0
            # v_j + A_c*x_j + B_c*y_j + C_c = 0
            # v_k + A_c*x_k + B_c*y_k + C_c = 0
            # Subtract first from second: (v_j-v_i) + A_c*(x_j-x_i) + B_c*(y_j-y_i) = 0
            # Subtract first from third: (v_k-v_i) + A_c*(x_k-x_i) + B_c*(y_k-y_i) = 0
            triple_powers = []  # list of (A_c, B_c, C_c, i, j, k)
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        xi, yi = points[i]
                        xj, yj = points[j]
                        xk, yk = points[k]
                        vi = xi * xi + D * yi * yi
                        vj = xj * xj + D * yj * yj
                        vk = xk * xk + D * yk * yk
                        # Solve 2x2 system:
                        # A_c*(x_j-x_i) + B_c*(y_j-y_i) = -(v_j - v_i)
                        # A_c*(x_k-x_i) + B_c*(y_k-y_i) = -(v_k - v_i)
                        a11 = xj - xi
                        a12 = yj - yi
                        a21 = xk - xi
                        a22 = yk - yi
                        b1 = -(vj - vi)
                        b2 = -(vk - vi)
                        det = a11 * a22 - a12 * a21
                        if det == 0:
                            continue  # Collinear triple (shouldn't happen)
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
                    for (A, B, C, i, j) in line_coeffs:
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

        # Try to extend from current size, searching progressively larger grids
        current = list(best_points)
        max_target = 8  # Aim for up to 8 points

        while len(current) < max_target and time.time() < deadline - 5:
            # Grid size grows with current size
            extent = 20 + 10 * len(current)
            x_range = range(-extent, extent + 1)
            y_range = range(-extent, extent + 1)
            t_limit = min(deadline - 3, time.time() + (deadline - 5 - time.time()) * 0.9)

            new_pt = try_extend(current, best_D, x_range, y_range, t_limit)
            if new_pt is not None:
                current.append(new_pt)
            else:
                # Try a second, offset grid
                if time.time() < deadline - 5:
                    x_range2 = range(-extent + 7, extent + 8)
                    y_range2 = range(-extent + 13, extent + 14)
                    t_limit2 = min(deadline - 3, time.time() + 30)
                    new_pt = try_extend(current, best_D, x_range2, y_range2, t_limit2)
                    if new_pt is not None:
                        current.append(new_pt)
                    else:
                        break

        if len(current) > len(best_points):
            best_points = current

    return format_result(best_points, best_D)

# EVOLVE-BLOCK-END
