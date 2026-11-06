"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import gcd, isqrt
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n < 1:
        return False
    if n == 1:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _is_perfect_square(n):
    """Check if non-negative integer n is a perfect square."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_no_three_collinear(points):
    """Check that no three points are collinear using exact rational arithmetic.
    points is a list of (x, y) Fractions."""
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
                    # 4x4 determinant: |v_i, x_i, y_i, 1|
                    term1 = v1 * (a2 * (b3 - b4) - a3 * (b2 - b4) + a4 * (b2 - b3))
                    term2 = -a1 * (v2 * (b3 - b4) - v3 * (b2 - b4) + v4 * (b2 - b3))
                    term3 = b1 * (v2 * (a3 - a4) - v3 * (a2 - a4) + v4 * (a2 - a3))
                    term4 = -(v2 * (a3 * b4 - b3 * a4) - v3 * (a2 * b4 - b2 * a4) + v4 * (a2 * b3 - b2 * a3))
                    det = term1 + term2 + term3 + term4
                    if det == 0:
                        return False
    return True

def _check_all_pairwise_distances_rational(points, D):
    """Check that all pairwise squared distances are rational squares."""
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

def _valid_config(points, D):
    """Full validation of a configuration."""
    if len(points) < 3:
        return False
    # Check distinctness
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    if not _check_all_pairwise_distances_rational(points, D):
        return False
    return True

def _format_points(points):
    return [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Known 5-point configuration with D=1
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63)),
    ]
    D = 1

    best_points = base_points[:]

    # Verify base
    if not _valid_config(best_points, D):
        # Fallback to a simple valid 3-point config
        best_points = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
        D = 1
        if not _valid_config(best_points, D):
            best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        return {'D': D, 'points': _format_points(best_points)}

    # Strategy: Try to find 6+ point configurations by systematic search
    # Use rational parametrization and known constructions

    # Try various D values and point sets
    # First, try to extend the base with additional points

    def try_add_point(current_points, D, candidate):
        """Try adding a candidate point; return new list or None."""
        # Quick check: not already present
        for p in current_points:
            if p[0] == candidate[0] and p[1] == candidate[1]:
                return None
        new_points = current_points + [candidate]
        if not _check_no_three_collinear(new_points):
            return None
        if not _check_no_four_concyclic(new_points, D):
            return None
        if not _check_all_pairwise_distances_rational(new_points, D):
            return None
        return new_points

    # Generate candidates using rational parametrization of circles
    # For D=1, we need points (x,y) where all pairwise distances are rational
    # This means the squared distances must be perfect squares of rationals

    # Try to find 6-point configurations by searching
    # Use a smarter approach: start from known configurations

    # Known 6-point rational distance sets (searching literature patterns)
    # Let's try systematic search with small coordinates

    # For D=1: points (x,y) with integer coordinates, pairwise distances rational
    # Try integer lattice points

    max_coord = 100
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((Fraction(x), Fraction(y)))

    # Sort by distance from origin for better heuristics
    candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for candidate in candidates:
            if time.time() > deadline:
                break
            new_points = try_add_point(best_points, D, candidate)
            if new_points is not None:
                best_points = new_points
                improved = True
                # Re-sort candidates to prioritize points near existing ones
                break

    # If we still only have 5, try different D values or different starting configs
    if len(best_points) <= 5 and time.time() < deadline:
        # Try D=2
        D2 = 2
        # For D=2, points (x, y*sqrt(2)), need (dx)^2 + 2*(dy)^2 to be a perfect square
        # Try small integer x, y
        base2 = [
            (Fraction(0), Fraction(0)),
            (Fraction(2), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ]
        if _valid_config(base2, D2):
            # Try to extend
            candidates2 = []
            for x in range(-50, 51):
                for y in range(-50, 51):
                    candidates2.append((Fraction(x), Fraction(y)))
            best2 = base2[:]
            improved2 = True
            while improved2 and time.time() < deadline:
                improved2 = False
                for candidate in candidates2:
                    if time.time() > deadline:
                        break
                    new_pts = try_add_point(best2, D2, candidate)
                    if new_pts is not None:
                        best2 = new_pts
                        improved2 = True
                        break
            if len(best2) > len(best_points):
                best_points = best2
                D = D2

    # Try D=5
    if len(best_points) <= 5 and time.time() < deadline:
        D5 = 5
        base5 = [
            (Fraction(0), Fraction(0)),
            (Fraction(5), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ]
        if _valid_config(base5, D5):
            candidates5 = []
            for x in range(-50, 51):
                for y in range(-50, 51):
                    candidates5.append((Fraction(x), Fraction(y)))
            best5 = base5[:]
            improved5 = True
            while improved5 and time.time() < deadline:
                improved5 = False
                for candidate in candidates5:
                    if time.time() > deadline:
                        break
                    new_pts = try_add_point(best5, D5, candidate)
                    if new_pts is not None:
                        best5 = new_pts
                        improved5 = True
                        break
            if len(best5) > len(best_points):
                best_points = best5
                D = D5

    # Try a different approach: use the rational parametrization more aggressively
    # For D=1, generate points on rational circles
    if len(best_points) <= 5 and time.time() < deadline:
        # Generate points using Pythagorean triples for distances
        # Point at origin, then points at rational distances
        # Use parametrization: point at (2t/(1+t^2), (1-t^2)/(1+t^2)) * r for rational t, r
        # But we need ALL pairwise distances rational, not just from origin

        # Try a grid search with rational coordinates (numerator/denominator small)
        max_num = 30
        max_den = 10
        candidates_rational = []
        for den in range(1, max_den + 1):
            for num in range(-max_num * den, max_num * den + 1):
                x = Fraction(num, den)
                for den2 in range(1, max_den + 1):
                    for num2 in range(-max_num * den2, max_num * den2 + 1):
                        y = Fraction(num2, den2)
                        candidates_rational.append((x, y))

        # Deduplicate
        candidates_rational = list(set(candidates_rational))
        candidates_rational.sort(key=lambda p: abs(p[0]) + abs(p[1]))

        # Try to build a 6-point set from scratch
        # Start with origin
        trial = [(Fraction(0), Fraction(0))]
        for candidate in candidates_rational:
            if time.time() > deadline:
                break
            if candidate == (Fraction(0), Fraction(0)):
                continue
            new_trial = trial + [candidate]
            if _valid_config(new_trial, D):
                trial = new_trial
                if len(trial) >= 6:
                    break

        if len(trial) > len(best_points):
            best_points = trial

    # Final validation
    if not _valid_config(best_points, D):
        # Emergency fallback
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        D = 1

    return {'D': D, 'points': _format_points(best_points)}

# EVOLVE-BLOCK-END
