"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import isqrt, gcd
from fractions import Fraction
import time
import random

# EVOLVE-BLOCK-START

def _sqfree_part(n):
    """Return (d, k) such that n = d * k^2 where d is squarefree."""
    if n < 0:
        return (-1, isqrt(-n) ** 2 * -1)
    if n == 0:
        return (1, 0)
    n = abs(n)
    result = 1
    k = 1
    for p in range(2, min(isqrt(n) + 1, 100000)):
        if n % (p * p) == 0:
            while n % (p * p) == 0:
                n //= (p * p)
                k *= p
    if n > 1:
        result *= n
    if k > 1:
        return (result, k * k)
    return (result, 1)


def _is_square(n):
    """Check if n is a perfect square."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def _rational_sqrt(n):
    """If n is a perfect square, return (r, 1) where r*r = n. Otherwise return None."""
    if _is_square(n):
        return (isqrt(n), 1)
    return None


def _check_all_conditions(points, D):
    """
    Check all conditions for a set of points with given D.
    points: list of (x_num, x_den, y_num, y_den) representing (x, y*sqrt(D))
    Returns True if all conditions are satisfied.
    """
    n = len(points)
    if n < 3:
        return False

    # Convert to Fractions
    pts = []
    for (xn, xd, yn, yd) in points:
        x = Fraction(xn, xd)
        y = Fraction(yn, yd)
        pts.append((x, y))

    # Check all pairwise squared distances are rational
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            sq_dist = dx * dx + D * dy * dy
            # sq_dist must be non-negative and its numerator/denominator must satisfy
            # that it's a rational number (always true for Fraction)
            # But we need the DISTANCE to be rational, i.e., sq_dist must be a perfect square of a rational
            if sq_dist < 0:
                return False
            # sq_dist is a Fraction. Check if it's a perfect square of a rational.
            num = sq_dist.numerator
            den = sq_dist.denominator
            if not _is_square(num) or not _is_square(den):
                return False

    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                # Collinear if (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0
                det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if det == 0:
                    return False

    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]

                    # Four points are concyclic iff the determinant of the
                    # circle equation is zero.
                    # Circle: x^2 + y^2*D + ax + by + c = 0
                    # For point (x, y): x^2 + y^2*D + ax + by + c = 0
                    # This is a linear system in a, b, c for each point.
                    # 4 points concyclic iff the 4x4 determinant is 0.

                    def circle_det(x, y):
                        return (x * x + y * y * D, x, y, 1)

                    M = []
                    for (x, y) in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
                        M.append([x * x + y * y * D, x, y, 1])

                    # Compute determinant using Gaussian elimination
                    det = _det4(M)
                    if det == 0:
                        return False

    return True


def _det4(M):
    """Compute determinant of a 4x4 matrix of Fractions using Gaussian elimination."""
    M = [row[:] for row in M]
    n = 4
    det = Fraction(1, 1)
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(col, n):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            return Fraction(0, 1)
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            det = -det
        det *= M[col][col]
        # Eliminate below
        for row in range(col + 1, n):
            if M[row][col] != 0:
                factor = M[row][col] / M[col][col]
                for k in range(col, n):
                    M[row][k] -= factor * M[col][k]
    return det


def _search_integer_points(max_coord=200, max_points=8, time_limit=100):
    """
    Search for sets of integer points with rational distances, no three collinear, no four concyclic.
    Start from known 5-point sets and try to add more.
    """
    start_time = time.time()

    # Start with the known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    # Try adding points one by one
    best = list(base_points)

    # Generate candidate points in a grid
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in base_points:
                candidates.append((x, y))

    # Shuffle candidates for variety
    random.seed(42)
    random.shuffle(candidates)

    # Try adding candidates to the set
    current = list(base_points)
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        # Check if cand can be added
        new_set = current + [cand]
        if _check_all_conditions([(x, 1, y, 1) for (x, y) in new_set], 1):
            current = new_set
            if len(current) > len(best):
                best = list(current)
            if len(best) >= max_points:
                break

    # Try different starting sets
    # Use random local search
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        # Start with 4 random points from a larger grid
        random.seed(attempt)
        start_pts = []
        used = set()
        for _ in range(4):
            while True:
                x = random.randint(-150, 150)
                y = random.randint(-150, 150)
                if (x, y) not in used:
                    used.add((x, y))
                    start_pts.append((x, y))
                    break

        # Check initial set
        if not _check_all_conditions([(x, 1, y, 1) for (x, y) in start_pts], 1):
            continue

        current = list(start_pts)
        # Try adding points
        for cand in candidates:
            if time.time() - start_time > time_limit:
                break
            if cand in current:
                continue
            new_set = current + [cand]
            if _check_all_conditions([(x, 1, y, 1) for (x, y) in new_set], 1):
                current = new_set
                if len(current) > len(best):
                    best = list(current)
                if len(best) >= max_points:
                    break

    return best


def _search_with_D(D, max_coord=100, max_points=8, time_limit=50):
    """Search for points with a given D."""
    start_time = time.time()
    best = []

    # Try small integer y values and rational x
    for y1 in range(-max_coord, max_coord + 1):
        for y2 in range(y1 + 1, max_coord + 1):
            if time.time() - start_time > time_limit:
                break

    # Simpler: just use integer points with D=1 for now
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Start with the known 5-point solution
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = list(base_points)
    best_D = 1

    # Try to extend with more points using integer search
    if time.time() < deadline - 5:
        extended = _search_integer_points(
            max_coord=300,
            max_points=8,
            time_limit=min(100, deadline - time.time() - 5)
        )
        if len(extended) > len(best):
            best = extended
            best_D = 1

    # Verify the best solution
    pts = [(x, 1, y, 1) for (x, y) in best]
    if not _check_all_conditions(pts, best_D):
        # Fallback to base
        best = list(base_points)
        best_D = 1

    return {
        'D': best_D,
        'points': [[[x, 1], [y, 1]] for (x, y) in best]
    }
# EVOLVE-BLOCK-END
