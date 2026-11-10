"""Search for larger integral point sets in general position."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 130.0  # keep a 20s safety margin before the 150s kill

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = int(n ** 0.5)
    # adjust for floating point error
    for d in (-2, -1, 0, 1, 2):
        c = r + d
        if c >= 0 and c * c == n:
            return True
    return False

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _dist_sq_p2_p3(p2, p3):
    """p2,p3 are (x_num, x_den, y_num, y_den) integers."""
    x2n, x2d, y2n, y2d = p2
    x3n, x3d, y3n, y3d = p3
    # dx = x2 - x3
    dxn = x2n * x3d - x3n * x2d
    dxd = x2d * x3d
    # dy = y2 - y3
    dyn = y2n * y3d - y3n * y2d
    dyd = y2d * y3d
    # dist^2 = dx^2 + dy^2
    # = (dxn/dxd)^2 + (dyn/dyd)^2
    # = (dxn^2 * dyd^2 + dyn^2 * dxd^2) / (dxd^2 * dyd^2)
    num = dxn * dxn * dyd * dyd + dyn * dyn * dxd * dxd
    den = dxd * dxd * dyd * dyd
    g = _gcd(num, den)
    if g > 1:
        num //= g
        den //= g
    return num, den

def _check_pairwise_rational(points):
    """points: list of (x_num, x_den, y_num, y_den)."""
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            num, den = _dist_sq_p2_p3(points[i], points[j])
            if den != 1:
                return False
    return True

def _check_collinear(points):
    """Check no three collinear. points: list of (x_num, x_den, y_num, y_den)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1n, x1d, y1n, y1d = points[i]
                x2n, x2d, y2n, y2d = points[j]
                x3n, x3d, y3n, y3d = points[k]
                # (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0 ?
                # dx12 = x2-x1, dy12 = y2-y1, dx13 = x3-x1, dy13 = y3-y1
                dx12n = x2n * x1d - x1n * x2d
                dx12d = x2d * x1d
                dy12n = y2n * y1d - y1n * y2d
                dy12d = y2d * y1d
                dx13n = x3n * x1d - x1n * x3d
                dx13d = x3d * x1d
                dy13n = y3n * y1d - y1n * y3d
                dy13d = y3d * y1d
                # cross = dx12*dy13 - dy12*dx13
                num = dx12n * dy13n * dy12d * dx13d - dy12n * dx13n * dx12d * dy13d
                if num == 0:
                    return False
    return True

def _check_concyclic(points):
    """Check no four concyclic. points: list of (x_num, x_den, y_num, y_den)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1n, x1d, y1n, y1d = points[i]
                    x2n, x2d, y2n, y2d = points[j]
                    x3n, x3d, y3n, y3d = points[k]
                    x4n, x4d, y4n, y4d = points[l]
                    # Four points concyclic iff determinant is zero:
                    # | x^2+y^2, x, y, 1 | for each point = 0
                    # To avoid fractions, multiply row i by xid^2 * yid^2 * (something)
                    rows = []
                    for idx in (i, j, k, l):
                        xn, xd, yn, yd = points[idx]
                        Xnum = xn * xn * yd * yd + yn * yn * xd * xd
                        Xden = xd * xd * yd * yd
                        r1 = Xnum * xd * yd
                        r2 = xn * Xden * yd
                        r3 = yn * Xden * xd
                        r4 = Xden * xd * yd
                        rows.append([r1, r2, r3, r4])
                    # Compute determinant of 4x4 matrix
                    a, b, c, d = rows[0]
                    e, f, g, h = rows[1]
                    i2, j2, k2, l2 = rows[2]
                    m, n2, o, p = rows[3]
                    det = (a * (f * (k2 * p - l2 * o) - g * (j2 * p - l2 * n2) + h * (j2 * o - k2 * n2))
                           - b * (e * (k2 * p - l2 * o) - g * (i2 * p - l2 * m) + h * (i2 * o - k2 * m))
                           + c * (e * (j2 * p - l2 * n2) - f * (i2 * p - l2 * m) + h * (i2 * o - k2 * m))
                           - d * (e * (j2 * o - k2 * n2) - f * (i2 * o - k2 * m) + g * (i2 * n2 - j2 * m)))
                    if det == 0:
                        return False
    return True

def _validate(points_int):
    """points_int: list of (x_num, x_den, y_num, y_den) with den=1 for D=1."""
    if not _check_pairwise_rational(points_int):
        return False
    if not _check_collinear(points_int):
        return False
    if not _check_concyclic(points_int):
        return False
    return True

def _brute_force_search(max_coord=200, target_size=6):
    """Search for target_size points in [-max_coord, max_coord]^2."""
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    base_int = [(x, 1, y, 1) for x, y in base]
    
    # Try to add a 6th point
    for x in range(-200, 201):
        if time.time() > _DEADLINE:
            break
        for y in range(-200, 201):
            if time.time() > _DEADLINE:
                break
            # Check distance to all base points
            ok = True
            for bx, by in base:
                d2 = (x - bx) ** 2 + (y - by) ** 2
                if not _is_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            # Check distinctness
            if (x, y) in base:
                continue
            # Check collinearity with pairs of base points
            collinear = False
            for i in range(5):
                for j in range(i + 1, 5):
                    if (x - base[i][0]) * (base[j][1] - base[i][1]) == (y - base[i][1]) * (base[j][0] - base[i][0]):
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            # Check concyclicity with triples of base points
            concyclic = False
            for i in range(5):
                for j in range(i + 1, 5):
                    for k in range(j + 1, 5):
                        pts = [base[i], base[j], base[k], (x, y)]
                        pts_int = [(px, 1, py, 1) for px, py in pts]
                        x1n, x1d, y1n, y1d = pts_int[0]
                        x2n, x2d, y2n, y2d = pts_int[1]
                        x3n, x3d, y3n, y3d = pts_int[2]
                        x4n, x4d, y4n, y4d = pts_int[3]
                        rows = []
                        for idx in range(4):
                            xn, xd, yn, yd = pts_int[idx]
                            Xnum = xn * xn * yd * yd + yn * yn * xd * xd
                            Xden = xd * xd * yd * yd
                            r1 = Xnum * xd * yd
                            r2 = xn * Xden * yd
                            r3 = yn * Xden * xd
                            r4 = Xden * xd * yd
                            rows.append([r1, r2, r3, r4])
                        a, b, c, d = rows[0]
                        e, f, g, h = rows[1]
                        i2, j2, k2, l2 = rows[2]
                        m, n2, o, p = rows[3]
                        det = (a * (f * (k2 * p - l2 * o) - g * (j2 * p - l2 * n2) + h * (j2 * o - k2 * n2))
                               - b * (e * (k2 * p - l2 * o) - g * (i2 * p - l2 * m) + h * (i2 * o - k2 * m))
                               + c * (e * (j2 * p - l2 * n2) - f * (i2 * p - l2 * m) + h * (i2 * o - k2 * m))
                               - d * (e * (j2 * o - k2 * n2) - f * (i2 * o - k2 * m) + g * (i2 * n2 - j2 * m)))
                        if det == 0:
                            concyclic = True
                            break
                        if time.time() > _DEADLINE:
                            break
                    if concyclic or time.time() > _DEADLINE:
                        break
                if concyclic or time.time() > _DEADLINE:
                    break
            if concyclic or time.time() > _DEADLINE:
                continue
            # Found a valid 6-point set!
            new_pts = base + [(x, y)]
            new_int = [(px, 1, py, 1) for px, py in new_pts]
            # Final validation
            if _validate(new_int):
                return new_int
    return None

def construct():
    # Start with the known 5-point set
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    base_int = [(x, 1, y, 1) for x, y in base]
    
    # Try to find a 6-point set
    result = _brute_force_search()
    if result is not None:
        points = result
    else:
        points = base_int
    
    return {'D': 1, 'points': [[[x_num, x_den], [y_num, y_den]] for x_num, x_den, y_num, y_den in points]}
# EVOLVE-BLOCK-END
