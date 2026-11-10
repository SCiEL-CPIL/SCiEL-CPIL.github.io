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

def _is_square_fast(n):
    """Fast integer square check using math.isqrt."""
    if n < 0:
        return False
    r = int(n ** 0.5)
    # Adjust for potential floating point issues
    for c in (r - 1, r, r + 1):
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

def _precompute_base_constraints(base):
    """Precompute squared distances, collinearity lines, and concyclicity circles for base set."""
    n = len(base)
    # Squared distances between base points (for reference)
    base_dists_sq = {}
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (base[i][0] - base[j][0]) ** 2 + (base[i][1] - base[j][1]) ** 2
            base_dists_sq[(i, j)] = d2
    
    # Collinearity: for each pair (i,j), store the line equation ax + by + c = 0
    # where a = yj - yi, b = xi - xj, c = xj*yi - xi*yj
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = base[i]
            xj, yj = base[j]
            a = yj - yi
            b = xi - xj
            c = xj * yi - xi * yj
            lines[(i, j)] = (a, b, c)
    
    # Concyclicity: for each triple (i,j,k), compute circle equation
    # Circle: x^2 + y^2 + Dx + Ey + F = 0
    # Solve for D, E, F using three points
    circles = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = base[i]
                x2, y2 = base[j]
                x3, y3 = base[k]
                # System:
                # 2*x1*D + 2*y1*E = x2^2 + y2^2 - x1^2 - y1^2 - 2*x1*x2 - 2*y1*y2 ... 
                # Actually: D = (x^2+y^2) coefficient
                # x1^2 + y1^2 + D*x1 + E*y1 + F = 0
                # x2^2 + y2^2 + D*x2 + E*y2 + F = 0
                # x3^2 + y3^2 + D*x3 + E*y3 + F = 0
                # Subtract eq1 from eq2: (x2^2-x1^2) + (y2^2-y1^2) + D*(x2-x1) + E*(y2-y1) = 0
                # Subtract eq1 from eq3: (x3^2-x1^2) + (y3^2-y1^2) + D*(x3-x1) + E*(y3-y1) = 0
                a1 = x2 - x1
                b1 = y2 - y1
                c1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
                a2 = x3 - x1
                b2 = y3 - y1
                c2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
                det = a1 * b2 - a2 * b1
                if det == 0:
                    # Collinear - shouldn't happen for valid base
                    circles[(i, j, k)] = None
                else:
                    D = (c1 * b2 - c2 * b1) / det
                    E = (a1 * c2 - a2 * c1) / det
                    F = -(x1*x1 + y1*y1 + D * x1 + E * y1)
                    circles[(i, j, k)] = (D, E, F)
    
    return base_dists_sq, lines, circles

def _search_one_base(base, max_coord=300):
    """Search for a 6th point given a valid 5-point base."""
    base_set = set(base)
    n = len(base)
    base_dists_sq, lines, circles = _precompute_base_constraints(base)
    
    # Precompute: for each candidate (x,y), we need:
    # 1. (x-bx)^2 + (y-by)^2 is a perfect square for all base points
    # 2. Not on any line through two base points
    # 3. Not on any circle through three base points
    
    found = None
    for x in range(-max_coord, max_coord + 1):
        if time.time() > _DEADLINE:
            break
        for y in range(-max_coord, max_coord + 1):
            if time.time() > _DEADLINE:
                break
            # Check distinctness
            if (x, y) in base_set:
                continue
            
            # Check distance to all base points
            ok = True
            for bx, by in base:
                d2 = (x - bx) ** 2 + (y - by) ** 2
                if not _is_square_fast(d2):
                    ok = False
                    break
            if not ok:
                continue
            
            # Check collinearity with pairs of base points
            collinear = False
            for (i, j), (a, b, c) in lines.items():
                if a * x + b * y + c == 0:
                    collinear = True
                    break
            if collinear:
                continue
            
            # Check concyclicity with triples of base points
            concyclic = False
            for (i, j, k), circle in circles.items():
                if circle is None:
                    continue
                D, E, F = circle
                # Check if x^2 + y^2 + D*x + E*y + F == 0
                # D, E, F may be fractions, so multiply through
                # Use exact arithmetic: check (x^2 + y^2)*den + D_num*x*den + E_num*y*den + F_num*den == 0
                # Since D, E, F are computed as fractions from integers, let's use Fraction
                from fractions import Fraction as Fr
                Df = Fr(D) if not isinstance(D, Fr) else D
                Ef = Fr(E) if not isinstance(E, Fr) else E
                Ff = Fr(F) if not isinstance(F, Fr) else F
                val = Fr(x*x + y*y) + Df * x + Ef * y + Ff
                if val == 0:
                    concyclic = True
                    break
            if concyclic:
                continue
            
            # Found a candidate! Verify with full validation
            new_pts = base + [(x, y)]
            new_int = [(px, 1, py, 1) for px, py in new_pts]
            if _validate(new_int):
                return new_int
    return None

def _brute_force_search(max_coord=300, target_size=6):
    """Search for target_size points by trying multiple base configurations."""
    # Known 5-point integral sets
    base_configs = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (5, 0), (0, 5), (5, 5), (3, 4)],
        [(0, 0), (3, 0), (0, 4), (3, 4), (1, 2)],
        [(0, 0), (4, 0), (0, 3), (4, 3), (1, 1)],
    ]
    
    # First verify each base config is valid
    valid_bases = []
    for base in base_configs:
        base_int = [(x, 1, y, 1) for x, y in base]
        if _validate(base_int):
            valid_bases.append(base)
    
    if not valid_bases:
        # Fallback to original
        valid_bases = [[(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]]
    
    # Try each valid base
    for base in valid_bases:
        if time.time() > _DEADLINE:
            break
        result = _search_one_base(base, max_coord)
        if result is not None:
            return result
    
    return None

def construct():
    # Start with the known 5-point set
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    base_int = [(x, 1, y, 1) for x, y in base]
    
    # Try to find a 6-point set
    result = _brute_force_search(max_coord=300)
    if result is not None:
        points = result
    else:
        points = base_int
    
    return {'D': 1, 'points': [[[x_num, x_den], [y_num, y_den]] for x_num, x_den, y_num, y_den in points]}
# EVOLVE-BLOCK-END
