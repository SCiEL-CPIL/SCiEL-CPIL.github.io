import time
from math import isqrt
from itertools import combinations

TIME_LIMIT = 150.0
SAFETY = 20.0
DEADLINE = TIME_LIMIT - SAFETY

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sqdist(p, q, D=1):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _collinear3(A, B, C):
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def _conic_coeff(A, B, C, D, D_val=1):
    (x1, y1) = A
    (x2, y2) = B
    (x3, y3) = C
    (x4, y4) = D
    # s = x^2 + D*y^2
    s1 = x1*x1 + D_val*y1*y1
    s2 = x2*x2 + D_val*y2*y2
    s3 = x3*x3 + D_val*y3*y3
    s4 = x4*x4 + D_val*y4*y4
    # Determinant of:
    # | x1 y1 s1 1 |
    # | x2 y2 s2 1 |
    # | x3 y3 s3 1 |
    # | x4 y4 s4 1 |
    # Using expansion along last column:
    # 1 * (det of 3x3) - 1 * ...
    # It's easier to use the formula for concyclicity in general metric?
    # No, concyclicity is a Euclidean property.
    # Points (x, y*sqrt(D)) are in the plane.
    # The circle equation is x^2 + (y*sqrt(D))^2 + Ax + By + C = 0 ?
    # No, a circle in the plane is x^2 + y^2 + ... = 0.
    # Here the coordinates are (X, Y) = (x, y*sqrt(D)).
    # So the circle equation is X^2 + Y^2 + aX + bY + c = 0.
    # X^2 + Y^2 = x^2 + D y^2.
    # So the condition is that the points (x_i, y_i) lie on a circle in the metric defined by D?
    # No, the points are embedded in Euclidean plane as (x, y*sqrt(D)).
    # So concyclicity means there exist a, b, c such that for all points:
    # x_i^2 + D y_i^2 + a x_i + b (y_i sqrt(D)) + c = 0
    # This is not a standard circle in (x,y) coordinates unless D=1.
    # Wait, the problem says "no four concyclic".
    # This refers to the geometric points in the plane.
    # The points are $P_i = (x_i, y_i \sqrt{D})$.
    # Four points $P_1, P_2, P_3, P_4$ are concyclic if there is a circle in the standard Euclidean plane containing them.
    # The equation of a circle is $X^2 + Y^2 + aX + bY + c = 0$.
    # Substituting $X=x, Y=y\sqrt{D}$:
    # $x^2 + D y^2 + a x + b y \sqrt{D} + c = 0$.
    # For this to hold for rational $x, y$, we must have $b \sqrt{D}$ be rational?
    # If $D$ is not a square, $\sqrt{D}$ is irrational.
    # Since $x, y$ are rational, $x^2 + D y^2 + a x + c$ is rational (if a,c rational).
    # For the sum to be 0, the irrational part $b y \sqrt{D}$ must be 0.
    # This implies either $b=0$ or $y=0$.
    # If $y$ is not always 0, then $b$ must be 0.
    # So the circle equation becomes $x^2 + D y^2 + a x + c = 0$.
    # This is an ellipse (or hyperbola) in the (x,y) plane, not a circle.
    # BUT, the problem says "concyclic". This usually means "lying on a circle in the Euclidean plane".
    # The points are in the Euclidean plane with coordinates $(x, y\sqrt{D})$.
    # So yes, they must lie on a circle $X^2 + Y^2 + aX + bY + c = 0$.
    # The determinant check for concyclicity of points $(X_i, Y_i)$ is:
    # | X1^2+Y1^2  X1  Y1  1 |
    # | X2^2+Y2^2  X2  Y2  1 |
    # | X3^2+Y3^2  X3  Y3  1 |
    # | X4^2+Y4^2  X4  Y4  1 | = 0
    # Here $X_i = x_i, Y_i = y_i \sqrt{D}$.
    # $X_i^2 + Y_i^2 = x_i^2 + D y_i^2$.
    # So the determinant is:
    # | x1^2 + D y1^2  x1  y1 sqrt(D)  1 |
    # ...
    # This involves $\sqrt{D}$.
    # If $D$ is not a square, we can factor out $\sqrt{D}$ from the 3rd column?
    # No, the entries are $y_i \sqrt{D}$.
    # Let's compute the determinant symbolically.
    # Let $M$ be the matrix.
    # If we multiply the 3rd column by $1/\sqrt{D}$, the determinant is scaled by $1/\sqrt{D}$.
    # The new 3rd column is $y_i$.
    # So the condition is:
    # | x1^2 + D y1^2  x1  y1  1 |
    # | x2^2 + D y2^2  x2  y2  1 |
    # | x3^2 + D y3^2  x3  y3  1 |
    # | x4^2 + D y4^2  x4  y4  1 | = 0
    # This is a rational determinant!
    # So we can check concyclicity using this determinant with rational arithmetic.

def _conic_det(pts, D):
    # pts: list of 4 (x,y) tuples
    # D: squarefree integer
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
    s1 = x1*x1 + D*y1*y1
    s2 = x2*x2 + D*y2*y2
    s3 = x3*x3 + D*y3*y3
    s4 = x4*x4 + D*y4*y4
    # Determinant of:
    # | s1 x1 y1 1 |
    # | s2 x2 y2 1 |
    # | s3 x3 y3 1 |
    # | s4 x4 y4 1 |
    # Expand along last column:
    # 1 * det(s2,x2,y2; s3,x3,y3; s4,x4,y4) 
    # - 1 * det(s1,x1,y1; s3,x3,y3; s4,x4,y4)
    # + 1 * det(s1,x1,y1; s2,x2,y2; s4,x4,y4)
    # - 1 * det(s1,x1,y1; s2,x2,y2; s3,x3,y3)
    
    def det3(a,b,c, d,e,f, g,h,i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
        
    d1 = det3(s2,x2,y2, s3,x3,y3, s4,x4,y4)
    d2 = det3(s1,x1,y1, s3,x3,y3, s4,x4,y4)
    d3 = det3(s1,x1,y1, s2,x2,y2, s4,x4,y4)
    d4 = det3(s1,x1,y1, s2,x2,y2, s3,x3,y3)
    
    return d1 - d2 + d3 - d4

def _valid_set(pts, D):
    # Check distinct
    seen = set(pts)
    if len(seen) != len(pts):
        return False
    # Check all squared distances are perfect squares
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _is_square(_sqdist(pts[i], pts[j], D)):
                return False
    # No 3 collinear
    for a, b, c in combinations(pts, 3):
        if _collinear3(a, b, c) == 0:
            return False
    # No 4 concyclic
    for a, b, c, d in combinations(pts, 4):
        if _conic_det([a, b, c, d], D) == 0:
            return False
    return True

def _format_points(pts, D):
    return {"D": D, "points": [[[x, 1], [y, 1]] for (x, y) in pts]}

def _find_next_point(current, D, deadline, max_radius=200):
    n = len(current)
    x0, y0 = current[0]
    existing = set(current)
    
    # Precompute constraints for faster rejection
    # We will iterate over dx, dy such that (x0+dx, y0+dy) is a candidate
    # Condition: dist^2 to P0 is a square.
    # dist^2 = dx^2 + D*dy^2 = k^2
    
    # Iterate over k
    max_k = int(max_radius * (1 + D**0.5)) + 10
    
    for k in range(1, max_k + 1):
        if time.perf_counter() > deadline:
            break
        k2 = k * k
        # Find integer solutions to dx^2 + D*dy^2 = k^2
        # dy^2 = (k^2 - dx^2) / D
        # So k^2 - dx^2 must be divisible by D and the quotient must be a square.
        for dx in range(-k, k + 1):
            rem = k2 - dx * dx
            if rem < 0:
                continue
            if rem % D != 0:
                continue
            dy2 = rem // D
            dy = isqrt(dy2)
            if dy * dy != dy2:
                continue
            
            # Candidates: (dx, dy), (dx, -dy)
            for dy_val in [dy, -dy] if dy > 0 else [dy]:
                x = x0 + dx
                y = y0 + dy_val
                if (x, y) in existing:
                    continue
                
                # Check distances to other points
                ok = True
                for i in range(1, n):
                    if not _is_square(_sqdist((x, y), current[i], D)):
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check collinearity
                bad = False
                for a in range(n):
                    for b in range(a + 1, n):
                        if _collinear3(current[a], current[b], (x, y)) == 0:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue
                
                # Check concyclicity
                for a, b, c in combinations(range(n), 3):
                    if _conic_det([current[a], current[b], current[c], (x, y)], D) == 0:
                        bad = True
                        break
                if bad:
                    continue
                
                return (x, y)
    return None

def construct():
    t0 = time.perf_counter()
    deadline = t0 + DEADLINE
    
    # Try D=1 first
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _valid_set(base, 1):
        return _format_points(base[:3], 1)
        
    best_set = base[:]
    best_D = 1
    
    # Try to extend D=1 set
    while len(best_set) < 12 and time.perf_counter() < deadline:
        next_pt = _find_next_point(best_set, 1, deadline, max_radius=150)
        if next_pt is None:
            break
        best_set.append(next_pt)
        
    # If we didn't find many points with D=1, try other D
    # But D=1 is the most likely to yield points.
    # If we have 5 or 6 points, that's good.
    
    # Try to find a better set with D=2, 3, etc?
    # The search space is larger.
    # For now, stick to D=1 as it's the standard "integral point set".
    
    if not _valid_set(best_set, best_D):
        best_set = base[:3]
        
    return _format_points(best_set, best_D)
