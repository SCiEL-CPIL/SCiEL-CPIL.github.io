"""Program using MODEL-AND-OPTIMIZE: local search + geometric verification for integral point sets.

Model:
- Parameterize as (D, list of points with rational x and y*sqrt(D)).
- Objective: maximize number of distinct points n (3..12) such that:
  * all pairwise squared distances are rational (equivalently integral after scaling for a fixed D)
  * no three points are collinear (exact determinant 0 check in affine 2-space)
  * no four points are concyclic (determinant-based cyclic condition using squared distances)
  * |numerators|, |denominators| <= 1e40, D squarefree <= 1e12
- For D=1, reduce to integer lattice with integer squared distances.

Optimization:
- Start from a known 5-point integral set.
- Use Pythagorean triple-based candidate generation for efficient point extension.
- Perform randomized hill-climbing to explore different configurations.
- Always return a valid configuration of maximum found n.

Deadline:
- Use time.perf_counter(); stop loops before ~110s wall-clock, then finalize.
"""
import time
from math import isqrt
from itertools import combinations
import random

TIME_LIMIT = 150.0
SAFETY = 30.0
DEADLINE = TIME_LIMIT - SAFETY

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _gcd(a, b):
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a

def _sqdist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _collinear3(A, B, C):
    # (B.x - A.x)*(C.y - A.y) - (B.y - A.y)*(C.x - A.x)
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def _conic_coeff(A, B, C, D):
    # determinant
    # | x1 y1 x1^2+y1^2 1 |
    # | x2 y2 x2^2+y2^2 1 |
    # | x3 y3 x3^2+y3^2 1 |
    # | x4 y4 x4^2+y4^2 1 |
    (x1, y1) = A
    (x2, y2) = B
    (x3, y3) = C
    (x4, y4) = D
    s1 = x1*x1 + y1*y1
    s2 = x2*x2 + y2*y2
    s3 = x3*x3 + y3*y3
    s4 = x4*x4 + y4*y4
    # Expand determinant with column 4 elimination (use row1 as pivot)
    # D = x1*y2*(s3 - s4) - x1*y3*(s2 - s4) + x1*y4*(s2 - s3)
    #   - y1*x2*(s3 - s4) + y1*x3*(s2 - s4) - y1*x4*(s2 - s3)
    #   + (s2*y3*x4 - s2*y4*x3) - (s3*y2*x4 - s3*y4*x2) + (s4*y2*x3 - s4*y3*x2)
    d = (
        x1 * y2 * (s3 - s4) - x1 * y3 * (s2 - s4) + x1 * y4 * (s2 - s3)
        - y1 * x2 * (s3 - s4) + y1 * x3 * (s2 - s4) - y1 * x4 * (s2 - s3)
        + s2 * (y3 * x4 - y4 * x3)
        - s3 * (y2 * x4 - y4 * x2)
        + s4 * (y2 * x3 - y3 * x2)
    )
    return d

def _valid_set(pts):
    """Validate that a set of integer points satisfies all constraints."""
    # Check distinct
    seen = set(pts)
    if len(seen) != len(pts):
        return False
    # Check all squared distances are perfect squares (D=1)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _is_square(_sqdist(pts[i], pts[j])):
                return False
    # No 3 collinear
    for a, b, c in combinations(pts, 3):
        if _collinear3(a, b, c) == 0:
            return False
    # No 4 concyclic
    for a, b, c, d in combinations(pts, 4):
        if _conic_coeff(a, b, c, d) == 0:
            return False
    return True

def _generate_pythagorean_triples(max_val):
    """Generate all Pythagorean triples (a, b, c) with a^2 + b^2 = c^2, a,b,c <= max_val."""
    triples = set()
    for a in range(1, max_val + 1):
        for b in range(a, max_val + 1):
            c_sq = a * a + b * b
            c = isqrt(c_sq)
            if c * c == c_sq and c <= max_val:
                triples.add((a, b, c))
                triples.add((b, a, c))
    return triples

def _candidates_from_pythagorean(center, max_dist):
    """Generate candidate points at integer distance from center using Pythagorean triples."""
    cx, cy = center
    candidates = set()
    # Use precomputed triples up to max_dist
    max_leg = max_dist
    for a in range(1, max_leg + 1):
        for b in range(a, max_leg + 1):
            c_sq = a * a + b * b
            c = isqrt(c_sq)
            if c * c == c_sq and c <= max_dist:
                # Add all rotations and reflections
                candidates.add((cx + a, cy + b))
                candidates.add((cx + b, cy + a))
                candidates.add((cx - a, cy + b))
                candidates.add((cx - b, cy + a))
                candidates.add((cx + a, cy - b))
                candidates.add((cx + b, cy - a))
                candidates.add((cx - a, cy - b))
                candidates.add((cx - b, cy - a))
    return candidates

def _format_points(pts):
    # D=1 so y is rational y with den=1
    return {"D": 1, "points": [[[x, 1], [y, 1]] for (x, y) in pts]}

def _try_extend(current, deadline):
    """Try to extend the current set by adding one valid point."""
    n = len(current)
    if n < 2:
        return current
    
    # Bound the search: points should be within a reasonable range
    xs = [p[0] for p in current]
    ys = [p[1] for p in current]
    xmin, xmax = min(xs) - 300, max(xs) + 300
    ymin, ymax = min(ys) - 300, max(ys) + 300
    
    existing = set(current)
    
    # Use Pythagorean triple-based candidate generation for efficiency
    # Generate candidates at integer distance from each existing point
    candidate_set = set()
    for p in current:
        # Generate candidates within bounding box
        cx, cy = p
        dx_range = max(abs(cx - xmin), abs(cx - xmax))
        dy_range = max(abs(cy - ymin), abs(cy - ymax))
        max_dist = isqrt(dx_range * dx_range + dy_range * dy_range) + 10
        
        # Generate using Pythagorean triples
        for a in range(1, min(max_dist, 200) + 1):
            for b in range(a, min(max_dist, 200) + 1):
                c_sq = a * a + b * b
                c = isqrt(c_sq)
                if c * c == c_sq and c <= max_dist:
                    # Add all rotations and reflections within bounds
                    for dx, dy in [(a, b), (b, a), (-a, b), (-b, a), 
                                   (a, -b), (b, -a), (-a, -b), (-b, -a)]:
                        nx, ny = cx + dx, cy + dy
                        if xmin <= nx <= xmax and ymin <= ny <= ymax:
                            if (nx, ny) not in existing:
                                candidate_set.add((nx, ny))
    
    # Shuffle candidates for randomization
    candidates = list(candidate_set)
    random.shuffle(candidates)
    
    for (x, y) in candidates:
        if time.perf_counter() > deadline:
            break
        
        # Check distance to all existing points is perfect square
        ok = True
        for p in current:
            dx = x - p[0]
            dy = y - p[1]
            if not _is_square(dx * dx + dy * dy):
                ok = False
                break
        if not ok:
            continue
        
        # Quick collinearity test vs pairs
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
        
        # Quick concyclicity with 3 of the existing (any 3)
        for a, b, c in combinations(range(n), 3):
            if _conic_coeff(current[a], current[b], current[c], (x, y)) == 0:
                bad = True
                break
        if bad:
            continue
        
        # Full verification
        new_set = current + [(x, y)]
        if _valid_set(new_set):
            return new_set
    
    return current

def _hill_climb(current, deadline, max_iters=100):
    """Randomized hill-climbing to try to find a better configuration."""
    best = current[:]
    best_size = len(best)
    
    for _ in range(max_iters):
        if time.perf_counter() > deadline:
            break
        
        # Try to extend
        extended = _try_extend(best, deadline)
        if len(extended) > best_size:
            best = extended
            best_size = len(best)
        
        # Occasionally try a different starting point
        if len(best) < 8 and random.random() < 0.3:
            # Generate a random 4-point configuration
            try:
                rand_pts = _generate_random_config(4, deadline)
                if rand_pts and _valid_set(rand_pts):
                    # Try to extend from this random config
                    extended_rand = rand_pts[:]
                    for _ in range(10):
                        if time.perf_counter() > deadline:
                            break
                        new_ext = _try_extend(extended_rand, deadline)
                        if len(new_ext) <= len(extended_rand):
                            break
                        extended_rand = new_ext
                    if len(extended_rand) > best_size:
                        best = extended_rand
                        best_size = len(best)
            except:
                pass
    
    return best

def _generate_random_config(n, deadline):
    """Generate a random n-point configuration with integer distances."""
    if time.perf_counter() > deadline:
        return None
    
    pts = [(0, 0)]
    
    for _ in range(n - 1):
        if time.perf_counter() > deadline:
            return None
        
        # Generate random point at integer distance from origin
        max_coord = 100
        found = False
        for _ in range(1000):
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
            if (x, y) in pts:
                continue
            # Check distance to all existing points
            ok = True
            for p in pts:
                dx = x - p[0]
                dy = y - p[1]
                if not _is_square(dx * dx + dy * dy):
                    ok = False
                    break
            if not ok:
                continue
            # Check collinearity
            bad = False
            for a, b in combinations(pts, 2):
                if _collinear3(a, b, (x, y)) == 0:
                    bad = True
                    break
            if bad:
                continue
            pts.append((x, y))
            found = True
            break
        
        if not found:
            return None
    
    return pts if len(pts) == n else None

def construct():
    """Main entry point: construct a valid integral point set."""
    # Seed 5-point set (known valid configuration)
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    # Validate base (should be valid)
    if not _valid_set(base):
        # Fallback to smaller valid known set: use the first 3 points
        base = base[:3]

    t0 = time.perf_counter()
    deadline = t0 + DEADLINE

    best = base[:]

    # Phase 1: Iterative extension from the base
    while len(best) < 12 and (time.perf_counter() < deadline):
        extended = _try_extend(best, deadline)
        if len(extended) <= len(best):
            break
        best = extended

    # Phase 2: Hill-climbing to explore alternative configurations
    if time.perf_counter() < deadline:
        improved = _hill_climb(best, deadline, max_iters=50)
        if len(improved) > len(best):
            best = improved

    # Final safeguard
    if not _valid_set(best):
        # Ensure we return at least a minimal valid set
        best = base[:3]

    return _format_points(best)
