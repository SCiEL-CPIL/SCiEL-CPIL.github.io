import random
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def is_rational_dist(p1, p2):
        return is_square(dist_sq(p1, p2))

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def are_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        ax, ay = x2 - x1, y2 - y1
        bx, by = x3 - x1, y3 - y1
        cx, cy = x4 - x1, y4 - y1
        a2 = ax*ax + ay*ay
        b2 = bx*bx + by*by
        c2 = cx*cx + cy*cy
        det = ax*(by*c2 - b2*cy) - ay*(bx*c2 - b2*cx) + a2*(bx*cy - by*cx)
        return det == 0

    # Precompute all valid displacement vectors with rational distance (D=1)
    max_coord = 400
    vectors = []
    for dx in range(-max_coord, max_coord + 1):
        for dy in range(-max_coord, max_coord + 1):
            if dx == 0 and dy == 0:
                continue
            if is_square(dx * dx + dy * dy):
                vectors.append((dx, dy))

    # Sort vectors: prefer smaller displacements first for denser configurations
    vectors.sort(key=lambda v: (v[0]*v[0] + v[1]*v[1], v[0], v[1]))

    # Deduplicate: keep unique vectors
    seen = set()
    unique_vectors = []
    for v in vectors:
        if v not in seen:
            seen.add(v)
            unique_vectors.append(v)
    vectors = unique_vectors

    def validate_new_point(points, p_new):
        """Check if p_new can be added to points: rational distances, no 3 collinear, no 4 concyclic"""
        n = len(points)
        # Check distances
        for p in points:
            if not is_rational_dist(p, p_new):
                return False
        # Check collinearity
        for i in range(n):
            for j in range(i + 1, n):
                if are_collinear(points[i], points[j], p_new):
                    return False
        # Check concyclicity
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_concyclic(points[i], points[j], points[k], p_new):
                        return False
        return True

    def find_next_point(points, rng, max_candidates=20000):
        """Find a point to add to points using smart candidate generation"""
        n = len(points)
        point_set = set(points)
        
        # Build candidate set: for each existing point, find all displacements
        # that give rational distance to it. This is a strong pre-filter.
        candidates = []
        for p_base in points:
            for dx, dy in vectors:
                p_new = (p_base[0] + dx, p_base[1] + dy)
                if p_new in point_set:
                    continue
                # Quick check: rational distance to p_base is guaranteed by construction
                # (since (dx,dy) is a valid vector). Check distance to all other points.
                valid = True
                for p in points:
                    if p == p_base:
                        continue
                    if not is_rational_dist(p, p_new):
                        valid = False
                        break
                if not valid:
                    continue
                candidates.append(p_new)
        
        if not candidates:
            return None
        
        # Deduplicate candidates
        candidates = list(set(candidates))
        
        # Shuffle for randomization
        rng.shuffle(candidates)
        
        # Try candidates
        for p_new in candidates[:max_candidates]:
            if validate_new_point(points, p_new):
                return p_new
        return None

    def find_next_point_fast(points, rng, max_candidates=50000):
        """Faster version: generate candidates on the fly with early termination"""
        n = len(points)
        point_set = set(points)
        
        # For each base point, iterate through vectors and check
        candidates_found = 0
        tried = 0
        
        for p_base in points:
            for dx, dy in vectors:
                tried += 1
                if tried > max_candidates:
                    return None
                p_new = (p_base[0] + dx, p_base[1] + dy)
                if p_new in point_set:
                    continue
                # Check rational distance to all points
                valid_dist = True
                for p in points:
                    if not is_rational_dist(p, p_new):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue
                # Check collinearity
                valid_collinear = True
                for i in range(n):
                    for j in range(i + 1, n):
                        if are_collinear(points[i], points[j], p_new):
                            valid_collinear = False
                            break
                    if not valid_collinear:
                        break
                if not valid_collinear:
                    continue
                # Check concyclicity
                valid_concyclic = True
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            if are_concyclic(points[i], points[j], points[k], p_new):
                                valid_concyclic = False
                                break
                        if not valid_concyclic:
                            break
                    if not valid_concyclic:
                        break
                if not valid_concyclic:
                    continue
                return p_new
        return None

    def backtrack_search(start_points, target_size, rng, max_depth=7, time_budget=10.0):
        """Backtracking search to extend start_points to target_size"""
        best = start_points[:]
        
        def _search(current, depth):
            nonlocal best
            if time.time() > start_time + time_budget:
                return
            if len(current) > len(best):
                best = current[:]
            if len(current) >= target_size:
                return
            if depth >= max_depth:
                return
            
            # Find a point to add
            p_new = find_next_point_fast(current, rng, max_candidates=10000)
            if p_new is None:
                return
            
            current.append(p_new)
            _search(current, depth + 1)
            current.pop()
        
        _search(start_points[:], 0)
        return best

    def try_extend(points, rng, max_rounds=10, time_budget=8.0):
        """Try to extend points by adding one at a time with backtracking"""
        current = points[:]
        for _ in range(max_rounds):
            if time.time() > start_time + time_budget:
                break
            if len(current) >= 12:
                break
            p_new = find_next_point_fast(current, rng, max_candidates=20000)
            if p_new is None:
                # Try with different randomization
                p_new = find_next_point(current, rng, max_candidates=5000)
                if p_new is None:
                    break
            current.append(p_new)
        return current

    # Known valid 5-point configuration
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_points = known_5[:]
    best_size = 5

    # Strategy 1: Extend known 5-point set with backtracking
    rng = random.Random(42)
    if time.time() < deadline - 5:
        extended = try_extend(known_5, rng, max_rounds=7, time_budget=15.0)
        if len(extended) > best_size:
            best_size = len(extended)
            best_points = extended[:]

    # Strategy 2: Try different 5-point configurations
    variants = [
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (-56, -33), (-16, -63), (0, -33), (16, -63)],
        [(0, 0), (56, -33), (16, -63), (0, -33), (-16, -63)],
        [(0, 0), (-56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (56, -33), (-16, -63), (0, -33), (16, -63)],
    ]
    
    for vi, variant in enumerate(variants):
        if time.time() > deadline - 5:
            break
        rng = random.Random(1000 + vi * 777)
        extended = try_extend(variant, rng, max_rounds=7, time_budget=10.0)
        if len(extended) > best_size:
            best_size = len(extended)
            best_points = extended[:]

    # Strategy 3: Random restarts - build from scratch
    while time.time() < deadline - 5:
        rng = random.Random()
        
        # Build a 3-point triangle
        p1 = (0, 0)
        pts = [p1]
        
        # Point 2: random valid displacement
        dx, dy = vectors[random.randint(0, min(len(vectors), 500) - 1)]
        p2 = (p1[0] + dx, p1[1] + dy)
        if p2 == p1:
            continue
        pts.append(p2)
        
        # Point 3: must be at rational distance from p1 and p2, not collinear
        found_p3 = False
        for _ in range(20000):
            base = random.choice(pts)
            dx, dy = vectors[random.randint(0, min(len(vectors), 1000) - 1)]
            p3 = (base[0] + dx, base[1] + dy)
            if p3 in pts:
                continue
            if not is_rational_dist(pts[0], p3):
                continue
            if not is_rational_dist(pts[1], p3):
                continue
            if are_collinear(pts[0], pts[1], p3):
                continue
            pts.append(p3)
            found_p3 = True
            break
        
        if not found_p3:
            continue
        
        # Extend to 4, 5, ...
        current = pts[:]
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            if len(current) >= 12:
                break
            p_new = find_next_point_fast(current, rng, max_candidates=15000)
            if p_new is None:
                p_new = find_next_point(current, rng, max_candidates=3000)
                if p_new is None:
                    break
            current.append(p_new)
        
        if len(current) > best_size:
            best_size = len(current)
            best_points = current[:]
            if best_size >= 8:
                break

    # Strategy 4: Backtracking from known 5-point set with more depth
    if time.time() < deadline - 3 and best_size < 8:
        rng = random.Random(9999)
        result = backtrack_search(known_5, 12, rng, max_depth=7, time_budget=15.0)
        if len(result) > best_size:
            best_size = len(result)
            best_points = result[:]

    # Final validation of best_points
    final_points = best_points
    # Verify all constraints
    n = len(final_points)
    for i in range(n):
        for j in range(i + 1, n):
            if not is_rational_dist(final_points[i], final_points[j]):
                # Fallback to known 5
                final_points = known_5
                break
    n = len(final_points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(final_points[i], final_points[j], final_points[k]):
                    final_points = known_5
                    break
    n = len(final_points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if are_concyclic(final_points[i], final_points[j], final_points[k], final_points[l]):
                        final_points = known_5
                        break

    points = final_points
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
