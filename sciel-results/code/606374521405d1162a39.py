"""Rewritten program with stronger search for equal-area triangle multiplicity."""
import random
import time
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin

    def elapsed():
        return time.time() - start_time

    def time_ok(margin=0):
        return elapsed() < deadline - margin

    def compute_score(points):
        """Compute largest multiplicity of a single positive doubled area."""
        n = len(points)
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    det = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    if det > 0:
                        areas.append(det)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())

    def compute_score_fast(points):
        """Faster score computation using numpy for determinant calculation."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        # Compute all triple determinants
        # det = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # We'll compute this in batches to avoid memory issues
        areas = []
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    det = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    if det > 0:
                        areas.append(det)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())

    def inbounds(points, limit=30):
        for x, y in points:
            if x < -limit or x > limit or y < -limit or y > limit:
                return False
        return True

    def local_improve(points, max_iters=2000, timeout=45):
        """Local improvement via point moves with better acceptance criteria."""
        start_time_li = time.time()
        best_points = list(points)
        best_score = compute_score(best_points)
        no_improve_count = 0
        max_no_improve = 200

        for iteration in range(max_iters):
            if time.time() - start_time_li > timeout:
                break
            if not time_ok(5):
                break

            idx = random.randint(0, len(best_points) - 1)
            old_point = best_points[idx]
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = (new_x, new_y)

            if new_point in best_points:
                continue

            best_points[idx] = new_point
            new_score = compute_score(best_points)

            if new_score > best_score:
                best_score = new_score
                no_improve_count = 0
            else:
                best_points[idx] = old_point
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        return best_points, best_score

    def local_improve_targeted(points, target_area, max_iters=1500, timeout=40):
        """Local improvement that tries to increase count of target_area."""
        start_time_li = time.time()
        best_points = list(points)
        best_score = compute_score(best_points)
        best_target_count = sum(1 for a in _all_areas(best_points) if a == target_area)
        no_improve_count = 0

        for iteration in range(max_iters):
            if time.time() - start_time_li > timeout:
                break
            if not time_ok(5):
                break

            idx = random.randint(0, len(best_points) - 1)
            old_point = best_points[idx]
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = (new_x, new_y)

            if new_point in best_points:
                continue

            best_points[idx] = new_point
            new_score = compute_score(best_points)
            new_target_count = sum(1 for a in _all_areas(best_points) if a == target_area)

            if new_score > best_score or (new_score == best_score and new_target_count > best_target_count):
                best_score = new_score
                best_target_count = new_target_count
                no_improve_count = 0
            else:
                best_points[idx] = old_point
                no_improve_count += 1
                if no_improve_count > 150:
                    break

        return best_points, best_score

    def _all_areas(points):
        n = len(points)
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    det = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    if det > 0:
                        areas.append(det)
        return areas

    def generate_lattice_pattern(a, b, c, d, m, n, ox=0, oy=0):
        """Generate points on a lattice: p = ox + i*a + j*c, oy + i*b + j*d for i in [0,m), j in [0,n)"""
        points = []
        for i in range(m):
            for j in range(n):
                x = ox + i * a + j * c
                y = oy + i * b + j * d
                if -30 <= x <= 30 and -30 <= y <= 30:
                    if (x, y) not in points:
                        points.append((x, y))
        return points

    def generate_sheared_grid(w, h, shear, ox=0, oy=0):
        """Generate a sheared grid."""
        points = []
        for i in range(w):
            for j in range(h):
                x = ox + i + shear * j
                y = oy + j
                if -30 <= x <= 30 and -30 <= y <= 30:
                    if (x, y) not in points:
                        points.append((x, y))
        return points

    def generate_cross_pattern(center, arm_len, offset=1):
        """Generate a cross/plus pattern."""
        cx, cy = center
        points = [(cx, cy)]
        for d in range(1, arm_len + 1):
            for ox, oy in [(d, 0), (-d, 0), (0, d), (0, -d)]:
                p = (cx + ox, cy + oy)
                if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in points:
                    points.append(p)
        return points

    # Generate diverse initial candidates
    best_points = None
    best_score = 0

    # 1. Standard grids
    grid_configs = [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30)]
    for w, h in grid_configs:
        if not time_ok(5):
            break
        for _ in range(2):
            if not time_ok(5):
                break
            ox = random.randint(-30, 30 - w + 1)
            oy = random.randint(-30, 30 - h + 1)
            points = [(ox + x, oy + y) for x in range(w) for y in range(h)]
            if len(points) == 30 and inbounds(points):
                score = compute_score(points)
                if score > best_score:
                    best_score = score
                    best_points = list(points)

    # 2. Sheared grids
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10)]:
        if not time_ok(5):
            break
        for shear in range(-3, 4):
            if not time_ok(5):
                break
            ox = random.randint(-30, 30)
            oy = random.randint(-30, 30)
            points = generate_sheared_grid(w, h, shear, ox, oy)
            if len(points) == 30 and inbounds(points):
                score = compute_score(points)
                if score > best_score:
                    best_score = score
                    best_points = list(points)

    # 3. Lattice patterns with various basis vectors
    for _ in range(20):
        if not time_ok(10):
            break
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
        c = random.randint(-5, 5)
        d = random.randint(-5, 5)
        if a == 0 and b == 0 or c == 0 and d == 0:
            continue
        ox = random.randint(-20, 20)
        oy = random.randint(-20, 20)
        for m in range(3, 11):
            if not time_ok(10):
                break
            for n in range(3, 11):
                if m * n != 30:
                    continue
                points = generate_lattice_pattern(a, b, c, d, m, n, ox, oy)
                if len(points) == 30 and inbounds(points):
                    score = compute_score(points)
                    if score > best_score:
                        best_score = score
                        best_points = list(points)

    # 4. Random starts
    for _ in range(10):
        if not time_ok(10):
            break
        points = []
        used = set()
        while len(points) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                points.append((x, y))
                used.add((x, y))
        score = compute_score(points)
        if score > best_score:
            best_score = score
            best_points = list(points)

    # Local improvement from best candidate
    if best_points is not None:
        if time_ok(15):
            improved_points, improved_score = local_improve(best_points, max_iters=1500, timeout=30)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points

    # Try targeted improvement based on the most common area
    if best_points is not None and time_ok(20):
        areas = _all_areas(best_points)
        if areas:
            counts = Counter(areas)
            target_area = counts.most_common(1)[0][0]
            improved_points, improved_score = local_improve_targeted(best_points, target_area, max_iters=1000, timeout=25)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points

    # More random restarts with local improvement
    for _ in range(5):
        if not time_ok(15):
            break
        points = []
        used = set()
        while len(points) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                points.append((x, y))
                used.add((x, y))
        score = compute_score(points)
        if score > best_score:
            best_score = score
            best_points = list(points)
        if time_ok(15):
            improved_points, improved_score = local_improve(points, max_iters=800, timeout=15)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points

    # Final local improvement
    if best_points is not None and time_ok(10):
        improved_points, improved_score = local_improve(best_points, max_iters=1000, timeout=20)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points

    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]

    return best_points
