import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 105.0  # Safety margin for 120s limit

    # --- PG(3,3) construction ---
    # Points: 16 non-zero vectors in F_3^4, up to scalar (1,2)
    # Lines: 21 lines, each with 4 points, every pair of points on exactly one line

    # Generate all 16 points as representatives of 1-dim subspaces
    pg33_points = []
    seen = set()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    if a == 0 and b == 0 and c == 0 and d == 0:
                        continue
                    # Representative: make first non-zero entry = 1
                    vec = [a, b, c, d]
                    for i in range(4):
                        if vec[i] != 0:
                            if vec[i] == 2:
                                vec = [(v * 2) % 3 for v in vec]
                            break
                    key = tuple(vec)
                    if key not in seen:
                        seen.add(key)
                        pg33_points.append(key)

    # Generate all 21 lines: each line is a set of 4 points
    # A line in PG(3,3) is a 2-dim subspace minus the 0 vector, which has 3^2 - 1 = 8 vectors
    # But each line has (3^2 - 1)/(3-1) = 4 points
    pg33_lines = []
    seen_lines = set()

    # For each pair of points, find the unique line containing them
    # The line through points p and q is the set of all points r such that p, q, r are collinear
    # In F_3, three points p, q, r are collinear iff r = a*p + b*q for some a,b in F_3, not both 0
    # Actually, the line through p and q contains p, q, p+q, p-q (i.e., p+2q) in F_3

    # Let's generate lines by taking all 2-dim subspaces
    # A 2-dim subspace is spanned by two linearly independent vectors
    # For each pair of basis vectors (u, v) with u < v in some ordering, the subspace is {a*u + b*v : a,b in F_3}
    # This gives 9 vectors, 8 non-zero, which form 4 lines' worth... no wait.

    # Simpler: for each pair of distinct points (p, q), the line through them contains:
    # p, q, (p+q) mod 3, (p-q) mod 3 = (p+2q) mod 3
    # But we need to make sure these are all distinct and are in our point set

    point_to_idx = {p: i for i, p in enumerate(pg33_points)}

    for i in range(16):
        for j in range(i + 1, 16):
            p = np.array(pg33_points[i], dtype=int)
            q = np.array(pg33_points[j], dtype=int)

            # The 4 points on the line through p and q in F_3^4:
            # p, q, p+q, p-q (mod 3)
            candidates = [
                p % 3,
                q % 3,
                (p + q) % 3,
                (p - q) % 3,
            ]

            # Normalize each to representative form
            line_pts = []
            for c in candidates:
                # Find representative
                for k in range(4):
                    if c[k] != 0:
                        if c[k] == 2:
                            c = (c * 2) % 3
                        break
                line_pts.append(tuple(c))

            # Sort for canonical form
            line_pts.sort()
            line_key = tuple(line_pts)

            if line_key not in seen_lines:
                seen_lines.add(line_key)
                # Convert to indices
                line_indices = [point_to_idx[pt] for pt in line_pts]
                pg33_lines.append(line_indices)

    # Verify: should have 21 lines
    # print(f"PG(3,3): {len(pg33_points)} points, {len(pg33_lines)} lines")

    # --- Fast evaluation ---
    def evaluate(points):
        """Evaluate a configuration of 16 points. Returns (score, valid)."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)

        # Generate all pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        p1 = pts[i_idx]
        p2 = pts[j_idx]

        dx = p2[:, 0] - p1[:, 0]
        dy = p2[:, 1] - p1[:, 1]

        # Check for duplicates
        if np.any((dx == 0) & (dy == 0)):
            return -10000, False

        # Compute gcd vectorized
        g = np.gcd(np.abs(dx), np.abs(dy))
        g[g == 0] = 1  # Safety

        dx_norm = dx // g
        dy_norm = dy // g

        # Normalize direction
        flip = (dx_norm < 0) | ((dx_norm == 0) & (dy_norm < 0))
        dx_norm = dx_norm.copy()
        dy_norm = dy_norm.copy()
        dx_norm[flip] = -dx_norm[flip]
        dy_norm[flip] = -dy_norm[flip]

        c = dy_norm * p1[:, 0] - dx_norm * p1[:, 1]

        # Create keys and count
        keys = np.stack([dx_norm, dy_norm, c], axis=1)
        # Use a dictionary for counting
        line_counts = {}
        for k in range(len(keys)):
            key = (int(keys[k, 0]), int(keys[k, 1]), int(keys[k, 2]))
            line_counts[key] = line_counts.get(key, 0) + 1

        score = 0
        valid = True
        for count in line_counts.values():
            if count == 6:  # 4 points
                score += 1
            elif count >= 10:  # 5+ points
                valid = False
                break

        if not valid:
            return -1000 + score, False
        return score, True

    def evaluate_fast_score(points):
        """Fast score-only evaluation for local search."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)

        i_idx, j_idx = np.triu_indices(n, k=1)
        p1 = pts[i_idx]
        p2 = pts[j_idx]

        dx = p2[:, 0] - p1[:, 0]
        dy = p2[:, 1] - p1[:, 1]

        if np.any((dx == 0) & (dy == 0)):
            return -10000

        g = np.gcd(np.abs(dx), np.abs(dy))
        g[g == 0] = 1

        dx_norm = dx // g
        dy_norm = dy // g

        flip = (dx_norm < 0) | ((dx_norm == 0) & (dy_norm < 0))
        dx_norm = dx_norm.copy()
        dy_norm = dy_norm.copy()
        dx_norm[flip] = -dx_norm[flip]
        dy_norm[flip] = -dy_norm[flip]

        c = dy_norm * p1[:, 0] - dx_norm * p1[:, 1]

        keys = np.stack([dx_norm, dy_norm, c], axis=1)
        line_counts = {}
        for k in range(len(keys)):
            key = (int(keys[k, 0]), int(keys[k, 1]), int(keys[k, 2]))
            line_counts[key] = line_counts.get(key, 0) + 1

        score = 0
        for count in line_counts.values():
            if count == 6:
                score += 1
            elif count >= 10:
                return -1000 + score

        return score

    # --- Phase 1: Find good affine mappings from PG(3,3) to integer lattice ---
    # Map each of the 16 PG(3,3) points to an integer point via:
    #   mapped[i] = [A * pg33_points[i] + b]
    # where A is a 2x4 integer matrix and b is a 2-vector
    # We'll search over small integer A and b

    best_points = None
    best_score = -1

    # Try a variety of affine mappings
    # The 16 points in F_3^4 can be thought of as coordinates
    # We map (a,b,c,d) -> (alpha1*a + alpha2*b + alpha3*c + alpha4*d + bx,
    #                       beta1*a + beta2*b + beta3*c + beta4*d + by)

    # To get distinct integer points, we need the mapping to be injective
    # Let's try small coefficient ranges

    coeff_range = range(-3, 4)
    bx_range = range(-50, 51)
    by_range = range(-50, 51)

    # We'll sample a subset of mappings to keep within time
    random.seed(42)

    candidates = []
    num_candidates = 50000

    for _ in range(num_candidates):
        # Random coefficients
        alpha = [random.choice(coeff_range) for _ in range(4)]
        beta = [random.choice(coeff_range) for _ in range(4)]
        bx = random.choice(bx_range)
        by = random.choice(by_range)

        # Map points
        mapped = []
        valid = True
        for pt in pg33_points:
            mx = alpha[0] * pt[0] + alpha[1] * pt[1] + alpha[2] * pt[2] + alpha[3] * pt[3] + bx
            my = beta[0] * pt[0] + beta[1] * pt[1] + beta[2] * pt[2] + beta[3] * pt[3] + by
            if abs(mx) > 1000 or abs(my) > 1000:
                valid = False
                break
            mapped.append([mx, my])

        if not valid:
            continue

        # Check distinctness
        if len(set(tuple(p) for p in mapped)) != 16:
            continue

        # Check no 5 collinear (quick check)
        score, valid2 = evaluate(mapped)
        if not valid2:
            continue

        if score > best_score:
            best_score = score
            best_points = [p[:] for p in mapped]

        if time.time() - start_time > time_limit * 0.4:
            break

    # If we didn't find anything good, fall back to 4x4 grid
    if best_points is None or best_score < 10:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = evaluate_fast_score(best_points)

    # --- Phase 2: Local search to improve ---
    current_points = [p[:] for p in best_points]
    current_score = best_score

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def random_move(points):
        idx = random.randint(0, 15)
        if random.random() < 0.5:
            new_x = points[idx][0] + random.randint(-5, 5)
            new_y = points[idx][1] + random.randint(-5, 5)
        else:
            new_x = points[idx][0] + random.randint(-20, 20)
            new_y = points[idx][1] + random.randint(-20, 20)

        if abs(new_x) > 1000 or abs(new_y) > 1000:
            return None

        new_point = [new_x, new_y]
        for k in range(16):
            if k != idx and points[k] == new_point:
                return None
        return idx, new_point

    def targeted_move(points):
        """Try to create a 4-point line by moving one point to align with 3 others."""
        n = len(points)

        # Find lines with exactly 3 points
        line_to_points = {}
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                if key not in line_to_points:
                    line_to_points[key] = set()
                line_to_points[key].add(i)
                line_to_points[key].add(j)

        # Find lines with exactly 3 points
        for key, point_set in line_to_points.items():
            if len(point_set) == 3:
                pts = [points[i] for i in point_set]
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                x3, y3 = pts[2]

                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g

                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy

                # Try positions along the line
                for t in range(-30, 31):
                    new_x = x1 + t * dx
                    new_y = y1 + t * dy

                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue

                    new_point = [new_x, new_y]

                    if new_point == pts[0] or new_point == pts[1] or new_point == pts[2]:
                        continue

                    is_distinct = True
                    for k in range(n):
                        if points[k] == new_point:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue

                    # Check that this doesn't create a 5-point line
                    line_count = 0
                    for k in range(n):
                        if k in point_set:
                            continue
                        px, py = points[k]
                        if (py - y1) * dx == (px - x1) * dy:
                            line_count += 1

                    if line_count >= 2:
                        continue

                    idx = random.choice(list(point_set))
                    return idx, new_point

        return None

    no_improve_count = 0
    max_no_improve = 300

    while time.time() - start_time < time_limit:
        # Try targeted move first
        if random.random() < 0.4:
            move = targeted_move(current_points)
        else:
            move = random_move(current_points)

        if move is None:
            continue

        idx, new_point = move
        old_point = current_points[idx][:]
        current_points[idx] = new_point
        new_score = evaluate_fast_score(current_points)

        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[idx] = old_point
            no_improve_count += 1

        if no_improve_count > max_no_improve:
            # Kick
            idx = random.randint(0, 15)
            new_x = random.randint(-200, 200)
            new_y = random.randint(-200, 200)
            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                is_distinct = True
                for k in range(16):
                    if k != idx and current_points[k] == [new_x, new_y]:
                        is_distinct = False
                        break
                if is_distinct:
                    current_points[idx] = [new_x, new_y]
                    current_score = evaluate_fast_score(current_points)
            no_improve_count = 0

    return best_points
