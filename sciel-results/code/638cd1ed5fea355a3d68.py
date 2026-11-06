import time
import random

def construct():
    start_time = time.time()
    timeout = 135  # Safety margin before 150s kill

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def has_4_collinear(points):
        n = len(points)
        # Precompute all pairs and check for 4 collinear
        # For each pair (i, j), count how many other points are collinear
        for i in range(n):
            for j in range(i + 1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 2:
                    return True
        return False

    def get_collinear_triples(points):
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def max_no_3_collinear_exact(points):
        n = len(points)
        if n <= 2:
            return n

        triples = get_collinear_triples(points)
        if not triples:
            return n

        # Precompute conflict pairs for fast validity checking
        # conflict_pairs[i] = list of (j, k) such that (i, j, k) is a triple
        conflict_pairs = [[] for _ in range(n)]
        for u, v, w in triples:
            conflict_pairs[u].append((v, w))
            conflict_pairs[v].append((u, w))
            conflict_pairs[w].append((u, v))

        # Precompute degree of each point
        deg = [len(conflict_pairs[i]) for i in range(n)]

        best_size = 0
        all_mask = (1 << n) - 1

        def solve(available_mask, current_size, current_mask):
            nonlocal best_size

            remaining = bin(available_mask).count('1')
            if current_size + remaining <= best_size:
                return

            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return

            # Pick point with highest degree among available
            # Use a simple heuristic: try points in order of degree
            # For speed, just pick the lowest set bit
            v = (available_mask & -available_mask).bit_length() - 1
            v_bit = 1 << v

            # Option 1: Exclude v
            solve(available_mask ^ v_bit, current_size, current_mask)

            # Option 2: Include v (if valid)
            valid = True
            for u, w in conflict_pairs[v]:
                if (current_mask & (1 << u)) and (current_mask & (1 << w)):
                    valid = False
                    break

            if valid:
                solve(available_mask ^ v_bit, current_size + 1, current_mask | v_bit)

        solve(all_mask, 0, 0)
        return best_size

    def score_config(points):
        return max_no_3_collinear_exact(points)

    best_score = 20
    best_points = None

    def local_search(points, max_time=5):
        """Local search to minimize the score."""
        current_points = [p[:] for p in points]
        current_score = score_config(current_points)
        start_ls = time.time()
        improved = True

        while improved and (time.time() - start_ls) < max_time and (time.time() - start_time) < timeout - 10:
            improved = False
            indices = list(range(20))
            random.shuffle(indices)

            for i in indices:
                if (time.time() - start_time) > timeout - 10:
                    return current_points, current_score

                old_x, old_y = current_points[i]

                # Try moving point i to various positions
                for attempt in range(20):
                    # Mix of small and medium moves
                    if attempt < 10:
                        dx = random.randint(-5, 5)
                        dy = random.randint(-5, 5)
                    else:
                        dx = random.randint(-20, 20)
                        dy = random.randint(-20, 20)

                    new_x = old_x + dx
                    new_y = old_y + dy

                    if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                        continue
                    if (new_x, new_y) == (old_x, old_y):
                        continue

                    # Check collision
                    collision = False
                    for j, p in enumerate(current_points):
                        if j != i and p[0] == new_x and p[1] == new_y:
                            collision = True
                            break
                    if collision:
                        continue

                    # Temporarily move
                    current_points[i] = [new_x, new_y]

                    if has_4_collinear(current_points):
                        current_points[i] = [old_x, old_y]
                        continue

                    new_score = score_config(current_points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current_points[i] = [old_x, old_y]

                # Early exit check
                if (time.time() - start_time) > timeout - 10:
                    return current_points, current_score

        return current_points, current_score

    # Strategy 1: Staggered grid with controlled collinearities
    def generate_staggered_grid(seed):
        random.seed(seed)
        pts = []
        used = set()

        # 7 rows, 3 points per row (last row has 2)
        # Stagger x-coordinates to avoid verticals and diagonals
        base_positions = [
            [0, 10, 20],
            [2, 12, 22],
            [4, 14, 24],
            [1, 11, 21],
            [3, 13, 23],
            [0, 10, 20],
            [5, 15]
        ]

        for y_idx, row in enumerate(base_positions):
            offset = random.randint(-3, 3)
            for x_base in row:
                x = x_base + offset + random.randint(-1, 1)
                y = y_idx * 2 + random.randint(0, 1)

                # Ensure distinct and in bounds
                attempts = 0
                while (x, y) in used and attempts < 50:
                    x += random.randint(-2, 2)
                    y += random.randint(-1, 1)
                    attempts += 1

                if -100 <= x <= 100 and -100 <= y <= 100 and (x, y) not in used:
                    pts.append([x, y])
                    used.add((x, y))

        # Fill remaining if needed
        while len(pts) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))

        return pts[:20]

    # Strategy 2: Line-based configuration with many 3-point lines
    def generate_line_based(seed):
        random.seed(seed)
        pts = []
        used = set()

        # Define lines with 3 points each
        # Use different slopes to avoid accidental 4-collinear
        line_defs = [
            # (slope, intercept, x_values)
            (0, 0, [-5, 0, 5]),
            (0, 2, [-3, 2, 7]),
            (0, 4, [-1, 4, 9]),
            (1, 0, [-4, 1, 6]),
            (1, 3, [-2, 3, 8]),
            (-1, 5, [-3, 2, 7]),
            (2, 0, [-2, 1, 4]),
        ]

        for m, b, xs in line_defs:
            if len(pts) >= 20:
                break
            line_pts = []
            valid = True
            for x in xs:
                y = m * x + b
                if -100 <= x <= 100 and -100 <= y <= 100 and (x, y) not in used:
                    line_pts.append((x, y))
                else:
                    valid = False
                    break

            if valid and len(line_pts) == 3:
                for p in line_pts:
                    pts.append([p[0], p[1]])
                    used.add(p)

        # Add remaining points
        while len(pts) < 20:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))

        return pts[:20]

    # Strategy 3: Random dense configuration
    def generate_random(seed):
        random.seed(seed)
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-40, 40)
            y = random.randint(-40, 40)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        return pts

    # Generate candidates from all strategies
    candidates = []

    # Staggered grids
    for seed in range(30):
        pts = generate_staggered_grid(seed)
        if len(pts) == 20 and not has_4_collinear(pts):
            candidates.append(pts)

    # Line-based
    for seed in range(20):
        pts = generate_line_based(seed)
        if len(pts) == 20 and not has_4_collinear(pts):
            candidates.append(pts)

    # Random
    for seed in range(30):
        pts = generate_random(1000 + seed)
        if not has_4_collinear(pts):
            candidates.append(pts)

    # Score all candidates
    scored_candidates = []
    for cand in candidates:
        if time.time() - start_time > timeout - 30:
            break
        s = score_config(cand)
        scored_candidates.append((s, cand))

    # Sort by score (lower is better)
    scored_candidates.sort(key=lambda x: x[0])

    # Run local search on top candidates
    for s, cand in scored_candidates[:40]:
        if time.time() - start_time > timeout - 15:
            break
        if s >= best_score:
            continue

        pts, new_s = local_search(cand, max_time=4)
        if new_s < best_score:
            best_score = new_s
            best_points = pts

    # If still not good enough, try more aggressive random search
    if best_score >= 17:
        for i in range(30):
            if time.time() - start_time > timeout - 10:
                break
            random.seed(5000 + i)
            pts = []
            used = set()
            while len(pts) < 20:
                x = random.randint(-80, 80)
                y = random.randint(-80, 80)
                if (x, y) not in used:
                    pts.append([x, y])
                    used.add((x, y))
            if not has_4_collinear(pts):
                pts, s = local_search(pts, max_time=3)
                if s < best_score:
                    best_score = s
                    best_points = pts

    # Fallback: simple valid config (parabola)
    if best_points is None:
        pts = []
        for x in range(-5, 6):
            pts.append([x, x * x])
        for x in range(-5, 6):
            pts.append([x, x * x + 25])
        best_points = pts
        best_score = score_config(best_points)

    return best_points
