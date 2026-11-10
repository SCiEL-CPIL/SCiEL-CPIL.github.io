import time
import random
import itertools
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0

    # Precompute triple indices
    n = 30
    triple_indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                triple_indices.append((i, j, k))
    triple_indices = np.array(triple_indices, dtype=np.int64)
    idx_i = triple_indices[:, 0]
    idx_j = triple_indices[:, 1]
    idx_k = triple_indices[:, 2]

    # 1. Fast scoring using precomputed indices
    def score(pts):
        arr = np.array(pts, dtype=np.int64)
        xi = arr[idx_i, 0]
        yi = arr[idx_i, 1]
        xj = arr[idx_j, 0]
        yj = arr[idx_j, 1]
        xk = arr[idx_k, 0]
        yk = arr[idx_k, 1]
        doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        nonzero = doubled_areas[doubled_areas > 0]
        if len(nonzero) == 0:
            return 0
        vals, cnts = np.unique(nonzero, return_counts=True)
        return int(cnts.max())

    def score_with_target(pts, target_area):
        """Score focusing on how many triangles have the target area"""
        arr = np.array(pts, dtype=np.int64)
        xi = arr[idx_i, 0]
        yi = arr[idx_i, 1]
        xj = arr[idx_j, 0]
        yj = arr[idx_j, 1]
        xk = arr[idx_k, 0]
        yk = arr[idx_k, 1]
        doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        count_target = np.sum(doubled_areas == target_area)
        nonzero = doubled_areas[doubled_areas > 0]
        if len(nonzero) == 0:
            return 0, 0
        vals, cnts = np.unique(nonzero, return_counts=True)
        return int(cnts.max()), int(count_target)

    # 2. Candidate Generation
    candidates = []

    # Rectangular grids with various spacings
    for w in [3, 5, 6, 10, 15]:
        h = 30 // w
        if h < 2:
            continue
        for dx in [1, 2, 3, 4, 5]:
            for dy in [1, 2, 3, 4, 5]:
                if dx * (w - 1) > 60 or dy * (h - 1) > 60:
                    continue
                for ox in range(-10, 11, 5):
                    for oy in range(-10, 11, 5):
                        pts = []
                        valid = True
                        for i in range(w):
                            for j in range(h):
                                x = ox + i * dx
                                y = oy + j * dy
                                if -30 <= x <= 30 and -30 <= y <= 30:
                                    pts.append((x, y))
                                else:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(pts) == 30:
                            candidates.append(pts)

    # Hexagonal lattice patterns
    for spacing in [3, 4, 5, 6]:
        for ox in range(-5, 6, 5):
            for oy in range(-5, 6, 5):
                pts = []
                for row in range(8):
                    for col in range(8):
                        x = ox + col * spacing
                        y = oy + row * spacing
                        if row % 2 == 1:
                            x += spacing // 2
                        if -30 <= x <= 30 and -30 <= y <= 30:
                            pts.append((x, y))
                if len(pts) >= 30:
                    pts = pts[:30]
                    candidates.append(pts)

    # Diagonal line patterns
    for slope in [1, 2, 3]:
        for intercept in range(-30, 31, 3):
            pts = []
            for x in range(-30, 31):
                y = slope * x + intercept
                if -30 <= y <= 30:
                    pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                candidates.append(pts)

    # Parabolic patterns
    for a in [1, 2, 3]:
        for c in range(-20, 21, 3):
            pts = []
            for x in range(-30, 31):
                y = a * x * x // 10 + c
                if -30 <= y <= 30:
                    pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                candidates.append(pts)

    # Random structured patterns
    random.seed(42)
    for _ in range(30):
        base_w = random.choice([3, 5, 6, 10])
        base_h = 30 // base_w
        dx = random.choice([1, 2, 3, 4])
        dy = random.choice([1, 2, 3, 4])
        ox = random.randint(-10, 10)
        oy = random.randint(-10, 10)

        pts = []
        for i in range(base_w):
            for j in range(base_h):
                x = ox + i * dx + random.randint(-1, 1)
                y = oy + j * dy + random.randint(-1, 1)
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))

        pts = list(set(pts))
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            p = (x, y)
            if p not in pts:
                pts.append(p)

        if len(pts) == 30:
            candidates.append(pts)

    # Pure random candidates
    for _ in range(15):
        random_pts = set()
        while len(random_pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            random_pts.add((x, y))
        candidates.append(list(random_pts))

    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # 3. Initial Evaluation
    scored_candidates = []
    for c in unique_candidates:
        if time.time() > total_deadline - 30:
            break
        s = score(c)
        scored_candidates.append((s, c))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    best_overall_score = scored_candidates[0][0] if scored_candidates else 0
    best_overall_pts = scored_candidates[0][1] if scored_candidates else []

    # 4. Local search with target-area guided moves
    def local_search(init_pts, duration, use_target_guidance=True):
        current_pts = [p for p in init_pts]
        current_score = score(current_pts)
        current_set = set(current_pts)
        end_time = time.time() + duration

        # Determine target area from current best
        arr = np.array(current_pts, dtype=np.int64)
        xi = arr[idx_i, 0]
        yi = arr[idx_i, 1]
        xj = arr[idx_j, 0]
        yj = arr[idx_j, 1]
        xk = arr[idx_k, 0]
        yk = arr[idx_k, 1]
        doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        nonzero = doubled_areas[doubled_areas > 0]
        if len(nonzero) > 0:
            vals, cnts = np.unique(nonzero, return_counts=True)
            target_area = int(vals[np.argmax(cnts)])
        else:
            target_area = 1

        T = 5.0
        T_min = 0.3
        cooling_rate = 0.9998

        while time.time() < end_time:
            i = random.randint(0, 29)
            old_p = current_pts[i]

            r = random.random()
            if r < 0.6:
                nx = old_p[0] + random.randint(-2, 2)
                ny = old_p[1] + random.randint(-2, 2)
            elif r < 0.85:
                nx = old_p[0] + random.randint(-5, 5)
                ny = old_p[1] + random.randint(-5, 5)
            else:
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)

            new_p = (nx, ny)
            if -30 <= nx <= 30 and -30 <= ny <= 30:
                if new_p != old_p and new_p not in current_set:
                    current_pts[i] = new_p
                    current_set.remove(old_p)
                    current_set.add(new_p)

                    if use_target_guidance:
                        new_score, target_count = score_with_target(current_pts, target_area)
                        # Use target count as a tiebreaker/secondary signal
                        if new_score > current_score:
                            delta = 1
                        elif new_score == current_score:
                            # Slight preference for moves that increase target area count
                            delta = 0.1 if target_count > 0 else -0.5
                        else:
                            delta = new_score - current_score
                    else:
                        new_score = score(current_pts)
                        delta = new_score - current_score

                    if delta > 0:
                        current_score = new_score if not use_target_guidance else new_score
                    else:
                        if random.random() < np.exp(delta / T):
                            if use_target_guidance:
                                current_score = new_score
                            else:
                                current_score = new_score
                        else:
                            current_pts[i] = old_p
                            current_set.remove(new_p)
                            current_set.add(old_p)

                    T = max(T_min, T * cooling_rate)

        return current_score, current_pts

    # 5. Polish pass - very fine-grained local search
    def polish(pts, duration):
        current_pts = [p for p in pts]
        current_score = score(current_pts)
        current_set = set(current_pts)
        end_time = time.time() + duration

        # Get target area
        arr = np.array(current_pts, dtype=np.int64)
        xi = arr[idx_i, 0]
        yi = arr[idx_i, 1]
        xj = arr[idx_j, 0]
        yj = arr[idx_j, 1]
        xk = arr[idx_k, 0]
        yk = arr[idx_k, 1]
        doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        nonzero = doubled_areas[doubled_areas > 0]
        if len(nonzero) > 0:
            vals, cnts = np.unique(nonzero, return_counts=True)
            target_area = int(vals[np.argmax(cnts)])
        else:
            target_area = 1

        improved = True
        while improved and time.time() < end_time:
            improved = False
            for i in range(30):
                if time.time() > end_time:
                    break
                old_p = current_pts[i]
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = old_p[0] + dx
                        ny = old_p[1] + dy
                        if -30 <= nx <= 30 and -30 <= ny <= 30:
                            new_p = (nx, ny)
                            if new_p != old_p and new_p not in current_set:
                                current_pts[i] = new_p
                                current_set.remove(old_p)
                                current_set.add(new_p)
                                new_score = score(current_pts)
                                if new_score > current_score:
                                    current_score = new_score
                                    improved = True
                                else:
                                    current_pts[i] = old_p
                                    current_set.remove(new_p)
                                    current_set.add(old_p)

        return current_score, current_pts

    # Run searches
    search_times = [25, 15, 10]
    for idx in range(min(3, len(scored_candidates))):
        if time.time() > total_deadline - 8:
            break

        s, p = scored_candidates[idx]
        rem_time = total_deadline - time.time() - 3
        if rem_time <= 0:
            break

        t = min(search_times[idx], rem_time)
        new_s, new_p = local_search(p, t, use_target_guidance=True)

        if new_s > best_overall_score:
            best_overall_score = new_s
            best_overall_pts = new_p

    # Final polish on best result
    if time.time() < total_deadline - 2 and best_overall_pts:
        polish_time = min(5.0, total_deadline - time.time() - 1)
        if polish_time > 1.0:
            polished_s, polished_p = polish(best_overall_pts, polish_time)
            if polished_s > best_overall_score:
                best_overall_score = polished_s
                best_overall_pts = polished_p

    return best_overall_pts
