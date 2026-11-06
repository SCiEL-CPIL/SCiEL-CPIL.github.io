import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Grid parameters - use a larger grid for more line options
    GRID_SIZE = 16
    HALF = GRID_SIZE // 2
    grid_points = []
    for x in range(-HALF, HALF):
        for y in range(-HALF, HALF):
            grid_points.append((x, y))

    N_GRID = len(grid_points)

    # Precompute all lines with their point sets
    line_map = {}
    line_points = []  # list of sets of point indices
    point_line_ids = [[] for _ in range(N_GRID)]

    for i in range(N_GRID):
        for j in range(i + 1, N_GRID):
            x1, y1 = grid_points[i]
            x2, y2 = grid_points[j]
            dx = x2 - x1
            dy = y2 - y1

            # Normalize direction
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
                elif dx == 0 and dy < 0:
                    dy = -dy

            # Normal vector (a, b)
            a = dy
            b = -dx
            c = a * x1 + b * y1

            key = (a, b, c)

            if key not in line_map:
                line_map[key] = len(line_points)
                line_points.append(set())

            lid = line_map[key]
            line_points[lid].add(i)
            line_points[lid].add(j)

            point_line_ids[i].append(lid)
            point_line_ids[j].append(lid)

    # Deduplicate point_line_ids
    for i in range(N_GRID):
        point_line_ids[i] = list(set(point_line_ids[i]))

    num_lines = len(line_points)

    # Find all lines with exactly 4 points
    four_point_lines = []
    for lid in range(num_lines):
        if len(line_points[lid]) == 4:
            four_point_lines.append(lid)

    num_four_lines = len(four_point_lines)
    four_point_indices = [line_points[lid] for lid in four_point_lines]

    # Build compatibility graph: two 4-point lines are compatible if they share at most 3 points
    # (sharing 4 points means they're the same line, which can't happen since we have unique lines)
    # Actually, two different lines can share at most 1 point in general position, but in a grid
    # they can share more. Two lines are compatible if their intersection has size <= 3.
    # If they share 4 points, they'd be the same line (impossible for distinct lines).
    # The real constraint: selecting a set of 4-point lines must use exactly 16 distinct points
    # and no point can be in 5 or more selected lines (to avoid 5-collinear).
    # Actually, a point in k selected 4-point lines is on k lines. If k >= 2, the point is shared.
    # The constraint is: no 5 points collinear. A point being in multiple 4-point lines is fine
    # as long as no line has 5+ of our 16 points.

    # Strategy: Greedy + local search over the set of 4-point lines
    # State: set of selected 4-point lines
    # Score: number of selected lines (each contributes 1 to the count of exactly-4-point lines)
    # Constraint: union of points has size <= 16, and no line has 5+ selected points

    # Precompute: for each 4-point line, which other 4-point lines share points with it
    # and how many points they share
    line_compat = [set() for _ in range(num_four_lines)]
    line_shared_count = [[0] * num_four_lines for _ in range(num_four_lines)]

    for i in range(num_four_lines):
        for j in range(i + 1, num_four_lines):
            shared = four_point_indices[i] & four_point_indices[j]
            if len(shared) <= 3:  # Always true for distinct lines
                line_compat[i].add(j)
                line_compat[j].add(i)
            line_shared_count[i][j] = len(shared)
            line_shared_count[j][i] = len(shared)

    def evaluate_line_set(selected_lines):
        """Evaluate a set of selected 4-point lines.
        Returns (score, num_points, max_line_count, is_valid)
        score = number of lines with exactly 4 of the 16 points
        """
        point_counts = {}
        for lid in selected_lines:
            for p in four_point_indices[lid]:
                point_counts[p] = point_counts.get(p, 0) + 1

        num_points = len(point_counts)
        if num_points > 16:
            return 0, num_points, 0, False

        # Count lines with exactly 4 points among our selected points
        # A line has exactly 4 points if all 4 of its points are in our set
        # and no 5th point from our set is on that line
        score = 0
        for lid in range(num_lines):
            lp = line_points[lid]
            count_in_set = 0
            for p in lp:
                if p in point_counts:
                    count_in_set += 1
            if count_in_set == 4:
                score += 1
            elif count_in_set >= 5:
                return 0, num_points, count_in_set, False

        return score, num_points, max(point_counts.values()) if point_counts else 0, True

    best_score = -1
    best_points = None

    # Greedy construction: start with a 4x4 grid, then add 4-point lines
    def get_grid_indices():
        indices = []
        for x in range(-2, 2):
            for y in range(-2, 2):
                idx = (x + HALF) * GRID_SIZE + (y + HALF)
                indices.append(idx)
        return indices

    # Local search over line selections
    def local_search_line_set(initial_lines, max_time):
        nonlocal best_score, best_points

        selected = set(initial_lines)
        point_counts = {}
        for lid in selected:
            for p in four_point_indices[lid]:
                point_counts[p] = point_counts.get(p, 0) + 1

        score, num_pts, max_cnt, valid = evaluate_line_set(selected)
        if not valid:
            # Try to fix by removing conflicting lines
            return None, -1

        best_local_score = score
        best_local_lines = set(selected)

        T = 2.0
        T_min = 0.01
        cooling = 0.995

        while time.time() < max_time and T > T_min:
            # Try to add or remove a line
            if random.random() < 0.7:
                # Add a random compatible line
                candidates = []
                for lid in range(num_four_lines):
                    if lid not in selected:
                        # Check if adding it keeps us valid
                        new_point_counts = dict(point_counts)
                        for p in four_point_indices[lid]:
                            new_point_counts[p] = new_point_counts.get(p, 0) + 1
                        if len(new_point_counts) <= 16:
                            candidates.append(lid)

                if candidates:
                    lid = random.choice(candidates)
                    # Add lid
                    selected.add(lid)
                    for p in four_point_indices[lid]:
                        point_counts[p] = point_counts.get(p, 0) + 1

                    new_score, new_num_pts, new_max, new_valid = evaluate_line_set(selected)

                    if new_valid and new_score >= score:
                        score = new_score
                        if score > best_local_score:
                            best_local_score = score
                            best_local_lines = set(selected)
                    elif new_valid:
                        delta = new_score - score
                        if random.random() < math.exp(delta / T):
                            score = new_score
                        else:
                            # Revert
                            selected.discard(lid)
                            for p in four_point_indices[lid]:
                                point_counts[p] -= 1
                                if point_counts[p] == 0:
                                    del point_counts[p]
                    else:
                        # Revert
                        selected.discard(lid)
                        for p in four_point_indices[lid]:
                            point_counts[p] -= 1
                            if point_counts[p] == 0:
                                del point_counts[p]
            else:
                # Remove a random line
                if selected:
                    lid = random.choice(list(selected))
                    selected.discard(lid)
                    for p in four_point_indices[lid]:
                        point_counts[p] -= 1
                        if point_counts[p] == 0:
                            del point_counts[p]

                    new_score, new_num_pts, new_max, new_valid = evaluate_line_set(selected)
                    score = new_score if new_valid else 0

            T *= cooling

        return best_local_lines, best_local_score

    # Initial: 4x4 grid
    grid_idx = get_grid_indices()
    grid_set = set(grid_idx)

    # Find 4-point lines fully contained in the 4x4 grid
    initial_lines = []
    for i, lid in enumerate(four_point_lines):
        if four_point_indices[i].issubset(grid_set):
            initial_lines.append(lid)

    # Run local search from multiple starts
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break

        if r == 0:
            start_lines = initial_lines[:]
        elif r % 2 == 0:
            # Random subset of 4-point lines
            start_lines = random.sample(range(num_four_lines), min(8, num_four_lines))
        else:
            # Start from a random 16-point set, find its 4-point lines
            random_pts = set(random.sample(range(N_GRID), 16))
            start_lines = []
            for i, lid in enumerate(four_point_lines):
                if four_point_indices[i].issubset(random_pts):
                    start_lines.append(lid)

        result_lines, result_score = local_search_line_set(start_lines, deadline - 20)

        if result_lines is not None and result_score > best_score:
            best_score = result_score
            # Extract the 16 points from the selected lines
            all_pts = set()
            for lid in result_lines:
                all_pts.update(four_point_indices[lid])

            if len(all_pts) <= 16:
                # Pad to 16 if needed
                if len(all_pts) < 16:
                    remaining = [i for i in range(N_GRID) if i not in all_pts]
                    if remaining:
                        all_pts.update(random.sample(remaining, 16 - len(all_pts)))

                best_points = [grid_points[i] for i in list(all_pts)[:16]]

    # Also try the original SA approach as backup
    if best_points is None or best_score < 10:
        # Fall back to point-based SA
        def evaluate_initial(points_indices):
            counts = [0] * num_lines
            score = 0
            invalid = 0
            for p_idx in points_indices:
                for lid in point_line_ids[p_idx]:
                    counts[lid] += 1
            for c in counts:
                if c == 4:
                    score += 1
                elif c >= 5:
                    invalid += 1
            return counts, score, invalid

        def get_score(score, invalid):
            if invalid > 0:
                return -10000
            return score

        for r in range(4):
            if time.time() > deadline - 10:
                break

            if r % 2 == 0:
                current_indices = get_grid_indices()
            else:
                current_indices = random.sample(range(N_GRID), 16)

            counts, current_score, current_invalid = evaluate_initial(current_indices)
            current_score_val = get_score(current_score, current_invalid)

            if current_score_val > best_score:
                best_score = current_score_val
                best_points = [grid_points[i] for i in current_indices]

            T = 5.0
            T_min = 0.01
            cooling = 0.995
            active_set = set(current_indices)

            while time.time() < deadline - 5 and T > T_min:
                pos = random.randint(0, 15)
                p_out = current_indices[pos]

                p_in = None
                for _ in range(50):
                    candidate = random.randint(0, N_GRID - 1)
                    if candidate not in active_set:
                        p_in = candidate
                        break

                if p_in is None:
                    continue

                for lid in point_line_ids[p_out]:
                    c_old = counts[lid]
                    c_new = c_old - 1
                    counts[lid] = c_new
                    if c_old == 4 and c_new != 4:
                        current_score -= 1
                    elif c_old != 4 and c_new == 4:
                        current_score += 1
                    elif c_old < 5 and c_new >= 5:
                        current_invalid += 1
                    elif c_old >= 5 and c_new < 5:
                        current_invalid -= 1

                for lid in point_line_ids[p_in]:
                    c_old = counts[lid]
                    c_new = c_old + 1
                    counts[lid] = c_new
                    if c_old == 4 and c_new != 4:
                        current_score -= 1
                    elif c_old != 4 and c_new == 4:
                        current_score += 1
                    elif c_old < 5 and c_new >= 5:
                        current_invalid += 1
                    elif c_old >= 5 and c_new < 5:
                        current_invalid -= 1

                new_score_val = get_score(current_score, current_invalid)

                if new_score_val >= current_score_val:
                    current_indices[pos] = p_in
                    active_set.remove(p_out)
                    active_set.add(p_in)
                    current_score_val = new_score_val
                    if current_score_val > best_score:
                        best_score = current_score_val
                        best_points = [grid_points[i] for i in current_indices]
                else:
                    delta = new_score_val - current_score_val
                    if random.random() < math.exp(delta / T):
                        current_indices[pos] = p_in
                        active_set.remove(p_out)
                        active_set.add(p_in)
                        current_score_val = new_score_val
                    else:
                        for lid in point_line_ids[p_out]:
                            c_old = counts[lid]
                            c_new = c_old + 1
                            counts[lid] = c_new
                            if c_old == 4 and c_new != 4:
                                current_score -= 1
                            elif c_old != 4 and c_new == 4:
                                current_score += 1
                            elif c_old < 5 and c_new >= 5:
                                current_invalid += 1
                            elif c_old >= 5 and c_new < 5:
                                current_invalid -= 1

                        for lid in point_line_ids[p_in]:
                            c_old = counts[lid]
                            c_new = c_old - 1
                            counts[lid] = c_new
                            if c_old == 4 and c_new != 4:
                                current_score -= 1
                            elif c_old != 4 and c_new == 4:
                                current_score += 1
                            elif c_old < 5 and c_new >= 5:
                                current_invalid += 1
                            elif c_old >= 5 and c_new < 5:
                                current_invalid -= 1

                T *= cooling

    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]

    return best_points
