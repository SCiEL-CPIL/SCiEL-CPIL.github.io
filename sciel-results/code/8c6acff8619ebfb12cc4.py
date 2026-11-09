import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 58.0  # Safety margin before 90s limit

    def elapsed():
        return time.time() - t0

    def compute_max_area_count_fast(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area.
        Optimized using pairwise difference vectors and numpy.
        """
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)

        # Build all pairwise difference vectors (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]  # shape (M, 2), M = n*(n-1)/2
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)

        # For each base vector a (index u), compute cross products with all later vectors v
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)

        if not area_counts:
            return 0
        return max(area_counts.values())

    def score_of(pts):
        return compute_max_area_count_fast(pts)

    def count_area_A(pts, A):
        """Count how many triples have absolute doubled area exactly A."""
        n = len(pts)
        if n < 3 or A <= 0:
            return 0
        P = np.array(pts, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)
        count = 0
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            count += int(np.sum(np.abs(cross) == A))
        return count

    def full_area_counter(pts):
        """Return dict mapping area -> count for all non-zero areas."""
        n = len(pts)
        if n < 3:
            return {}
        P = np.array(pts, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)
        return dict(area_counts)

    def local_search_for_A(pts, A, time_budget):
        """
        Local search maximizing count of triangles with area exactly A.
        Returns (best_points, best_count_A, best_max_area).
        """
        current = list(pts)
        current_count_A = count_area_A(current, A)
        current_max_area = score_of(current)
        best = list(current)
        best_count_A = current_count_A
        best_max_area = current_max_area

        t_start = time.time()
        no_improve = 0
        step_size = 3
        random.seed(int(A) % 10000 + 42)

        while time.time() - t_start < time_budget:
            idx = random.randint(0, 29)
            old_x, old_y = current[idx]

            r = random.random()
            if r < 0.05:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            elif r < 0.2:
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
            else:
                new_x = old_x + random.randint(-step_size, step_size)
                new_y = old_y + random.randint(-step_size, step_size)

            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                continue
            new_pt = (new_x, new_y)
            if new_pt == (old_x, old_y):
                continue
            is_valid = True
            for j in range(30):
                if j != idx and current[j] == new_pt:
                    is_valid = False
                    break
            if not is_valid:
                continue

            current[idx] = new_pt
            new_count_A = count_area_A(current, A)

            if new_count_A > current_count_A:
                # Accept if it improves area A count
                current_count_A = new_count_A
                no_improve = 0
                # Also check max area
                new_max_area = score_of(current)
                if new_max_area > best_max_area:
                    best_max_area = new_max_area
                    best = list(current)
                    best_count_A = new_count_A
                elif new_count_A > best_count_A:
                    best_count_A = new_count_A
                    best = list(current)
                    best_max_area = new_max_area
                if step_size > 1 and random.random() < 0.2:
                    step_size = max(1, step_size - 1)
            else:
                current[idx] = (old_x, old_y)
                no_improve += 1

            if no_improve > 1000:
                current = list(best)
                current_count_A = best_count_A
                no_improve = 0
                step_size = random.randint(4, 10)

            if random.random() < 0.005:
                step_size = random.randint(1, 6)

        return best, best_count_A, best_max_area

    # Generate diverse initial candidates
    candidates = []

    # Candidate 1: Two horizontal lines
    c1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    candidates.append(c1)

    # Candidate 2: Grid 6x5
    c2 = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(c2)

    # Candidate 3: Three lines
    c3 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    candidates.append(c3)

    # Candidate 4: Random
    random.seed(42)
    c4 = []
    used = set()
    while len(c4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            c4.append((x, y))
    candidates.append(c4)

    # Candidate 5: Five lines, 6 each
    c5 = [(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + \
         [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + \
         [(x, 4) for x in range(6)]
    candidates.append(c5)

    # Candidate 6: Two lines offset
    c6 = [(x, 0) for x in range(15)] + [(x, 2) for x in range(15)]
    candidates.append(c6)

    # Candidate 7: Three lines with spacing
    c7 = [(x, 0) for x in range(10)] + [(x, 2) for x in range(10)] + [(x, 4) for x in range(10)]
    candidates.append(c7)

    # Candidate 8: Four lines
    c8 = [(x, 0) for x in range(8)] + [(x, 1) for x in range(8)] + \
         [(x, 3) for x in range(7)] + [(x, 5) for x in range(7)]
    candidates.append(c8)

    # Evaluate all candidates
    best_points = None
    best_score = 0
    best_area_A = 0

    for cand in candidates:
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            # Find the dominant area
            counter = full_area_counter(cand)
            if counter:
                best_area_A = max(counter, key=counter.get)

    # Phase 1: Local search from best candidate (maximizing max area)
    current_points = list(best_points)
    current_score = best_score

    t_phase1_start = time.time()
    phase1_budget = min(15.0, DEADLINE - elapsed() - 20.0)
    if phase1_budget < 2.0:
        phase1_budget = 2.0

    random.seed(123)
    no_improve = 0
    step_size = 3

    while time.time() - t_phase1_start < phase1_budget and elapsed() < DEADLINE - 15.0:
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]

        r = random.random()
        if r < 0.03:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.15:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)

        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
        new_pt = (new_x, new_y)
        if new_pt == (old_x, old_y):
            continue
        is_valid = True
        for j in range(30):
            if j != idx and current_points[j] == new_pt:
                is_valid = False
                break
        if not is_valid:
            continue

        current_points[idx] = new_pt
        new_score = score_of(current_points)

        if new_score > current_score:
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                counter = full_area_counter(current_points)
                if counter:
                    best_area_A = max(counter, key=counter.get)
            if step_size > 1 and random.random() < 0.2:
                step_size = max(1, step_size - 1)
        else:
            current_points[idx] = (old_x, old_y)
            no_improve += 1

        if no_improve > 1500:
            current_points = list(best_points)
            current_score = best_score
            no_improve = 0
            step_size = random.randint(4, 10)

        if random.random() < 0.003:
            step_size = random.randint(1, 6)

    # Phase 2: Model-and-optimize - try multiple target areas A
    # Use remaining time for targeted area optimization
    remaining_time = DEADLINE - elapsed() - 2.0
    if remaining_time > 3.0:
        # Get top candidate areas from current best
        counter = full_area_counter(best_points)
        if counter:
            # Sort areas by count, try top few
            sorted_areas = sorted(counter.items(), key=lambda x: -x[1])
            # Also add some small areas that might be promising
            candidate_areas = [a for a, c in sorted_areas[:5]]
            # Add some small values
            for a in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
                if a not in candidate_areas:
                    candidate_areas.append(a)
            candidate_areas = candidate_areas[:12]  # Limit number of areas to try

            time_per_area = max(1.0, remaining_time / len(candidate_areas))

            for A in candidate_areas:
                if elapsed() > DEADLINE - 2.0:
                    break
                t_area_start = time.time()
                area_budget = min(time_per_area, DEADLINE - elapsed() - 2.0)
                if area_budget < 0.5:
                    break

                # Start from current best and do local search for area A
                pts_for_A = list(best_points)
                result_pts, result_count_A, result_max = local_search_for_A(
                    pts_for_A, A, area_budget
                )

                if result_max > best_score:
                    best_score = result_max
                    best_points = result_pts

    # Final check
    if elapsed() < DEADLINE - 1.0:
        final_score = score_of(best_points)
        if final_score > best_score:
            best_score = final_score

    return best_points
