import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Candidate grid: 6x6 = 36 points
    N = 6
    points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(points)  # 36
    n_select = 16

    # Precompute all lines (maximal collinear subsets of size >= 2)
    # Represent each line as a bitmask of point indices
    line_set = set()  # frozenset of indices -> line
    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            pts = {i, j}
            for k in range(n_points):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    pts.add(k)
            line_set.add(frozenset(pts))

    # Convert to bitmask representation
    all_lines = []  # list of (bitmask, size)
    for line in line_set:
        mask = 0
        for idx in line:
            mask |= (1 << idx)
        all_lines.append((mask, len(line)))

    # Lines by size
    lines_by_size = defaultdict(list)
    for mask, sz in all_lines:
        lines_by_size[sz].append(mask)

    # For fast delta scoring: for each point, which lines contain it?
    point_lines = defaultdict(list)  # point_idx -> list of line indices
    for li, (mask, sz) in enumerate(all_lines):
        for idx in range(n_points):
            if mask & (1 << idx):
                point_lines[idx].append(li)

    # Precompute line bitmasks as a list for fast access
    line_masks = [m for m, sz in all_lines]
    line_sizes = [sz for m, sz in all_lines]
    n_lines = len(all_lines)

    def is_valid_mask(mask):
        """Check no 5 points are collinear."""
        for sz, masks in lines_by_size.items():
            if sz >= 5:
                for lm in masks:
                    if (mask & lm) == lm:
                        return False
        return True

    def score_mask(mask):
        """Count lines with exactly 4 of the selected points."""
        score = 0
        for li in range(n_lines):
            lm = line_masks[li]
            common = mask & lm
            cnt = bin(common).count('1')
            if cnt == 4:
                score += 1
        return score

    def delta_score(mask, remove_idx, add_idx):
        """
        Compute score change when removing remove_idx and adding add_idx.
        Only lines containing remove_idx or add_idx can change.
        Returns (new_score - old_score, new_mask).
        """
        new_mask = mask & ~(1 << remove_idx)
        new_mask |= (1 << add_idx)

        # Lines affected: those containing remove_idx or add_idx
        affected = set(point_lines[remove_idx]) | set(point_lines[add_idx])

        delta = 0
        for li in affected:
            lm = line_masks[li]
            # Old count
            old_common = mask & lm
            old_cnt = bin(old_common).count('1')
            # New count
            new_common = new_mask & lm
            new_cnt = bin(new_common).count('1')

            old_contributes = 1 if old_cnt == 4 else 0
            new_contributes = 1 if new_cnt == 4 else 0
            delta += new_contributes - old_contributes

        return delta, new_mask

    def is_valid_mask_fast(mask):
        """Fast validity check using precomputed 5+ lines."""
        for sz, masks in lines_by_size.items():
            if sz >= 5:
                for lm in masks:
                    if (mask & lm) == lm:
                        return False
        return True

    # Simulated annealing
    def simulated_annealing(initial_mask, iterations, start_temp, end_temp):
        nonlocal best_score, best_mask
        current = initial_mask
        current_score = score_mask(current)

        if not is_valid_mask_fast(current):
            return

        if current_score > best_score:
            best_score = current_score
            best_mask = current

        for it in range(iterations):
            if time.time() > deadline:
                return

            # Pick random point to swap out (from current) and in (from outside)
            in_list = [i for i in range(n_points) if current & (1 << i)]
            out_list = [i for i in range(n_points) if not (current & (1 << i))]

            if not in_list or not out_list:
                continue

            p_in = random.choice(in_list)
            p_out = random.choice(out_list)

            delta, new_mask = delta_score(current, p_in, p_out)

            if not is_valid_mask_fast(new_mask):
                continue

            new_score = current_score + delta

            # Temperature
            t = start_temp * (end_temp / start_temp) ** (it / max(iterations - 1, 1))

            if new_score >= current_score or random.random() < math.exp((new_score - current_score) / max(t, 1e-10)):
                current = new_mask
                current_score = new_score

                if current_score > best_score:
                    best_score = current_score
                    best_mask = current

    def local_search(mask, max_passes=10):
        """Exhaustive single-swap local search."""
        nonlocal best_score, best_mask
        current = mask
        current_score = score_mask(current)

        if not is_valid_mask_fast(current):
            return

        if current_score > best_score:
            best_score = current_score
            best_mask = current

        for _pass in range(max_passes):
            if time.time() > deadline - 2:
                break

            improved = False
            in_list = [i for i in range(n_points) if current & (1 << i)]
            out_list = [i for i in range(n_points) if not (current & (1 << i))]

            for p_in in in_list:
                if time.time() > deadline - 2:
                    break
                for p_out in out_list:
                    delta, new_mask = delta_score(current, p_in, p_out)
                    if not is_valid_mask_fast(new_mask):
                        continue
                    new_score = current_score + delta
                    if new_score > current_score:
                        current = new_mask
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current

            if not improved:
                break

        return current, current_score

    def double_swap_search(mask, max_passes=5):
        """Try all double swaps (remove 2, add 2) for further improvement."""
        nonlocal best_score, best_mask
        current = mask
        current_score = score_mask(current)

        if not is_valid_mask_fast(current):
            return

        for _pass in range(max_passes):
            if time.time() > deadline - 2:
                break

            improved = False
            in_list = [i for i in range(n_points) if current & (1 << i)]
            out_list = [i for i in range(n_points) if not (current & (1 << i))]

            for i1 in range(len(in_list)):
                if time.time() > deadline - 2:
                    break
                for i2 in range(i1 + 1, len(in_list)):
                    if time.time() > deadline - 2:
                        break
                    r1, r2 = in_list[i1], in_list[i2]
                    for a1 in range(len(out_list)):
                        if time.time() > deadline - 2:
                            break
                        for a2 in range(a1 + 1, len(out_list)):
                            if time.time() > deadline - 2:
                                break
                            ad1, ad2 = out_list[a1], out_list[a2]

                            new_mask = current & ~(1 << r1) & ~(1 << r2)
                            new_mask |= (1 << ad1) | (1 << ad2)

                            if not is_valid_mask_fast(new_mask):
                                continue

                            new_score = score_mask(new_mask)
                            if new_score > current_score:
                                current = new_mask
                                current_score = new_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_mask = current

            if not improved:
                break

        return current, current_score

    # Main search
    best_score = -1
    best_mask = None

    # Generate diverse initial configurations
    def make_initial(r):
        if r == 0:
            # Structured: first 16 points (4x4 subgrid)
            return (1 << 16) - 1
        elif r == 1:
            # Rows 0-2, all 6 columns = 18, pick 16
            mask = 0
            for x in range(3):
                for y in range(N):
                    mask |= (1 << (x * N + y))
            # Remove 2 random
            bits = [i for i in range(n_points) if mask & (1 << i)]
            rem = random.sample(bits, 2)
            for i in rem:
                mask &= ~(1 << i)
            return mask
        elif r == 2:
            # Checkerboard-like
            mask = 0
            for x in range(N):
                for y in range(N):
                    if (x + y) % 2 == 0:
                        mask |= (1 << (x * N + y))
            # 18 points, remove 2
            bits = [i for i in range(n_points) if mask & (1 << i)]
            rem = random.sample(bits, 2)
            for i in rem:
                mask &= ~(1 << i)
            return mask
        else:
            # Random
            bits = random.sample(range(n_points), n_select)
            mask = 0
            for i in bits:
                mask |= (1 << i)
            return mask

    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break

        initial = make_initial(r)

        # Try to make valid
        attempts = 0
        while not is_valid_mask_fast(initial) and attempts < 20:
            # Perturb: swap one point
            in_list = [i for i in range(n_points) if initial & (1 << i)]
            out_list = [i for i in range(n_points) if not (initial & (1 << i))]
            if not in_list or not out_list:
                break
            p_in = random.choice(in_list)
            p_out = random.choice(out_list)
            initial = initial & ~(1 << p_in) | (1 << p_out)
            attempts += 1

        if not is_valid_mask_fast(initial):
            continue

        # Simulated annealing
        remaining = deadline - 5 - time.time()
        if remaining < 1:
            break
        iters = max(200, int(remaining * 15))
        simulated_annealing(initial, iters, 5.0, 0.05)

        # Local search polish
        if best_mask is not None:
            local_search(best_mask, max_passes=8)

    # Final polish with double swaps
    if best_mask is not None and time.time() < deadline - 3:
        double_swap_search(best_mask, max_passes=3)
        # One more single-swap pass
        local_search(best_mask, max_passes=5)

    # Fallback
    if best_mask is None:
        # Try to find any valid 16-point subset
        for _ in range(500):
            bits = random.sample(range(n_points), n_select)
            mask = 0
            for i in bits:
                mask |= (1 << i)
            if is_valid_mask_fast(mask):
                best_mask = mask
                best_score = score_mask(mask)
                break
        if best_mask is None:
            # Last resort: use first 16
            best_mask = (1 << 16) - 1

    # Extract points
    result = []
    for i in range(n_points):
        if best_mask & (1 << i):
            result.append([points[i][0], points[i][1]])

    return result
