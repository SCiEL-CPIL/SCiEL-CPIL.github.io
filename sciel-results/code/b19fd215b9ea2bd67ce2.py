"""3D container loading with spatial grid acceleration and diverse restarts."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 125.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return elapsed() > DEADLINE

    # Pre-compute unique rotations for each box
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots

    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]

    # 3D spatial grid for O(1) average collision checks
    # We use a 3D array of booleans marking occupied unit cells.
    # To keep memory reasonable, we only use it if container volume is not too large.
    # For large containers, we fall back to list-based overlap check.
    # Given typical sizes (e.g., up to 100^3 = 1e6 cells), a numpy bool array is fine.
    try:
        import numpy as np
        grid = np.zeros((W, H, D), dtype=bool)
        use_grid = True
    except:
        use_grid = False

    def mark_occupied(x, y, z, w, h, d):
        if use_grid:
            grid[x:x+w, y:y+h, z:z+d] = True

    def unmark_occupied(x, y, z, w, h, d):
        if use_grid:
            grid[x:x+w, y:y+h, z:z+d] = False

    def overlaps_grid(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return True
        if not use_grid:
            return False  # fallback handled elsewhere
        if grid[x:x+w, y:y+h, z:z+d].any():
            return True
        return False

    # List-based overlap for fallback or when grid is not used
    def overlaps_list(x, y, z, w, h, d, placed):
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    def overlaps(x, y, z, w, h, d, placed):
        if use_grid:
            return overlaps_grid(x, y, z, w, h, d)
        else:
            return overlaps_list(x, y, z, w, h, d, placed)

    # Generate candidate positions
    def candidate_positions(placed, W, H, D):
        candidates = set()
        candidates.add((0, 0, 0))
        for px, py, pz, pw, ph, pd in placed:
            if px + pw < W:
                candidates.add((px + pw, py, pz))
            if py + ph < H:
                candidates.add((px, py + ph, pz))
            if pz + pd < D:
                candidates.add((px, py, pz + pd))
            if px + pw < W and py + ph < H:
                candidates.add((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D:
                candidates.add((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D:
                candidates.add((px, py + ph, pz + pd))
            if px + pw < W and py + ph < H and pz + pd < D:
                candidates.add((px + pw, py + ph, pz + pd))
        valid = []
        for (x, y, z) in candidates:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        return valid

    def score_position(x, y, z, w, h, d, W, H, D):
        # Prefer lower y, then x, then z
        return y * 1.0 + x * 0.3 + z * 0.2

    def try_place_box(w, h, d, rots, placed, W, H, D, deadline_check):
        best = None
        best_score = float('inf')
        positions = candidate_positions(placed, W, H, D)
        for (rw, rh, rd) in rots:
            for (x, y, z) in positions:
                if x + rw > W or y + rh > H or z + rd > D:
                    continue
                if overlaps(x, y, z, rw, rh, rd, placed):
                    continue
                sc = score_position(x, y, z, rw, rh, rd, W, H, D)
                if sc < best_score:
                    best_score = sc
                    best = (x, y, z, rw, rh, rd)
                if deadline_check():
                    return best
            if deadline_check():
                return best
        return best

    def sort_by_volume(box_list):
        return sorted(box_list, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))

    def sort_by_max_dim(box_list):
        return sorted(box_list, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_by_surface(box_list):
        return sorted(box_list, key=lambda t: (-2*(t[1]*t[2] + t[2]*t[3] + t[1]*t[3]), -t[1]*t[2]*t[3]))

    def sort_by_min_dim(box_list):
        return sorted(box_list, key=lambda t: (-min(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_by_base_area(box_list):
        return sorted(box_list, key=lambda t: (-t[1]*t[2], -t[1]*t[2]*t[3]))

    def pack_with_order(order, boxes, W, H, D, deadline_check):
        placed = []
        result = [None] * len(boxes)
        for idx in order:
            w, h, d = boxes[idx]
            if deadline_check():
                break
            rots = box_rots[idx]
            pos = try_place_box(w, h, d, rots, placed, W, H, D, deadline_check)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                mark_occupied(x, y, z, rw, rh, rd)
                result[idx] = (0, x, y, z, rw, rh, rd)
        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        # Clear grid
        if use_grid:
            grid[:] = False
        return result

    def volume_filled(placements):
        total = 0
        for (b, x, y, z, w, h, d) in placements:
            if b == 0:
                total += w * h * d
        return total

    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    sort_strategies = [
        sort_by_volume,
        sort_by_max_dim,
        sort_by_surface,
        sort_by_min_dim,
        sort_by_base_area,
    ]

    best_result = None
    best_vol = 0

    for sort_fn in sort_strategies:
        if check_deadline():
            break
        order = [t[0] for t in sort_fn(indexed_boxes)]
        result = pack_with_order(order, boxes, W, H, D, check_deadline)
        vol = volume_filled(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result

    if check_deadline():
        return best_result if best_result else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Random restarts with diverse perturbations
    rng = random.Random(42)
    max_restarts = 500  # More restarts enabled by grid acceleration

    for restart in range(max_restarts):
        if check_deadline():
            break

        shuffled = list(indexed_boxes)
        # Base sort by volume
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])

        # Different perturbation strategies
        if restart % 3 == 0:
            # Adjacent swaps
            num_swaps = len(shuffled) // 3
            for _ in range(num_swaps):
                i = rng.randint(0, len(shuffled) - 2)
                shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        elif restart % 3 == 1:
            # Random transpositions
            num_swaps = len(shuffled) // 4
            for _ in range(num_swaps):
                i = rng.randint(0, len(shuffled) - 1)
                j = rng.randint(0, len(shuffled) - 1)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        else:
            # Partial shuffle of bottom half
            mid = len(shuffled) // 2
            bottom = shuffled[mid:]
            rng.shuffle(bottom)
            shuffled[mid:] = bottom

        order_r = [t[0] for t in shuffled]
        result_r = pack_with_order(order_r, boxes, W, H, D, check_deadline)
        vol_r = volume_filled(result_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = result_r

    return best_result if best_result else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
# EVOLVE-BLOCK-END
