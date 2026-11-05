"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    container_volume = W * H * D
    n_boxes = len(boxes)

    # Precompute orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        dims = sorted([w, h, d], reverse=True)
        seen = set()
        orientations = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orientations.append(perm)
        box_orientations.append(orientations)

    def get_orientations(w, h, d):
        """Get unique orientations of a box."""
        dims = sorted([w, h, d], reverse=True)
        seen = set()
        orientations = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orientations.append(perm)
        return orientations

    def check_overlap(x, y, z, w, h, d, placed):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any placed box."""
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    def find_placement(x, y, z, w, h, d, placed, W, H, D):
        """
        Find a valid placement for a box using a sliding window approach.
        Returns (bx, by, bz, bw, bh, bd) or None.
        """
        orientations = get_orientations(w, h, d)
        best_pos = None
        best_score = float('inf')

        for ow, oh, od in orientations:
            if x + ow > W or y + oh > H or z + od > D:
                continue

            # Try to find a non-overlapping position near (x, y, z)
            # Use a grid of candidate points around the desired location
            # First try the exact position
            if not check_overlap(x, y, z, ow, oh, od, placed):
                score = x / W + y / H + z / D
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z, ow, oh, od)
                continue

            # Try sliding the box in each direction
            # Slide along x
            for dx in range(0, W - x + 1, max(1, int(ow / 4))):
                nx = x + dx
                if nx + ow > W:
                    break
                if not check_overlap(nx, y, z, ow, oh, od, placed):
                    score = nx / W + y / H + z / D
                    if score < best_score:
                        best_score = score
                        best_pos = (nx, y, z, ow, oh, od)
                    break

            # Slide along y
            for dy in range(0, H - y + 1, max(1, int(oh / 4))):
                ny = y + dy
                if ny + oh > H:
                    break
                if not check_overlap(x, ny, z, ow, oh, od, placed):
                    score = x / W + ny / H + z / D
                    if score < best_score:
                        best_score = score
                        best_pos = (x, ny, z, ow, oh, od)
                    break

            # Slide along z
            for dz in range(0, D - z + 1, max(1, int(od / 4))):
                nz = z + dz
                if nz + od > D:
                    break
                if not check_overlap(x, y, nz, ow, oh, od, placed):
                    score = x / W + y / H + nz / D
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, nz, ow, oh, od)
                    break

        return best_pos

    def run_sliding_window_strategy(order_key_func, use_reinsert=True, reinsert_iters=200):
        """
        Sliding window strategy: place boxes in order, but use a sliding window
        to find valid positions. Then try to re-insert left-out boxes.
        """
        if time.time() > deadline:
            return 0, None

        indexed_boxes = list(enumerate(boxes))
        try:
            indexed_boxes.sort(key=order_key_func)
        except Exception:
            return 0, None

        placed = []  # List of (x, y, z, w, h, d)
        results = [None] * n_boxes
        total_vol = 0

        # Candidate points: start with origin and some spread points
        candidate_points = [(0, 0, 0)]
        # Add some spread points to explore the space
        for i in range(1, 5):
            for j in range(1, 5):
                for k in range(1, 5):
                    candidate_points.append((int(W * i / 5), int(H * j / 5), int(D * k / 5)))

        for orig_idx, (w, h, d) in indexed_boxes:
            if time.time() > deadline:
                break

            found_placement = False

            # Try each candidate point
            for cx, cy, cz in candidate_points:
                if time.time() > deadline:
                    break

                result = find_placement(cx, cy, cz, w, h, d, placed, W, H, D)
                if result is not None:
                    bx, by, bz, bw, bh, bd = result
                    placed.append((bx, by, bz, bw, bh, bd))
                    results[orig_idx] = (0, bx, by, bz, bw, bh, bd)
                    total_vol += bw * bh * bd
                    found_placement = True

                    # Generate new candidate points from this placement
                    new_candidates = [
                        (bx + bw, by, bz),
                        (bx, by + bh, bz),
                        (bx, by, bz + bd),
                        (bx + bw, by + bh, bz),
                        (bx + bw, by, bz + bd),
                        (bx, by + bh, bz + bd),
                        (bx + bw, by + bh, bz + bd),
                    ]
                    for nc in new_candidates:
                        if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                            candidate_points.append(nc)
                    break

            if not found_placement:
                results[orig_idx] = (-1, 0, 0, 0, w, h, d)

        # Re-insertion: try to insert left-out boxes using sliding window
        if use_reinsert and time.time() < deadline - 5:
            for _ in range(reinsert_iters):
                if time.time() > deadline - 2:
                    break

                inserted = False
                for i in range(n_boxes):
                    if results[i][0] == -1:
                        w, h, d = boxes[i]
                        # Try to find a placement for this box
                        for cx, cy, cz in candidate_points:
                            result = find_placement(cx, cy, cz, w, h, d, placed, W, H, D)
                            if result is not None:
                                bx, by, bz, bw, bh, bd = result
                                placed.append((bx, by, bz, bw, bh, bd))
                                results[i] = (0, bx, by, bz, bw, bh, bd)
                                total_vol += bw * bh * bd
                                inserted = True
                                break
                        if inserted:
                            break

                if not inserted:
                    break

        # Fill any remaining results
        for i in range(n_boxes):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)

        return total_vol, results

    def run_multi_start_randomized(order_key_func, n_restarts=3, use_reinsert=True):
        """
        Multi-start randomized strategy: run the sliding window strategy multiple
        times with randomized perturbations, keep the best result.
        """
        best_vol = 0
        best_results = None

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Add random perturbation to the ordering
            indexed_boxes = list(enumerate(boxes))
            try:
                indexed_boxes.sort(key=order_key_func)
            except Exception:
                continue

            # Randomly swap some adjacent elements to create variation
            if restart > 0:
                for _ in range(random.randint(1, 5)):
                    i = random.randint(0, len(indexed_boxes) - 2)
                    indexed_boxes[i], indexed_boxes[i + 1] = indexed_boxes[i + 1], indexed_boxes[i]

            # Run the strategy with this ordering
            placed = []
            results = [None] * n_boxes
            total_vol = 0

            candidate_points = [(0, 0, 0)]
            for i in range(1, 5):
                for j in range(1, 5):
                    for k in range(1, 5):
                        candidate_points.append((int(W * i / 5), int(H * j / 5), int(D * k / 5)))

            for orig_idx, (w, h, d) in indexed_boxes:
                if time.time() > deadline:
                    break

                found_placement = False
                for cx, cy, cz in candidate_points:
                    if time.time() > deadline:
                        break

                    result = find_placement(cx, cy, cz, w, h, d, placed, W, H, D)
                    if result is not None:
                        bx, by, bz, bw, bh, bd = result
                        placed.append((bx, by, bz, bw, bh, bd))
                        results[orig_idx] = (0, bx, by, bz, bw, bh, bd)
                        total_vol += bw * bh * bd
                        found_placement = True

                        new_candidates = [
                            (bx + bw, by, bz),
                            (bx, by + bh, bz),
                            (bx, by, bz + bd),
                            (bx + bw, by + bh, bz),
                            (bx + bw, by, bz + bd),
                            (bx, by + bh, bz + bd),
                            (bx + bw, by + bh, bz + bd),
                        ]
                        for nc in new_candidates:
                            if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                                candidate_points.append(nc)
                        break

                if not found_placement:
                    results[orig_idx] = (-1, 0, 0, 0, w, h, d)

            # Re-insertion
            if use_reinsert and time.time() < deadline - 5:
                for _ in range(100):
                    if time.time() > deadline - 2:
                        break
                    inserted = False
                    for i in range(n_boxes):
                        if results[i][0] == -1:
                            w, h, d = boxes[i]
                            for cx, cy, cz in candidate_points:
                                result = find_placement(cx, cy, cz, w, h, d, placed, W, H, D)
                                if result is not None:
                                    bx, by, bz, bw, bh, bd = result
                                    placed.append((bx, by, bz, bw, bh, bd))
                                    results[i] = (0, bx, by, bz, bw, bh, bd)
                                    total_vol += bw * bh * bd
                                    inserted = True
                                    break
                            if inserted:
                                break
                    if not inserted:
                        break

            for i in range(n_boxes):
                if results[i] is None:
                    w, h, d = boxes[i]
                    results[i] = (-1, 0, 0, 0, w, h, d)

            if total_vol > best_vol:
                best_vol = total_vol
                best_results = results

        return best_vol, best_results

    # Define sorting strategies
    def sort_vol_desc(x):
        w, h, d = x[1]
        return -(w * h * d)

    def sort_height_desc(x):
        w, h, d = x[1]
        return -max(w, h, d)

    def sort_sum_desc(x):
        w, h, d = x[1]
        return -(w + h + d)

    def sort_aspect(x):
        w, h, d = x[1]
        vol = w * h * d
        if vol == 0:
            return 0
        s = (w + h + d) / 3.0
        dev = abs(w - s) + abs(h - s) + abs(d - s)
        return (-vol, dev)

    def sort_vol_height(x):
        w, h, d = x[1]
        return (-(w * h * d), -max(w, h, d))

    best_vol = 0
    best_results = None

    # Run sliding window strategies
    strategies = [
        sort_vol_desc,
        sort_height_desc,
        sort_sum_desc,
        sort_aspect,
        sort_vol_height,
    ]

    for strat in strategies:
        if time.time() > deadline:
            break

        vol, res = run_sliding_window_strategy(strat)
        if res is not None and vol > best_vol:
            best_vol = vol
            best_results = res

    # Run multi-start randomized strategies
    if time.time() < deadline - 10:
        for strat in [sort_vol_desc, sort_height_desc]:
            if time.time() > deadline:
                break
            vol, res = run_multi_start_randomized(strat, n_restarts=2)
            if res is not None and vol > best_vol:
                best_vol = vol
                best_results = res

    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_results
# EVOLVE-BLOCK-END
