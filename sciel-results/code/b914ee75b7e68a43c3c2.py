"""
3D Container Loading - Structural change: skyline representation + best-fit decreasing.

Instead of enumerating all corner candidates (O(n^3) candidates per box), we maintain
skyline arrays per (x,z) pair. This reduces candidate positions dramatically and
allows faster placement checks. We also add a local-search phase that attempts to
reposition unplaced boxes into gaps.
"""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def load(boxes, container):
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin vs 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    container_volume = W * H * D

    # Precompute orientations for each box
    box_data = []
    for idx, (w, h, d) in enumerate(boxes):
        seen = set()
        orients = []
        for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)

        # Filter to only orientations that fit in container
        valid_orients = [(bw, bh, bd) for bw, bh, bd in orients if bw <= W and bh <= H and bd <= D]
        if not valid_orients:
            valid_orients = [(w, h, d)]

        # Sort orientations: prefer those with smaller footprint area (bw*bd), then height
        valid_orients.sort(key=lambda o: (o[0] * o[2], o[1], o[0], o[2]))

        volume = w * h * d
        box_data.append((idx, valid_orients, volume, w, h, d))

    # Multiple sorting strategies
    box_by_vol = sorted(box_data, key=lambda x: x[2], reverse=True)
    box_by_maxdim = sorted(box_data, key=lambda x: max(x[3], x[4], x[5]), reverse=True)
    box_by_ratio = sorted(box_data, key=lambda x: x[3] * x[4] * x[5] / max(x[3], x[4], x[5], 1), reverse=True)

    def skyline_place(box_order, W, H, D, deadline, candidate_strategy='best'):
        """
        Place boxes using skyline representation.
        
        Skyline: dict mapping (x, z) -> height (top of stack at that column).
        Candidate positions: for each placed box, consider (px, py_top, pz) and (px, py, pz_top).
        We use a set of candidate (x, y, z) positions.
        """
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * len(box_order)
        total_placed_volume = 0

        # Skyline: set of (x, z) columns with their max height
        # We'll maintain a dict: (x, z) -> max_y_top
        skyline = {}

        def update_skyline(px, py, pz, pw, ph, pd):
            """Update skyline after placing a box."""
            # For all (x, z) in the footprint of this box
            for cx in range(px, px + pw, 1):
                for cz in range(pz, pz + pd, 1):
                    key = (cx, cz)
                    new_top = py + ph
                    if key not in skyline or skyline[key] < new_top:
                        skyline[key] = new_top

        def get_candidates():
            """Generate candidate positions from skyline and placed boxes."""
            cand_set = set()
            cand_set.add((0, 0, 0))

            # From skyline: for each (x, z) column, the top is a candidate y
            for (cx, cz), top_y in skyline.items():
                if top_y <= H:
                    cand_set.add((cx, top_y, cz))

            # From placed boxes: right side, top, front
            for (px, py, pz, pw, ph, pd) in placed:
                # Right
                if px + pw < W:
                    cand_set.add((px + pw, py, pz))
                # Top
                if py + ph < H:
                    cand_set.add((px, py + ph, pz))
                # Front (z direction)
                if pz + pd < D:
                    cand_set.add((px, py, pz + pd))
                # Also combinations
                if px + pw < W and py + ph < H:
                    cand_set.add((px + pw, py + ph, pz))
                if px + pw < W and pz + pd < D:
                    cand_set.add((px + pw, py, pz + pd))
                if py + ph < H and pz + pd < D:
                    cand_set.add((px, py + ph, pz + pd))

            # Sort candidates by priority: lower z, then lower y, then lower x (pack from back-bottom)
            cand_list = list(cand_set)
            cand_list.sort(key=lambda c: (c[2], c[1], c[0]))
            return cand_list

        def check_fit(cx, cy, cz, bw, bh, bd):
            """Check if a box fits at (cx, cy, cz) with dimensions (bw, bh, bd)."""
            # Bounds check
            if cx + bw > W or cy + bh > H or cz + bd > D:
                return False

            # Overlap check with all placed boxes
            for (px, py, pz, pw, ph, pd) in placed:
                # No overlap if separated in any dimension
                if (cx + bw <= px) or (px + pw <= cx):
                    continue
                if (cy + bh <= py) or (py + ph <= cy):
                    continue
                if (cz + bd <= pz) or (pz + pd <= cz):
                    continue
                return False
            return True

        for i, (idx, orients, volume, ow, oh, od) in enumerate(box_order):
            if time.time() > deadline:
                break

            found = False
            candidates = get_candidates()

            for (cx, cy, cz) in candidates:
                for (bw, bh, bd) in orients:
                    if check_fit(cx, cy, cz, bw, bh, bd):
                        # Place it
                        placed.append((cx, cy, cz, bw, bh, bd))
                        update_skyline(cx, cy, cz, bw, bh, bd)
                        results[idx] = (0, cx, cy, cz, bw, bh, bd)
                        total_placed_volume += volume
                        found = True
                        break
                if found:
                    break

            if not found:
                results[idx] = (-1, 0, 0, 0, ow, oh, od)

        # Fill remaining
        for i in range(len(box_order)):
            if results[i] is None:
                w_i, h_i, d_i = box_order[i][3], box_order[i][4], box_order[i][5]
                results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)

        return results, total_placed_volume, placed

    # Run multiple strategies
    best_results = None
    best_vol = -1

    strategies = [
        ("vol_zyx", box_by_vol),
        ("maxdim_zyx", box_by_maxdim),
        ("ratio_zyx", box_by_ratio),
    ]

    for name, box_order in strategies:
        if time.time() > deadline:
            break
        res, vol, placed = skyline_place(box_order, W, H, D, deadline, 'best')
        if vol > best_vol:
            best_vol = vol
            best_results = res

    # Local search: try to place unplaced boxes by rearranging
    if best_results is not None and time.time() < deadline - 20:
        # Identify unplaced boxes
        unplaced_indices = [i for i, r in enumerate(best_results) if r[0] == -1]
        if unplaced_indices:
            # For each unplaced box, try all positions and orientations
            # We need to temporarily remove some placed boxes to make room
            # Simpler approach: try to place unplaced boxes in current gaps
            placed_boxes = [(i, best_results[i]) for i in range(len(best_results)) if best_results[i][0] == 0]
            
            # Build occupied grid for fast overlap check
            # Use a set of occupied cells at unit resolution if container is small enough
            if W * H * D <= 100000:
                occ = np.zeros((W, H, D), dtype=bool)
                for (i, r) in placed_boxes:
                    _, x, y, z, w, h, d = r
                    occ[x:x+w, y:y+h, z:z+d] = True
                
                # Try to place unplaced boxes
                for ui in unplaced_indices:
                    if time.time() > deadline:
                        break
                    idx, orients, volume, ow, oh, od = box_data[ui]
                    for (bw, bh, bd) in orients:
                        placed_here = False
                        for cx in range(0, W - bw + 1):
                            if placed_here:
                                break
                            for cy in range(0, H - bh + 1):
                                if placed_here:
                                    break
                                for cz in range(0, D - bd + 1):
                                    if not occ[cx:cx+bw, cy:cy+bh, cz:cz+bd].any():
                                        occ[cx:cx+bw, cy:cy+bh, cz:cz+bd] = True
                                        best_results[ui] = (0, cx, cy, cz, bw, bh, bd)
                                        best_vol += volume
                                        placed_here = True
                                        break
            else:
                # For large containers, use overlap check with placed list
                placed_list = [(r[1], r[2], r[3], r[4], r[5], r[6]) for r in best_results if r[0] == 0]
                
                for ui in unplaced_indices:
                    if time.time() > deadline:
                        break
                    idx, orients, volume, ow, oh, od = box_data[ui]
                    # Generate candidate positions
                    cand_set = {(0, 0, 0)}
                    for (px, py, pz, pw, ph, pd) in placed_list:
                        if px + pw < W:
                            cand_set.add((px + pw, py, pz))
                        if py + ph < H:
                            cand_set.add((px, py + ph, pz))
                        if pz + pd < D:
                            cand_set.add((px, py, pz + pd))
                    
                    for (cx, cy, cz) in sorted(cand_set, key=lambda c: (c[2], c[1], c[0])):
                        found_pos = False
                        for (bw, bh, bd) in orients:
                            if cx + bw > W or cy + bh > H or cz + bd > D:
                                continue
                            ok = True
                            for (px, py, pz, pw, ph, pd) in placed_list:
                                if not ((cx + bw <= px) or (px + pw <= cx) or
                                        (cy + bh <= py) or (py + ph <= cy) or
                                        (cz + bd <= pz) or (pz + pd <= cz)):
                                    ok = False
                                    break
                            if ok:
                                placed_list.append((cx, cy, cz, bw, bh, bd))
                                best_results[ui] = (0, cx, cy, cz, bw, bh, bd)
                                best_vol += volume
                                found_pos = True
                                break
                        if found_pos:
                            break

    return best_results

# EVOLVE-BLOCK-END
