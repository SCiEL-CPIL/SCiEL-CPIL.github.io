"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    n = len(boxes)

    # Precompute all valid orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        orientations = set()
        for perm in itertools.permutations((w, h, d)):
            orientations.add(perm)
        box_orientations.append(list(orientations))

    # Score boxes by volume (larger first often helps packing)
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]))

    def try_placement(box_order, orientation_strategy='best_fit'):
        """
        Try to place boxes in the given order using shelf-based packing.
        orientation_strategy: 'first', 'best_fit', or 'random'
        """
        placements = [None] * n
        placed_count = 0
        total_vol = 0

        placed_boxes = []  # list of (x, y, z, w, h, d)

        def can_place(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Check overlap with all placed boxes
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # Check if boxes overlap
                if (x < px + pw and x + w > px and
                    y < py + ph and y + h > py and
                    z < pz + pd and z + d > pz):
                    return False
            return True

        def find_position(w, h, d):
            """Find a valid position for a box with dimensions w, h, d."""
            # Try to place at the "lowest, leftmost, backmost" position
            # We'll scan a coarse grid for efficiency
            best_pos = None
            best_score = None

            # Generate candidate positions
            # For each placed box, we can place our box adjacent to it in x, y, or z
            # Also try (0,0,0)
            candidates = [(0, 0, 0)]

            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # Place to the right
                candidates.append((px + pw, py, pz))
                # Place in front
                candidates.append((px, py + ph, pz))
                # Place above
                candidates.append((px, py, pz + pd))
                # Also try combinations of two offsets
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py + ph, pz + pd))

            # Deduplicate and filter
            seen = set()
            valid_candidates = []
            for (cx, cy, cz) in candidates:
                if (cx, cy, cz) in seen:
                    continue
                seen.add((cx, cy, cz))
                if can_place(cx, cy, cz, w, h, d):
                    valid_candidates.append((cx, cy, cz))

            if not valid_candidates:
                return None

            # Score: prefer lower z, then lower y, then lower x (compact packing)
            valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
            return valid_candidates[0]

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]

            if orientation_strategy == 'best_fit':
                # Try all orientations, pick the one that allows placement with lowest position
                best_orientation = None
                best_pos = None
                best_pos_score = None
                for (ow, oh, od) in box_orientations[idx]:
                    pos = find_position(ow, oh, od)
                    if pos is not None:
                        # Score the position
                        pos_score = (pos[2], pos[1], pos[0])
                        if best_pos_score is None or pos_score < best_pos_score:
                            best_pos_score = pos_score
                            best_pos = pos
                            best_orientation = (ow, oh, od)
                if best_pos is not None:
                    ow, oh, od = best_orientation
                    placed_boxes.append((best_pos[0], best_pos[1], best_pos[2], ow, oh, od))
                    placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                    placed_count += 1
                    total_vol += ow * oh * od
            else:
                # Use first orientation
                ow, oh, od = box_orientations[idx][0]
                pos = find_position(ow, oh, od)
                if pos is not None:
                    placed_boxes.append((pos[0], pos[1], pos[2], ow, oh, od))
                    placements[idx] = (0, pos[0], pos[1], pos[2], ow, oh, od)
                    placed_count += 1
                    total_vol += ow * oh * od

        # Fill in unplaced boxes
        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    def shelf_packing(box_order, use_rotation=True):
        """
        Shelf-based packing: boxes are placed on horizontal shelves.
        Each shelf has a fixed height (max height of boxes on it) and depth.
        Boxes are placed side-by-side in x, with y=0 for all boxes on a shelf.
        New shelves are stacked in z.
        """
        placements = [None] * n
        placed_count = 0
        total_vol = 0

        # Each shelf: dict with z (base z), h (height), x (current x cursor), boxes (list of placed box indices)
        shelves = []  # list of shelf dicts

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]

            if not use_rotation:
                orientations = [(w_orig, h_orig, d_orig)]
            else:
                orientations = box_orientations[idx]

            best_shelf_idx = None
            best_pos = None
            best_orientation = None
            best_score = None

            for (ow, oh, od) in orientations:
                # Try to place on an existing shelf
                for s_idx, s in enumerate(shelves):
                    # Shelf height must accommodate this box
                    if oh <= s['h'] and s['x'] + ow <= W and od <= D:
                        pos = (s['x'], 0, s['z'])
                        # Score: prefer lower z, then lower x
                        score = (s['z'], s['x'])
                        if best_score is None or score < best_score:
                            best_score = score
                            best_shelf_idx = s_idx
                            best_pos = pos
                            best_orientation = (ow, oh, od)

                # Try to create a new shelf
                if oh <= H:
                    # Find the minimum z where a new shelf of height oh can fit
                    min_z = 0
                    for s in shelves:
                        if s['z'] + s['h'] > min_z:
                            min_z = s['z'] + s['h']

                    if min_z + oh <= H:
                        pos = (0, 0, min_z)
                        score = (min_z, 0)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_shelf_idx = -1  # signal to create new shelf
                            best_pos = pos
                            best_orientation = (ow, oh, od)

            if best_shelf_idx is not None and best_pos is not None:
                ow, oh, od = best_orientation
                if best_shelf_idx >= 0:
                    # Place on existing shelf
                    shelf = shelves[best_shelf_idx]
                    shelf['x'] += ow
                else:
                    # Create new shelf
                    new_shelf = {'z': best_pos[2], 'h': oh, 'x': ow, 'boxes': []}
                    shelves.append(new_shelf)

                placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                placed_count += 1
                total_vol += ow * oh * od

        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    def layer_packing(box_order, use_rotation=True):
        """
        Layer-based packing similar to the inspiration:
        - Process boxes in order.
        - Maintain current x, y, z, row_h, layer_d.
        - If box doesn't fit in current row (x + w > W), start new row (y += row_h, x = 0, row_h = 0).
        - If box doesn't fit in current layer (y + h > H), start new layer (z += layer_d, y = 0, x = 0, row_h = 0, layer_d = 0).
        - If box doesn't fit in container (z + d > D), skip.
        - With rotation, try all orientations and pick the one that fits best (lowest z, then y, then x).
        """
        placements = [None] * n
        placed_count = 0
        total_vol = 0

        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]

            if not use_rotation:
                orientations = [(w_orig, h_orig, d_orig)]
            else:
                orientations = box_orientations[idx]

            best_orientation = None
            best_x, best_y, best_z = None, None, None
            best_row_h, best_layer_d = None, None
            best_score = None

            for (ow, oh, od) in orientations:
                # Simulate placement
                sx, sy, sz = x, y, z
                srow_h, slayer_d = row_h, layer_d

                if sx + ow > W:
                    sx, sy, srow_h = 0, sy + srow_h, 0
                if sy + oh > H:
                    sy, sz, slayer_d, sx, srow_h = 0, sz + slayer_d, 0, 0, 0
                if sz + od > D:
                    continue  # Doesn't fit

                # This orientation fits. Score it.
                score = (sz, sy, sx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_orientation = (ow, oh, od)
                    best_x, best_y, best_z = sx, sy, sz
                    best_row_h, best_layer_d = srow_h, slayer_d

            if best_orientation is not None:
                ow, oh, od = best_orientation
                placements[idx] = (0, best_x, best_y, best_z, ow, oh, od)
                placed_count += 1
                total_vol += ow * oh * od

                # Update state
                x = best_x + ow
                y = best_y
                z = best_z
                row_h = max(best_row_h, oh)
                layer_d = max(best_layer_d, od)

        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    # Try multiple strategies and pick the best
    best_placements = None
    best_vol = 0

    # Strategy 1: Sort by volume descending, best_fit orientation (free placement)
    result1 = try_placement(box_indices, 'best_fit')
    if result1:
        best_placements, best_vol = result1

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 2: Sort by max dimension descending, best_fit
    box_indices2 = sorted(range(n), key=lambda i: -max(boxes[i]))
    result2 = try_placement(box_indices2, 'best_fit')
    if result2:
        _, vol2 = result2
        if vol2 > best_vol:
            best_vol = vol2
            best_placements = result2[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 3: Sort by min dimension ascending (thin boxes first), best_fit
    box_indices3 = sorted(range(n), key=lambda i: min(boxes[i]))
    result3 = try_placement(box_indices3, 'best_fit')
    if result3:
        _, vol3 = result3
        if vol3 > best_vol:
            best_vol = vol3
            best_placements = result3[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 4: Shelf packing with volume sort
    result4 = shelf_packing(box_indices, True)
    if result4:
        _, vol4 = result4
        if vol4 > best_vol:
            best_vol = vol4
            best_placements = result4[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 5: Shelf packing with max dimension sort
    result5 = shelf_packing(box_indices2, True)
    if result5:
        _, vol5 = result5
        if vol5 > best_vol:
            best_vol = vol5
            best_placements = result5[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 6: Layer packing with volume sort (inspiration-based)
    result6 = layer_packing(box_indices, True)
    if result6:
        _, vol6 = result6
        if vol6 > best_vol:
            best_vol = vol6
            best_placements = result6[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 7: Layer packing with max dimension sort
    result7 = layer_packing(box_indices2, True)
    if result7:
        _, vol7 = result7
        if vol7 > best_vol:
            best_vol = vol7
            best_placements = result7[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 8: Layer packing with min dimension sort
    result8 = layer_packing(box_indices3, True)
    if result8:
        _, vol8 = result8
        if vol8 > best_vol:
            best_vol = vol8
            best_placements = result8[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    if best_placements is None:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    return best_placements

# EVOLVE-BLOCK-END
