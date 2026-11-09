"""3D Bin Packing: Best-Fit Decreasing with Local Search Refinement."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    container_vol = W * H * D

    # Sort boxes by volume descending, with tie-breaking by max dimension descending
    # This helps pack larger boxes first when space is most flexible
    indexed_boxes = [(w * h * d, max(w, h, d), i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[0], -x[1]))
    # Strip the tie-breaking key for downstream use
    indexed_boxes = [(vol, i, w, h, d) for (vol, _, i, w, h, d) in indexed_boxes]

    # Bins: list of lists of (x, y, z, w, h, d, orig_idx)
    bins = []
    placement_map = [None] * n  # (bin_idx, x, y, z)

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)

    def find_best_position(bin_boxes, w, h, d):
        """
        Find the best valid position for a box in a bin.
        Heuristic: Minimize (z + y + x) first? Or minimize distance to far walls?
        Let's use a composite score: prioritize low z, then low y, then low x.
        Actually, a better heuristic for packing is to minimize the "waste" or maximize support.
        Let's stick to a strong geometric heuristic:
        1. Generate candidates from corners of existing boxes and (0,0,0).
        2. Filter valid non-overlapping positions.
        3. Score each position: 
           - Primary: Minimize z (stack as low as possible)
           - Secondary: Minimize y
           - Tertiary: Minimize x
           This is effectively "First Fit" in z, then y, then x.
           
           Alternative: Minimize (W - (x+w)) + (H - (y+h)) + (D - (z+d)) 
           This pushes boxes to the origin.
           
           Let's try: Minimize (z, y, x) is standard and works well.
        """
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate positions
        candidates = set()
        candidates.add((0, 0, 0))

        # Add corners of existing boxes
        for (px, py, pz, pw, ph, pd, _) in bin_boxes:
            # Right face
            if px + pw <= W:
                candidates.add((px + pw, py, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # Top face
            if py + ph <= H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # Front face
            if pz + pd <= D:
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            
            # Also add the box's own corners just in case (though they are usually occupied or invalid for new box unless touching)
            # Actually, just the "outer" corners of the union of boxes are usually sufficient.
            # The above adds the "positive" corners of each box.

        # Limit candidates if too many? 40 boxes * 8 corners = 320 candidates. Manageable.
        
        best_pos = None
        best_score = None

        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            # Check overlap
            valid = True
            for (px, py, pz, pw, ph, pd, _) in bin_boxes:
                if overlaps(cx, cy, cz, w, h, d, px, py, pz, pw, ph, pd):
                    valid = False
                    break
            
            if valid:
                # Score: Minimize z, then y, then x
                # This packs tightly to the origin
                score = (cz, cy, cx)
                
                # Alternative Score: Minimize distance to far walls (push to origin)
                # score = (W - (cx + w)) + (H - (cy + h)) + (D - (cz + d))
                # This is equivalent to maximizing (cx+cy+cz) roughly? No.
                # Let's stick to (cz, cy, cx) as it is stable and effective.
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)

        return best_pos

    # --- Phase 1: Initial Placement (Best-Fit Decreasing) ---
    # For each box, try all bins and pick the one that minimizes wasted space
    # (i.e., the position with smallest z, then y, then x across all bins)
    for vol, orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place in new bin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)
            continue

        placed = False
        best_bin = -1
        best_pos = None
        best_score = None

        # Try existing bins with best-fit: pick bin+position with min (z, y, x)
        for b_idx in range(len(bins)):
            if time.time() > deadline:
                break
            pos = find_best_position(bins[b_idx], w, h, d)
            if pos is not None:
                score = (pos[2], pos[1], pos[0])  # (z, y, x)
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos

        if best_bin >= 0:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, orig_idx))
            placement_map[orig_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2])
            placed = True

        if not placed:
            # Open new bin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)

    # --- Phase 2: Local Search Refinement ---
    # Goal: Reduce number of bins (B) by moving boxes from later bins to earlier ones.
    # Strategy: Repeatedly try to move boxes from the last occupied bin to earlier bins.
    # If the last bin becomes empty, remove it and repeat.

    def try_move_box(bin_from, box_idx_in_bin, bin_to):
        """Try to move a specific box from bin_from to bin_to."""
        if bin_from == bin_to or bin_from < 0 or bin_to < 0:
            return False
        if box_idx_in_bin < 0 or box_idx_in_bin >= len(bins[bin_from]):
            return False
            
        box = bins[bin_from][box_idx_in_bin]
        bx, by, bz, bw, bh, bd, bidx = box

        # Remove from source
        bins[bin_from].pop(box_idx_in_bin)

        # Try to place in target
        pos = find_best_position(bins[bin_to], bw, bh, bd)
        if pos is not None:
            bins[bin_to].append((pos[0], pos[1], pos[2], bw, bh, bd, bidx))
            placement_map[bidx] = (bin_to, pos[0], pos[1], pos[2])
            return True
        else:
            # Put back
            bins[bin_from].insert(box_idx_in_bin, box)
            return False

    # Aggressive consolidation: repeatedly try to empty the last bin
    max_rounds = 50
    for _ in range(max_rounds):
        if time.time() > deadline - 10:
            break
        if len(bins) <= 1:
            break

        # Clean up any empty bins first
        new_bins = []
        for b in bins:
            if len(b) > 0:
                new_bins.append(b)
        if len(new_bins) != len(bins):
            bins.clear()
            bins.extend(new_bins)
            # Rebuild placement_map with correct indices
            for i in range(n):
                old_b, x, y, z = placement_map[i]
                # Find new index
                new_idx = -1
                for nb in bins:
                    for (bx, by, bz, bw, bh, bd, bidx) in nb:
                        if bidx == i:
                            new_idx = bins.index(nb)
                            break
                    if new_idx >= 0:
                        break
                if new_idx >= 0:
                    placement_map[i] = (new_idx, x, y, z)
            continue

        last_bin_idx = len(bins) - 1
        if len(bins[last_bin_idx]) == 0:
            bins.pop()
            continue

        moved_any_this_round = False
        # Try to move each box from the last bin to any earlier bin
        # Process boxes in order of increasing volume (smaller boxes are easier to fit)
        box_order = list(range(len(bins[last_bin_idx])))
        box_order.sort(key=lambda idx: bins[last_bin_idx][idx][3] * bins[last_bin_idx][idx][4] * bins[last_bin_idx][idx][5])

        for box_idx in box_order:
            if time.time() > deadline - 10:
                break
            if box_idx >= len(bins[last_bin_idx]):
                continue
            # Try all earlier bins
            for target_bin in range(last_bin_idx):
                if try_move_box(last_bin_idx, box_idx, target_bin):
                    moved_any_this_round = True
                    break
            if moved_any_this_round:
                # After a successful move, the box indices in last bin may have shifted
                # So restart the box iteration
                break

        if not moved_any_this_round:
            # No more boxes can be moved from the last bin
            break

    # Final cleanup: Remove any empty bins from the end
    while len(bins) > 0 and len(bins[-1]) == 0:
        bins.pop()

    # Build result
    placements = []
    for i in range(n):
        w, h, d = boxes[i]
        b_idx, x, y, z = placement_map[i]
        placements.append((b_idx, x, y, z, w, h, d))

    return placements
# EVOLVE-BLOCK-END
