"""3D Bin Packing: Best-Fit Decreasing with Enhanced Local Search Refinement."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    container_vol = W * H * D

    # Sort boxes by volume descending, with tie-breaking by max dimension descending
    indexed_boxes = [(w * h * d, max(w, h, d), i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[0], -x[1]))
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
        Uses skyline-based candidate generation for denser packing.
        """
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate positions using extreme points (skyline approach)
        candidates = set()
        candidates.add((0, 0, 0))

        # Build skyline by finding maximal points
        # A point (x, y, z) is an extreme point if no other box covers it from below
        # We generate candidates at the "corners" where new boxes can be placed
        
        for (px, py, pz, pw, ph, pd, _) in bin_boxes:
            # Right face - try all y, z positions along this face
            if px + pw <= W:
                # Bottom-right corner
                candidates.add((px + pw, py, pz))
                # Top-right corner  
                candidates.add((px + pw, py + ph, pz))
                # Back-right corner
                candidates.add((px + pw, py, pz + pd))
                # Back-top-right corner
                candidates.add((px + pw, py + ph, pz + pd))
            
            # Top face - try all x, z positions along this face
            if py + ph <= H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            
            # Front face - try all x, y positions along this face
            if pz + pd <= D:
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            
            # Also add positions that are "supported" - where the box would rest on top
            # of existing boxes. For each existing box, the position directly above it
            # (aligned in x,y) is a valid supported position if within bounds
            if py + ph + h <= H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))

        best_pos = None
        best_score = None

        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue

            # Check if position is valid (no overlap)
            valid = True
            for (px, py, pz, pw, ph, pd, _) in bin_boxes:
                if overlaps(cx, cy, cz, w, h, d, px, py, pz, pw, ph, pd):
                    valid = False
                    break

            if valid:
                # Check if position is "supported" - either on floor or on top of another box
                # A position is supported if there's no gap below it (in z direction)
                # For simplicity, we accept all valid positions but prefer lower z
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)

        return best_pos

    # --- Phase 1: Initial Placement (Best-Fit Decreasing) ---
    for vol, orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
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
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)

    # --- Phase 2: Local Search Refinement ---
    import random
    rng = random.Random(42)

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
            bins[bin_from].insert(box_idx_in_bin, box)
            return False

    def try_swap_boxes(bin1, box_idx1, bin2, box_idx2):
        """Try to swap two boxes between bins."""
        if bin1 == bin2:
            return False
        if box_idx1 < 0 or box_idx1 >= len(bins[bin1]):
            return False
        if box_idx2 < 0 or box_idx2 >= len(bins[bin2]):
            return False

        box1 = bins[bin1][box_idx1]
        box2 = bins[bin2][box_idx2]

        # Remove both
        bins[bin1].pop(box_idx1)
        bins[bin2].pop(box_idx2)

        # Try to place box1 in bin2 and box2 in bin1
        pos1 = find_best_position(bins[bin2], box1[3], box1[4], box1[5])
        pos2 = find_best_position(bins[bin1], box2[3], box2[4], box2[5])

        if pos1 is not None and pos2 is not None:
            bins[bin2].append((pos1[0], pos1[1], pos1[2], box1[3], box1[4], box1[5], box1[6]))
            bins[bin1].append((pos2[0], pos2[1], pos2[2], box2[3], box2[4], box2[5], box2[6]))
            placement_map[box1[6]] = (bin2, pos1[0], pos1[1], pos1[2])
            placement_map[box2[6]] = (bin1, pos2[0], pos2[1], pos2[2])
            return True
        else:
            # Put back
            bins[bin1].insert(box_idx1, box1)
            bins[bin2].insert(box_idx2, box2)
            return False

    def shake_bin(bin_idx, num_shakes=5):
        """
        Attempt to improve packing within a bin by repositioning boxes.
        This helps create more space by finding denser arrangements.
        """
        if bin_idx < 0 or bin_idx >= len(bins) or len(bins[bin_idx]) <= 1:
            return False

        bin_boxes = bins[bin_idx]
        if not bin_boxes:
            return False

        # Try to reposition each box to a better (lower) position
        improved = False
        for _ in range(num_shakes):
            if time.time() > deadline - 10:
                break

            # Pick a random box and try to move it to a lower/better position
            box_idx = rng.randint(0, len(bin_boxes) - 1)
            box = bin_boxes[box_idx]
            bx, by, bz, bw, bh, bd, bidx = box

            # Remove the box
            bin_boxes.pop(box_idx)

            # Find the best position for this box in the bin (without itself)
            pos = find_best_position(bin_boxes, bw, bh, bd)
            if pos is not None:
                # Check if this is a better position (lower z, y, x)
                if (pos[2], pos[1], pos[0]) < (bz, by, bx):
                    bin_boxes.append((pos[0], pos[1], pos[2], bw, bh, bd, bidx))
                    placement_map[bidx] = (bin_idx, pos[0], pos[1], pos[2])
                    improved = True
                else:
                    # Put it back in original position
                    bin_boxes.insert(box_idx, box)
            else:
                # Couldn't place it, put it back
                bin_boxes.insert(box_idx, box)

        return improved

    def consolidate_last_bin():
        """Try to empty the last bin by moving/swapping its boxes."""
        if len(bins) <= 1:
            return False

        last_bin_idx = len(bins) - 1
        if len(bins[last_bin_idx]) == 0:
            bins.pop()
            return True

        moved_any = False

        # Strategy 1: Try to move each box from the last bin to any earlier bin
        # Sort by volume ascending (try smaller boxes first, they're easier to fit)
        box_order = list(range(len(bins[last_bin_idx])))
        box_order.sort(key=lambda idx: bins[last_bin_idx][idx][3] * bins[last_bin_idx][idx][4] * bins[last_bin_idx][idx][5])

        for box_idx in box_order:
            if time.time() > deadline - 15:
                break
            if box_idx >= len(bins[last_bin_idx]):
                continue
            for target_bin in range(last_bin_idx):
                if try_move_box(last_bin_idx, box_idx, target_bin):
                    moved_any = True
                    break
            if moved_any:
                break

        if moved_any:
            return True

        # Strategy 2: Try swaps between last bin and other bins
        for box_idx in range(len(bins[last_bin_idx])):
            if time.time() > deadline - 15:
                break
            for other_bin in range(last_bin_idx):
                for other_box_idx in range(len(bins[other_bin])):
                    if try_swap_boxes(last_bin_idx, box_idx, other_bin, other_box_idx):
                        moved_any = True
                        break
                if moved_any:
                    break
            if moved_any:
                break

        # Strategy 3: Shake other bins to create more space, then retry moves
        if not moved_any and time.time() < deadline - 20:
            # Try shaking each non-last bin to create space
            for b_idx in range(last_bin_idx):
                if time.time() > deadline - 15:
                    break
                if shake_bin(b_idx, num_shakes=3):
                    # After shaking, retry moving boxes from last bin
                    for box_idx in range(len(bins[last_bin_idx])):
                        if time.time() > deadline - 15:
                            break
                        if box_idx >= len(bins[last_bin_idx]):
                            continue
                        for target_bin in range(last_bin_idx):
                            if try_move_box(last_bin_idx, box_idx, target_bin):
                                moved_any = True
                                break
                        if moved_any:
                            break
                    if moved_any:
                        break

        return moved_any

    # Aggressive consolidation: repeatedly try to empty the last bin
    max_rounds = 200
    for _ in range(max_rounds):
        if time.time() > deadline - 15:
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

        if not consolidate_last_bin():
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
