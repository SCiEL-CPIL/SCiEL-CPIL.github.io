"""3D Bin Packing: FDD + best-fit with skyline pruning and multi-pass refinement."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    deadline = time.time() + 110.0  # safety margin

    # Sort boxes by volume descending (FDD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)

    # Bins will store list of (x, y, z, w, h, d)
    bins = []
    placement_map = {}  # original index -> (bin_idx, x, y, z)

    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes greedily in new bins
            for remaining_orig_idx, (rw, rh, rd) in indexed_boxes:
                if remaining_orig_idx not in placement_map:
                    b_idx = len(bins)
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placement_map[remaining_orig_idx] = (b_idx, 0, 0, 0)
            break

        placed = False
        # Best-fit: try all existing bins and pick the one with the best position
        # Prefer bins where the box fits lowest (smallest z), then smallest y, then smallest x
        best_b_idx = -1
        best_pos = None
        best_score = None
        for b_idx, bin_contents in enumerate(bins):
            pos = _find_position(bin_contents, W, H, D, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score for bin selection: prefer lower z, then y, then x
                # Also consider how much the box "fills" the remaining space
                score = (z, y, x)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = pos
                    best_b_idx = b_idx

        if best_b_idx >= 0:
            x, y, z = best_pos
            placement_map[orig_idx] = (best_b_idx, x, y, z)
            bins[best_b_idx].append((x, y, z, w, h, d))
            placed = True

        if not placed:
            # Create new bin
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placement_map[orig_idx] = (b_idx, 0, 0, 0)

    # Refinement pass: try to improve by moving boxes to earlier bins or better positions
    if time.time() < deadline - 5.0:
        _refine(bins, placement_map, boxes, W, H, D, deadline)
    
    # Second refinement: try to repack the last bin more tightly
    if time.time() < deadline - 3.0 and len(bins) > 0:
        _tighten_last_bin(bins, placement_map, boxes, W, H, D, deadline)

    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement_map[i]
        w, h, d = boxes[i]
        result.append((b_idx, x, y, z, w, h, d))

    return result


def _find_position(bin_contents, W, H, D, w, h, d):
    """Find the best position for a box in a bin using a composite score.
    
    Prefers positions that are well-supported (touching floor or other boxes)
    and minimize the 'skyline' height, leading to tighter packing.
    """
    if not bin_contents:
        if w <= W and h <= H and d <= D:
            return (0, 0, 0)
        return None

    best_pos = None
    best_score = None

    # Candidate x positions: 0 and right edges of existing boxes
    x_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        x_candidates.add(bx + bw)
    x_candidates = sorted(x for x in x_candidates if x + w <= W)

    # Candidate y positions: 0 and top edges of existing boxes
    y_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        y_candidates.add(by + bh)
    y_candidates = sorted(y for y in y_candidates if y + h <= H)

    # Candidate z positions: 0 and front edges of existing boxes
    z_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        z_candidates.add(bz + bd)
    z_candidates = sorted(z for z in z_candidates if z + d <= D)

    for z in z_candidates:
        for y in y_candidates:
            for x in x_candidates:
                if _fits(bin_contents, x, y, z, w, h, d):
                    # Compute support score: how many faces are in contact
                    # with floor or another box. More support = more stable/tight.
                    support = 0
                    # Floor support (y=0)
                    if y == 0:
                        support += 1
                    # Check contact with other boxes on all 6 faces
                    for (bx, by, bz, bw, bh, bd) in bin_contents:
                        # X-face contact
                        if x == bx + bw or bx == x + w:
                            if (y < by + bh and by < y + h) and (z < bz + bd and bz < z + d):
                                support += 0.5
                        # Y-face contact
                        if y == by + bh or by == y + h:
                            if (x < bx + bw and bx < x + w) and (z < bz + bd and bz < z + d):
                                support += 1.0  # Stronger weight for vertical support
                        # Z-face contact
                        if z == bz + bd or bz == z + d:
                            if (x < bx + bw and bx < x + w) and (y < by + bh and by < y + h):
                                support += 0.5

                    # Composite score: prefer lower z, lower y, lower x, but reward support
                    # We want to minimize: (z, y, x) but also maximize support
                    # Use a weighted tuple: primary is z, then y, then x, with support as tiebreaker
                    # Actually, let's use: score = (z * 1000 - support * 10, y, x)
                    # This way, a box with more support at the same z is preferred
                    score = (z * 1000 - support * 10, y, x)
                    
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (x, y, z)

    return best_pos


def _fits(bin_contents, x, y, z, w, h, d):
    """Check if a box at (x,y,z) with dims (w,h,d) fits without overlap."""
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        if not (x + w <= bx or bx + bw <= x or
                y + h <= by or by + bh <= y or
                z + d <= bz or bz + bd <= z):
            return False
    return True


def _tighten_last_bin(bins, placement_map, boxes, W, H, D, deadline):
    """Try to repack the last bin more tightly to improve volume fraction."""
    if not bins:
        return
    
    last_b = len(bins) - 1
    if not bins[last_b]:
        return
    
    # Try reordering boxes in the last bin and repacking
    current_boxes = list(bins[last_b])
    if len(current_boxes) < 2:
        return
    
    # Try a few different orderings
    best_arrangement = None
    best_volume = 0
    
    # Original order
    test_bins = [list(bins[last_b])]
    vol = sum(bw * bh * bd for (_, _, _, bw, bh, bd) in test_bins[0])
    if vol > best_volume:
        best_volume = vol
        best_arrangement = test_bins[0]
    
    # Try sorting by height (h) descending - pack tall boxes first
    sorted_by_h = sorted(current_boxes, key=lambda b: b[4], reverse=True)
    test_bins = [sorted_by_h]
    # Verify validity
    if _verify_bin(test_bins[0], W, H, D):
        vol = sum(bw * bh * bd for (_, _, _, bw, bh, bd) in test_bins[0])
        if vol > best_volume:
            best_volume = vol
            best_arrangement = test_bins[0]
    
    # Try sorting by width (w) descending
    sorted_by_w = sorted(current_boxes, key=lambda b: b[3], reverse=True)
    test_bins = [sorted_by_w]
    if _verify_bin(test_bins[0], W, H, D):
        vol = sum(bw * bh * bd for (_, _, _, bw, bh, bd) in test_bins[0])
        if vol > best_volume:
            best_volume = vol
            best_arrangement = test_bins[0]
    
    # Try sorting by depth (d) descending
    sorted_by_d = sorted(current_boxes, key=lambda b: b[5], reverse=True)
    test_bins = [sorted_by_d]
    if _verify_bin(test_bins[0], W, H, D):
        vol = sum(bw * bh * bd for (_, _, _, bw, bh, bd) in test_bins[0])
        if vol > best_volume:
            best_volume = vol
            best_arrangement = test_bins[0]
    
    # Try repacking with the best arrangement using _find_position
    if best_arrangement is not None:
        # Actually, we need to repack properly
        repacked = _repack_bin(best_arrangement, W, H, D, deadline)
        if repacked is not None and _verify_bin(repacked, W, H, D):
            bins[last_b] = repacked
            # Update placement_map
            for box_data in repacked:
                bx, by, bz, bw, bh, bd = box_data
                for idx, (b, x, y, z) in placement_map.items():
                    if b == last_b:
                        # Find which original box this is
                        # This is tricky - we need to track which box is which
                        pass
    
    # Simpler approach: just try to move boxes from last bin to earlier bins
    for box_data in list(bins[last_b]):
        if time.time() > deadline - 2.0:
            break
        bx, by, bz, bw, bh, bd = box_data
        for new_b in range(last_b):
            pos = _find_position(bins[new_b], W, H, D, bw, bh, bd)
            if pos is not None:
                x, y, z = pos
                bins[last_b].remove(box_data)
                bins[new_b].append((x, y, z, bw, bh, bd))
                # Update placement_map
                for idx, (b, px, py, pz) in placement_map.items():
                    if b == last_b and px == bx and py == by and pz == bz:
                        placement_map[idx] = (new_b, x, y, z)
                        break
                break


def _verify_bin(bin_contents, W, H, D):
    """Verify that all boxes in a bin are within bounds and don't overlap."""
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        if bx < 0 or by < 0 or bz < 0:
            return False
        if bx + bw > W or by + bh > H or bz + bd > D:
            return False
    for i in range(len(bin_contents)):
        for j in range(i + 1, len(bin_contents)):
            if not _fits([bin_contents[i]], *bin_contents[j][:3], *bin_contents[j][3:]):
                return False
    return True


def _repack_bin(box_list, W, H, D, deadline):
    """Repack a list of boxes into a single bin using _find_position."""
    if not box_list:
        return []
    
    # Sort by volume descending for better packing
    sorted_boxes = sorted(box_list, key=lambda b: b[3] * b[4] * b[5], reverse=True)
    
    result = []
    for box in sorted_boxes:
        bx, by, bz, bw, bh, bd = box
        pos = _find_position(result, W, H, D, bw, bh, bd)
        if pos is None:
            return None  # Can't pack all boxes
        x, y, z = pos
        result.append((x, y, z, bw, bh, bd))
    
    return result


def _refine(bins, placement_map, boxes, W, H, D, deadline):
    """Try to move boxes to earlier bins or better positions to reduce bin count."""
    max_b = len(bins) - 1
    for b_idx in range(max_b, -1, -1):
        if time.time() > deadline - 3.0:
            break
        if not bins[b_idx]:
            continue

        # Try to move each box in this bin to an earlier bin
        to_move = list(bins[b_idx])
        for box_data in to_move:
            if time.time() > deadline - 3.0:
                break
            bx, by, bz, bw, bh, bd = box_data
            # Find original index
            orig_idx = None
            for idx, (b, x, y, z) in placement_map.items():
                if b == b_idx and x == bx and y == by and z == bz:
                    orig_idx = idx
                    break
            if orig_idx is None:
                continue

            # Try to place in earlier bins
            moved = False
            for new_b in range(b_idx):
                pos = _find_position(bins[new_b], W, H, D, bw, bh, bd)
                if pos is not None:
                    x, y, z = pos
                    # Remove from current bin
                    bins[b_idx].remove(box_data)
                    # Add to new bin
                    bins[new_b].append((x, y, z, bw, bh, bd))
                    placement_map[orig_idx] = (new_b, x, y, z)
                    moved = True
                    break
            if moved:
                continue

            # Try to find a better position in the same bin
            bins[b_idx].remove(box_data)
            pos = _find_position(bins[b_idx], W, H, D, bw, bh, bd)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, bw, bh, bd))
                placement_map[orig_idx] = (b_idx, x, y, z)
            else:
                bins[b_idx].append(box_data)
                placement_map[orig_idx] = (b_idx, bx, by, bz)

    # Remove empty bins and compact labels
    new_bins = []
    label_map = {}
    for old_label in range(len(bins)):
        if bins[old_label]:
            label_map[old_label] = len(new_bins)
            new_bins.append(bins[old_label])
    bins[:] = new_bins
    for idx in placement_map:
        b = placement_map[idx][0]
        if b in label_map:
            placement_map[idx] = (label_map[b],) + placement_map[idx][1:]
# EVOLVE-BLOCK-END
