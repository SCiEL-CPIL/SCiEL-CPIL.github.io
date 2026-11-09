"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import numpy as np
from collections import defaultdict

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)

    if n == 0:
        return []

    # Sort boxes by volume descending (FFD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)

    bins = []  # List of lists of (x, y, z, w, h, d)
    placements = [None] * n

    def check_overlap(x, y, z, w, h, d, items):
        for ex, ey, ez, ew, eh, ed in items:
            if not (x + w <= ex or ex + ew <= x or
                    y + h <= ey or ey + eh <= y or
                    z + d <= ez or ez + ed <= z):
                return True
        return False

    def get_candidates(items, W, H, D):
        xs = {0}
        ys = {0}
        zs = {0}
        for ex, ey, ez, ew, eh, ed in items:
            if ex + ew <= W:
                xs.add(ex + ew)
            if ey + eh <= H:
                ys.add(ey + eh)
            if ez + ed <= D:
                zs.add(ez + ed)
        return sorted(xs), sorted(ys), sorted(zs)

    def find_place_in_bin(items, w, h, d, W, H, D):
        xs, ys, zs = get_candidates(items, W, H, D)
        valid_xs = [x for x in xs if x + w <= W]
        if not valid_xs:
            return None
        valid_ys = [y for y in ys if y + h <= H]
        if not valid_ys:
            return None
        valid_zs = [z for z in zs if z + d <= D]
        if not valid_zs:
            return None

        best_pos = None
        best_score = float('inf')

        for x in valid_xs:
            for y in valid_ys:
                for z in valid_zs:
                    if check_overlap(x, y, z, w, h, d, items):
                        continue
                    # Score: Minimize x, then y, then z (Bottom-Left-Front)
                    # Using weighted sum to break ties effectively
                    score = x * (H * D) + y * D + z
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
        return best_pos

    # Phase 1: Greedy Best Fit Decreasing
    # We use Best Fit instead of First Fit to potentially pack tighter
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placements[rem_idx] = (len(bins) - 1, 0, 0, 0, rw, rh, rd)
            return placements

        best_bin_idx = -1
        best_pos = None
        best_waste = float('inf')

        # Try to fit in existing bins
        for b_idx, items in enumerate(bins):
            pos = find_place_in_bin(items, w, h, d, W, H, D)
            if pos is not None:
                x, y, z = pos
                # Calculate "waste" or tightness
                # Prefer positions that are lower (smaller z), then y, then x
                # But also consider how much space is left?
                # Simple heuristic: minimize z, then y, then x
                # This tends to fill the bottom of the bin first.
                waste = z * 10000 + y * 100 + x
                if waste < best_waste:
                    best_waste = waste
                    best_bin_idx = b_idx
                    best_pos = pos

        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d))
            placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # New bin
            bins.append([(0, 0, 0, w, h, d)])
            placements[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Phase 2: Local Search - Relocation to reduce bin count
    # We try to move boxes from the last occupied bin into earlier bins.
    # If successful, the last bin becomes empty (or has fewer boxes), potentially reducing B or improving last-bin fill.

    # Get list of bin indices that are non-empty
    # We need to track original indices to update placements correctly.
    # Let's create a mapping from (bin_idx, box_index_in_bin) to orig_idx?
    # Or better, store orig_idx in the bin items.
    # Current bin items: (x, y, z, w, h, d)
    # Let's convert bins to include orig_idx: (x, y, z, w, h, d, orig_idx)
    # We need to rebuild the bins structure with orig_idx.

    # Rebuild bins with orig_idx
    # We have placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
    # Let's create new_bins
    new_bins = [[] for _ in range(len(bins))]
    for orig_idx, (bin_idx, x, y, z, w, h, d) in enumerate(placements):
        if bin_idx is not None:
            new_bins[bin_idx].append((x, y, z, w, h, d, orig_idx))

    bins = new_bins

    # Now bins[b_idx] is a list of (x, y, z, w, h, d, orig_idx)
    # Update find_place_in_bin to ignore the last element
    def find_place_in_bin_v2(items, w, h, d, W, H, D):
        # items are (x, y, z, w, h, d, orig_idx)
        # Extract geometry
        geom_items = [(x, y, z, ww, hh, dd) for x, y, z, ww, hh, dd, _ in items]
        return find_place_in_bin(geom_items, w, h, d, W, H, D)

    # Relocation Loop
    relocation_attempts = 0
    max_relocations = 2000

    # We'll try to empty the last bin first, then the second last, etc.
    # Or just iterate: for each bin from last to first, try to move its items to earlier bins.
    # If a bin becomes empty, remove it (or just leave it empty, B counts occupied).
    # But removing empty bins at the end reduces the "last bin" index, which might help the score if the last bin was partially filled?
    # Score: mean B + 0.1*(1 - last-bin occupied volume fraction).
    # If we empty the last bin, B decreases by 1. The new last bin is the one before.
    # This is a clear improvement.

    # Strategy:
    # While there are occupied bins:
    #   Let last_bin be the highest index with items.
    #   Try to move one item from last_bin to an earlier bin (0 to last_bin-1).
    #   If successful, remove it from last_bin.
    #   If last_bin becomes empty, remove it from the list (and adjust indices? No, just leave it empty or pop if it's the last one).
    #   If we can't move any item from last_bin, stop? Or try to move items from last_bin-1 to last_bin? No, that increases B or keeps it same.
    #   Actually, if we can't move items out of the last bin, we are stuck with B = len(occupied_bins).
    #   We can try to rearrange items within bins to free space? That's complex.
    #   Let's stick to moving items out of the last bin.

    # To be efficient, we'll iterate through bins from the last one backwards.
    # For each bin, we try to move its items to earlier bins.
    # We'll do this in a loop until no more moves are possible or time runs out.

    # Note: Moving an item changes the available space in the source and destination bins.
    # We must be careful with indices.

    # Let's implement a simpler version:
    # Iterate bin_idx from len(bins)-1 down to 1:
    #   While bins[bin_idx] is not empty:
    #     Pick an item (e.g., the one with largest volume, or first one)
    #     Try to place it in bins 0 to bin_idx-1
    #     If placed, remove from bins[bin_idx] and add to target bin. Update placements.
    #     If not placed, break (try next bin_idx? No, if we can't move one, maybe we can't move others either? Not necessarily. But let's try all items in this bin).
    #     If we successfully move an item, continue checking this bin.
    #     If we fail to move any item from this bin, we move to bin_idx-1.

    # Let's try to move items from the last occupied bin to earlier bins.
    # We'll keep a list of occupied bin indices.
    # Actually, just iterating backwards is fine.

    # Optimization: Only try to move if time allows.
    # We'll use a flag to indicate if a move was made in the last pass. If no moves, we might stop.

    # Let's do a few passes.
    for pass_num in range(5):
        if time.time() > deadline:
            break

        moved_any = False
        # Iterate from last bin to first
        for b_idx in range(len(bins) - 1, 0, -1):
            if time.time() > deadline:
                break
            if not bins[b_idx]:
                continue

            # Try to move each item in this bin to an earlier bin
            # We'll create a list of items to try, to avoid modifying list while iterating
            items_to_try = list(bins[b_idx])
            for item in items_to_try:
                if time.time() > deadline:
                    break
                x, y, z, w, h, d, orig_idx = item

                # Check if item is still in this bin (it might have been moved)
                if item not in bins[b_idx]:
                    continue

                # Try to place in earlier bins
                best_target_bin = -1
                best_target_pos = None
                best_waste = float('inf')

                for t_idx in range(b_idx):
                    if not bins[t_idx]:
                        continue
                    pos = find_place_in_bin_v2(bins[t_idx], w, h, d, W, H, D)
                    if pos is not None:
                        tx, ty, tz = pos
                        waste = tz * 10000 + ty * 100 + tx
                        if waste < best_waste:
                            best_waste = waste
                            best_target_bin = t_idx
                            best_target_pos = pos

                if best_target_bin != -1:
                    # Move the item
                    bins[b_idx].remove(item)
                    tx, ty, tz = best_target_pos
                    bins[best_target_bin].append((tx, ty, tz, w, h, d, orig_idx))
                    placements[orig_idx] = (best_target_bin, tx, ty, tz, w, h, d)
                    moved_any = True
                    relocation_attempts += 1
                    if relocation_attempts > max_relocations:
                        break

            if relocation_attempts > max_relocations:
                break

        if not moved_any:
            break

    # Final cleanup: Remove empty bins from the end to minimize max label?
    # If bin 5 is empty, and bin 4 is full, max label is 5.
    # If we remove bin 5, max label is 4.
    # This improves the "last-bin occupied volume fraction" term if the last bin was empty (0 fraction -> 0.1 * 1 = 0.1 penalty).
    # If last bin is full, removing empty bins before it doesn't change the last bin.
    # So we should compact the bin list from the end.
    # But we need to update placements?
    # If we remove a bin, the indices of subsequent bins change?
    # No, we are removing from the end. So indices of remaining bins are unchanged.
    # But we need to update the bin index in `placements` for any box that was in a removed bin?
    # If the bin was empty, no boxes were in it. So no placements to update.
    # So we can just pop empty bins from the end.

    while bins and not bins[-1]:
        bins.pop()

    return placements
# EVOLVE-BLOCK-END
