import time

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []

    # Sort indices by volume descending
    sorted_indices = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # bins: list of lists of (x, y, z, w, h, d, original_index)
    bins = []
    
    def fits_at(x, y, z, w, h, d, bin_boxes):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd, _) in bin_boxes:
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return False
        return True

    def get_candidates(bin_boxes):
        # Generate candidate positions: (0,0,0) and corners of existing boxes
        xs = {0}
        ys = {0}
        zs = {0}
        for (bx, by, bz, bw, bh, bd, _) in bin_boxes:
            xs.add(bx + bw)
            ys.add(by + bh)
            zs.add(bz + bd)
        
        # Filter out of bounds
        xs = [x for x in xs if x + 0 <= W] # We check +w later
        ys = [y for y in ys if y + 0 <= H]
        zs = [z for z in zs if z + 0 <= D]
        
        # To keep it fast, we don't cross-product all, we iterate
        return xs, ys, zs

    # Main packing loop
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        placed = False
        
        # Try to fit in existing bins
        for b_idx, bin_boxes in enumerate(bins):
            xs, ys, zs = get_candidates(bin_boxes)
            # Sort candidates to prefer lower x, y, z (compact packing)
            for x in sorted(xs):
                if x + w > W: continue
                for y in sorted(ys):
                    if y + h > H: continue
                    for z in sorted(zs):
                        if z + d > D: continue
                        if fits_at(x, y, z, w, h, d, bin_boxes):
                            bin_boxes.append((x, y, z, w, h, d, idx))
                            placed = True
                            break
                    if placed: break
                if placed: break
            if placed: break
        
        if not placed:
            # Open new bin
            bins.append([(0, 0, 0, w, h, d, idx)])
            
    # Post-processing: Try to move boxes from earlier bins to the last bin to increase last bin fill
    # and potentially empty earlier bins (reducing B).
    if len(bins) > 1:
        last_bin_idx = len(bins) - 1
        last_bin = bins[last_bin_idx]
        
        # Iterate through bins from 0 to last_bin_idx - 1
        for b_idx in range(last_bin_idx):
            if not bins[b_idx]:
                continue
                
            boxes_in_bin = list(bins[b_idx])
            for i, box_info in enumerate(boxes_in_bin):
                x, y, z, w, h, d, orig_idx = box_info
                
                # Try to place this box in the last bin
                xs, ys, zs = get_candidates(last_bin)
                found = False
                for nx in sorted(xs):
                    if nx + w > W: continue
                    for ny in sorted(ys):
                        if ny + h > H: continue
                        for nz in sorted(zs):
                            if nz + d > D: continue
                            if fits_at(nx, ny, nz, w, h, d, last_bin):
                                # It fits. Remove from source, add to last.
                                bins[b_idx].pop(i)
                                last_bin.append((nx, ny, nz, w, h, d, orig_idx))
                                found = True
                                break
                        if found: break
                    if found: break
                
                if found:
                    # If source bin is empty, we might want to remove it to keep indices clean?
                    # But bin labels are just integers. If bin 0 is empty, it's not "occupied".
                    # The evaluator counts "occupied bins".
                    # So we don't strictly need to remove empty bins from the list, 
                    # but we must ensure we don't count them in B.
                    # The output will just not have entries for that bin label.
                    # However, if we remove it, the "last bin" index might change if we remove the last one?
                    # We are moving TO the last bin, so last bin stays full.
                    # If we empty bin 0, B decreases.
                    # We can filter out empty bins at the end.
                    pass
            
            # Re-evaluate if bin is empty? 
            # If bins[b_idx] is empty, we can remove it from the list to simplify?
            # But removing from list changes indices of subsequent bins.
            # Since we are iterating forward, and removing from current, it's safe if we are careful.
            # But it's easier to just leave empty bins in the list and filter at the end.
            # However, if we leave empty bins, `len(bins)` is not B.
            # We must calculate B as number of non-empty bins.
            
    # Filter out empty bins and re-index? 
    # The problem says "Bin labels are nonnegative integers and need not be consecutive".
    # So we can keep the original indices (0, 1, 2...) even if some are empty.
    # But to be safe and clean, let's re-index them to 0..B-1.
    # This ensures "largest label" is B-1.
    
    new_bins = []
    for bin_boxes in bins:
        if bin_boxes:
            new_bins.append(bin_boxes)
    
    bins = new_bins
    
    # Construct result in original order
    placements = [None] * n
    for b_idx, bin_boxes in enumerate(bins):
        for (x, y, z, w, h, d, orig_idx) in bin_boxes:
            placements[orig_idx] = (b_idx, x, y, z, w, h, d)
            
    # Safety check: if any placement is None, something went wrong.
    # This should not happen if the logic above is correct.
    # But if it does, we need to handle it to avoid NoneType error.
    # If a box is missing, we can try to place it in a new bin.
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            # Place in a new bin (or existing if we missed it)
            # Since we just re-indexed, let's add a new bin.
            new_bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, i)])
            placements[i] = (new_bin_idx, 0, 0, 0, w, h, d)
            
    return placements
