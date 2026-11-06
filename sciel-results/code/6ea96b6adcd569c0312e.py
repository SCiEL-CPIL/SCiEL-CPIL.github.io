"""3D Bin Packing: BFD with adjacency-aware positioning, fragmentation-aware bin selection, and consolidation polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def adjacency_score(bin_boxes, x, y, z, w, h, d):
        """
        Score how tightly a box at (x,y,z) is packed against existing boxes.
        Lower is better. Rewards being flush against existing box faces on multiple sides.
        """
        if not bin_boxes:
            return 0.0
        # For each of the 6 faces of the new box, check if it's flush with an existing box face
        # Face at x: flush if any existing box has px+pw == x (box is to the left)
        # Face at x+w: flush if any existing box has px == x+w (box is to the right)
        # Similarly for y and z
        flush_count = 0
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw == x:
                flush_count += 1
            if px == x + w:
                flush_count += 1
            if py + ph == y:
                flush_count += 1
            if py == y + h:
                flush_count += 1
            if pz + pd == z:
                flush_count += 1
            if pz == z + d:
                flush_count += 1
        # Higher flush count = tighter packing = lower score
        return -flush_count
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in a bin using candidate points."""
        xs = [0]
        ys = [0]
        zs = [0]
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                xs.append(px + pw)
            if py + ph <= H:
                ys.append(py + ph)
            if pz + pd <= D:
                zs.append(pz + pd)
        
        xs = sorted(set(xs))
        ys = sorted(set(ys))
        zs = sorted(set(zs))
        
        best_score = float('inf')
        best_pos = None
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        # Combined score: prefer corner (low coords) and tight adjacency
                        # Weight: corner position is primary, adjacency is secondary
                        corner_score = 0.5*x + 0.3*y + 0.2*z
                        adj = adjacency_score(bin_boxes, x, y, z, w, h, d)
                        score = corner_score + 0.1 * adj
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def bin_fill_fraction(bin_boxes):
        """Compute the volume fraction of a bin that is occupied."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in bin_boxes)
        return total_vol / (W * H * D)
    
    def place_box(w, h, d):
        """Place a box in the best existing bin or create a new one."""
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Bin selection score: prefer bins where the box fits tightly
                # and where the bin is already well-filled (consolidation)
                fill = bin_fill_fraction(bin_boxes)
                # Lower score is better: prefer higher fill (consolidate) and lower max coord
                max_coord = max(x, y, z)
                score = max_coord - 2.0 * fill * max(W, H, D)
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            bins.append([])
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            for i, (w2, h2, d2) in enumerate(boxes[idx:]):
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[idx + i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Consolidation polish: try to move boxes from later bins into earlier bins
    # Only if we have time remaining
    if time.time() < deadline - 10.0 and len(bins) > 1:
        for b_idx in range(len(bins) - 1, 0, -1):
            if time.time() > deadline - 10.0:
                break
            if not bins[b_idx]:
                continue
            # Try to move each box in this bin to an earlier bin
            for box_idx in range(len(bins[b_idx]) - 1, -1, -1):
                if time.time() > deadline - 10.0:
                    break
                px, py, pz, pw, ph, pd = bins[b_idx][box_idx]
                # Remove from current bin
                bins[b_idx].pop(box_idx)
                # Try to place in earlier bins
                moved = False
                for target_bin in range(b_idx):
                    if not bins[target_bin]:
                        continue
                    pos = find_position(bins[target_bin], pw, ph, pd)
                    if pos is not None:
                        tx, ty, tz = pos
                        bins[target_bin].append((tx, ty, tz, pw, ph, pd))
                        # Update placement
                        for orig_idx, (bi, bx, by, bz, bw, bh, bd) in enumerate(placements):
                            if bi == b_idx and bx == px and by == py and bz == pz and bw == pw and bh == ph and bd == pd:
                                placements[orig_idx] = (target_bin, tx, ty, tz, pw, ph, pd)
                                break
                        moved = True
                        break
                if not moved:
                    # Put back
                    bins[b_idx].append((px, py, pz, pw, ph, pd))
    
    return placements
# EVOLVE-BLOCK-END
