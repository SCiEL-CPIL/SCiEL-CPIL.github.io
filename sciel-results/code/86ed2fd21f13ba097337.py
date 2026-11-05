"""3D Bin Packing: Best-fit decreasing with efficient candidate generation and bin selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin for 150s limit
    
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
        """Check if box fits within container and doesn't overlap with existing boxes."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_best_position(bin_boxes, w, h, d):
        """
        Find the best position for a box in a bin.
        Returns (x, y, z) or None if no valid position exists.
        Uses candidate points from box faces and scores by tightness/compactness.
        """
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Generate candidate coordinates from existing box faces
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                xs.add(px + pw)
            if py + ph <= H:
                ys.add(py + ph)
            if pz + pd <= D:
                zs.add(pz + pd)
        
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
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
                        # Score: prefer positions that minimize wasted space
                        # Lower coordinates are better (pack toward origin)
                        # Also penalize positions that leave gaps
                        score = x + y + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            break
                if best_pos is not None:
                    break
            if best_pos is not None:
                break
        
        return best_pos
    
    def bin_utilization(bin_boxes):
        """Calculate the volume utilization of a bin."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_boxes)
        return total_vol / (W * H * D)
    
    def place_box(w, h, d):
        """
        Place a box in the best existing bin or create a new one.
        Returns (bin_index, x, y, z).
        """
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_best_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Combined score: prefer bins with higher utilization and tighter positions
                util = bin_utilization(bin_boxes)
                pos_score = 0.3*x + 0.3*y + 0.4*z
                # We want to minimize pos_score and maximize util
                # So: minimize pos_score - lambda * util
                score = pos_score - 0.5 * util
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            # Create a new bin
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
            # Emergency: place remaining boxes in new bins (valid but suboptimal)
            for i in range(idx, n):
                _, w2, h2, d2, _ = indexed_boxes[i]
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Final polish: try to improve last bin occupancy by moving small boxes
    # This helps with the 0.1*(1 - last-bin fraction) term
    if time.time() < deadline - 10:
        # Identify the last bin
        if bins:
            last_bin_idx = max(b[0] for b in placements if b is not None)
            last_bin_boxes = bins[last_bin_idx] if last_bin_idx < len(bins) else []
            
            # Try to move some small boxes from other bins to fill last bin better
            # Only if it doesn't increase bin count
            other_bins = [i for i in range(len(bins)) if i != last_bin_idx and bins[i]]
            
            # Sort other bins by how much volume they have (move from fuller bins first)
            other_bins.sort(key=lambda i: -sum(b[3]*b[4]*b[5] for b in bins[i]))
            
            for ob_idx in other_bins:
                if time.time() > deadline - 5:
                    break
                # Try moving the smallest box from this bin to last bin
                if not bins[ob_idx]:
                    continue
                # Find smallest box in other bin
                min_vol_box = min(range(len(bins[ob_idx])), 
                                  key=lambda j: bins[ob_idx][j][3]*bins[ob_idx][j][4]*bins[ob_idx][j][5])
                bx, by, bz, bw, bh, bd = bins[ob_idx][min_vol_box]
                
                # Check if it can fit in last bin
                pos = find_best_position(last_bin_boxes, bw, bh, bd)
                if pos is not None:
                    # Move it
                    x, y, z = pos
                    bins[ob_idx].pop(min_vol_box)
                    bins[last_bin_idx].append((x, y, z, bw, bh, bd))
                    # Update placement
                    for p_idx, p in enumerate(placements):
                        if p is not None and p[0] == ob_idx and p[1] == bx and p[2] == by and p[3] == bz:
                            placements[p_idx] = (last_bin_idx, x, y, z, bw, bh, bd)
                            break
                    if not bins[ob_idx]:
                        # Bin is now empty, remove it (but keep label for consistency)
                        pass
    
    return placements
# EVOLVE-BLOCK-END
