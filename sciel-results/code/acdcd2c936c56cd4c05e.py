"""3D Bin Packing: Best-fit decreasing with 2D Skyline representation and consolidation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d, max(w,h,d)) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[4], x[5], x[1], x[2], x[3]))
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits within container and doesn't overlap with existing boxes."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # No overlap if separated in any dimension
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in a bin using candidate points with tightness scoring."""
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Generate candidate positions from all combinations of existing box faces
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
        
        # Remove duplicates and sort
        xs = sorted(set(xs))
        ys = sorted(set(ys))
        zs = sorted(set(zs))
        
        # Limit candidates for performance
        if len(xs) > 25:
            xs = xs[:25]
        if len(ys) > 25:
            ys = ys[:25]
        if len(zs) > 25:
            zs = zs[:25]
        
        # Score each valid position; prefer positions that minimize coordinates
        # and reward tightness (contact with existing boxes)
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
                        # Base score: prefer lower coordinates (tighter packing toward origin)
                        # Weight z higher since gravity typically stacks downward
                        score = 0.5 * x + 0.5 * y + 1.0 * z
                        
                        # Tightness bonus: reward positions adjacent to existing boxes
                        tightness = 0.0
                        for (px, py, pz, pw, ph, pd) in bin_boxes:
                            # Check adjacency in each dimension
                            if abs(x - (px + pw)) < 1e-9 or abs((x + w) - px) < 1e-9:
                                if not (x + w <= px or px + pw <= x):
                                    if not (y + h <= py or py + ph <= y):
                                        if not (z + d <= pz or pz + pd <= z):
                                            tightness += 1
                            if abs(y - (py + ph)) < 1e-9 or abs((y + h) - py) < 1e-9:
                                if not (y + h <= py or py + ph <= y):
                                    if not (x + w <= px or px + pw <= x):
                                        if not (z + d <= pz or pz + pd <= z):
                                            tightness += 1
                            if abs(z - (pz + pd)) < 1e-9 or abs((z + d) - pz) < 1e-9:
                                if not (z + d <= pz or pz + pd <= z):
                                    if not (x + w <= px or px + pw <= x):
                                        if not (y + h <= py or py + ph <= y):
                                            tightness += 1
                        
                        # Reward tightness: subtract a small amount for each contact
                        score -= 0.1 * tightness
                        
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def bin_utilization(bin_boxes):
        """Calculate the volume utilization of a bin."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_boxes)
        return total_vol / container_vol
    
    def bin_fragmentation(bin_boxes):
        """
        Estimate fragmentation: how spread out the boxes are.
        Lower = more compact. Uses bounding box volume / actual volume ratio.
        """
        if not bin_boxes:
            return 0.0
        if len(bin_boxes) == 1:
            return 0.0
        
        min_x = min(b[0] for b in bin_boxes)
        min_y = min(b[1] for b in bin_boxes)
        min_z = min(b[2] for b in bin_boxes)
        max_x = max(b[0]+b[3] for b in bin_boxes)
        max_y = max(b[1]+b[4] for b in bin_boxes)
        max_z = max(b[2]+b[5] for b in bin_boxes)
        
        bbox_vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        actual_vol = sum(b[3]*b[4]*b[5] for b in bin_boxes)
        
        if actual_vol == 0:
            return 0.0
        return bbox_vol / actual_vol - 1.0  # 0 = perfect, higher = more fragmented
    
    def place_box(w, h, d):
        """Place a box in the best existing bin or create a new one."""
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Combined score: prefer bins with higher utilization, lower fragmentation, tighter positions
                util = bin_utilization(bin_boxes)
                frag = bin_fragmentation(bin_boxes)
                pos_score = 0.5 * x + 0.5 * y + 1.0 * z
                
                # We want: high util, low frag, low pos_score
                # Score to minimize: pos_score + 0.5*frag - 0.3*util
                score = pos_score + 0.5 * frag - 0.3 * util
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
    for idx, w, h, d, vol, maxdim in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in a new bin each (valid but suboptimal)
            for i in range(idx, n):
                _, w2, h2, d2, _, _ = indexed_boxes[i]
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Bin consolidation pass: try to merge small bins into larger ones to reduce B
    if time.time() < deadline - 5.0 and len(bins) > 1:
        consolidation_deadline = deadline - 2.0
        max_bins = len(bins)
        
        # Try to move boxes from smaller bins to larger bins
        # Sort bins by size (smallest first) to try to eliminate them
        bin_indices = sorted(range(max_bins), key=lambda i: len(bins[i]))
        
        for source_bin in bin_indices:
            if time.time() > consolidation_deadline:
                break
            if len(bins[source_bin]) == 0:
                continue
            
            # Try to move each box from source bin to other bins
            boxes_to_move = list(bins[source_bin])
            bins[source_bin] = []
            
            for box_info in boxes_to_move:
                bx, by, bz, bw, bh, bd = box_info
                placed = False
                
                for target_bin in range(max_bins):
                    if time.time() > consolidation_deadline:
                        break
                    if target_bin == source_bin:
                        continue
                    if len(bins[target_bin]) == 0:
                        continue
                    
                    # Try to find a position in target bin
                    pos = find_position(bins[target_bin], bw, bh, bd)
                    if pos is not None:
                        px, py, pz = pos
                        bins[target_bin].append((px, py, pz, bw, bh, bd))
                        placed = True
                        break
                
                if not placed:
                    # Put it back in source bin
                    bins[source_bin].append(box_info)
        
        # Remove empty bins and renumber
        new_bins = []
        bin_remap = {}
        for old_idx, b in enumerate(bins):
            if len(b) > 0:
                bin_remap[old_idx] = len(new_bins)
                new_bins.append(b)
        
        # Update placements with new bin indices
        for i in range(n):
            if placements[i] is not None:
                old_bin = placements[i][0]
                if old_bin in bin_remap:
                    placements[i] = (bin_remap[old_bin],) + placements[i][1:]
        
        bins = new_bins
    
    return placements
# EVOLVE-BLOCK-END
