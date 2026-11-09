import time
import math
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed_boxes = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    # bins: list of lists of (x, y, z, w, h, d, orig_idx)
    bins = []
    bin_volumes = []  # cumulative volume in each bin
    
    # Helper: check if a box at (cx, cy, cz) with dims (w, h, d) overlaps with any box in bin_boxes
    def check_overlap(cx, cy, cz, w, h, d, bin_boxes):
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            # Check if intervals overlap in all 3 dimensions
            if (cx + w <= bx or bx + bw <= cx):
                continue  # no overlap in x
            if (cy + h <= by or by + bh <= cy):
                continue  # no overlap in y
            if (cz + d <= bz or bz + bd <= cz):
                continue  # no overlap in z
            # Overlap in all 3 dimensions
            return True
        return False
    
    # Helper: generate candidate positions for a bin
    def generate_candidates(bin_boxes, w, h, d):
        candidates = set()
        candidates.add((0, 0, 0))
        
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            # Face points
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            
            # Also add combinations of face points for better gap-filling
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
        
        # Filter out positions that exceed container bounds
        valid_candidates = []
        for cx, cy, cz in candidates:
            if cx + w <= W and cy + h <= H and cz + d <= D:
                valid_candidates.append((cx, cy, cz))
        
        return valid_candidates
    
    # Helper: score a placement (lower is better)
    def placement_score(cx, cy, cz, w, h, d, bin_boxes, bin_vol):
        # Primary: minimize distance to origin (packs tighter)
        dist = cx + cy + cz
        
        # We use a weighted combination to pack toward the origin
        return (dist, cz, cy, cx)
    
    # Main packing loop
    for w, h, d, orig_idx in indexed_boxes:
        # Check time
        if time.time() > deadline:
            # Fallback: place remaining boxes in new bins at (0,0,0)
            for rem_w, rem_h, rem_d, rem_idx in indexed_boxes:
                if rem_idx == orig_idx:
                    continue
                bins.append([])
                bins[-1].append((0, 0, 0, rem_w, rem_h, rem_d, rem_idx))
                bin_volumes.append(rem_w * rem_h * rem_d)
            # Also place the current box
            bins.append([])
            bins[-1].append((0, 0, 0, w, h, d, orig_idx))
            bin_volumes.append(w * h * d)
            break
        
        best_bin_idx = -1
        best_pos = None
        best_score = None
        
        # Try existing bins, preferring fuller bins first (to minimize B)
        if bins:
            bin_order = sorted(range(len(bins)), key=lambda i: bin_volumes[i], reverse=True)
        else:
            bin_order = []
        
        for b_idx in bin_order:
            bin_boxes = bins[b_idx]
            bin_vol = bin_volumes[b_idx]
            
            # Generate candidate positions
            candidates = generate_candidates(bin_boxes, w, h, d)
            
            for cx, cy, cz in candidates:
                # Check overlap
                if check_overlap(cx, cy, cz, w, h, d, bin_boxes):
                    continue
                
                # Score
                score = placement_score(cx, cy, cz, w, h, d, bin_boxes, bin_vol)
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin_idx = b_idx
                    best_pos = (cx, cy, cz)
        
        if best_bin_idx == -1:
            # Open new bin
            bins.append([])
            bin_volumes.append(0)
            best_bin_idx = len(bins) - 1
            best_pos = (0, 0, 0)
        
        bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, orig_idx))
        bin_volumes[best_bin_idx] += w * h * d
    
    # Construct result in original order
    final_placements = [None] * len(boxes)
    for b_idx, bin_boxes in enumerate(bins):
        for x, y, z, w, h, d, orig_idx in bin_boxes:
            final_placements[orig_idx] = (b_idx, x, y, z, w, h, d)
    
    return final_placements
