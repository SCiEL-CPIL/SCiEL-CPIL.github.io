"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Precompute volumes
    volumes = [w * h * d for w, h, d in boxes]
    
    # Sort indices by volume descending for Best Fit Decreasing
    order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    def try_place(bin_state, box_idx):
        """Try to place box_idx into bin_state. Returns (success, new_state, x, y, z).
        bin_state is a list of (x, y, z, w, h, d) representing placed boxes in this bin.
        We use a simple bottom-left-front (BLF) heuristic with candidate positions.
        """
        w, h, d = boxes[box_idx]
        
        # Generate candidate positions: (0,0,0) and corners of existing boxes
        candidates = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in bin_state:
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
        
        # Also add some "face" candidates for better packing
        for (px, py, pz, pw, ph, pd) in bin_state:
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
        
        # Remove duplicates and sort by (x+y+z) for BLF preference
        candidates = list(set(candidates))
        candidates.sort(key=lambda c: (c[0] + c[1] + c[2], c[2], c[1], c[0]))
        
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            # Check overlap with all existing boxes in this bin
            overlap = False
            for (px, py, pz, pw, ph, pd) in bin_state:
                if not (cx + w <= px or px + pw <= cx or
                        cy + h <= py or py + ph <= cy or
                        cz + d <= pz or pz + pd <= cz):
                    overlap = True
                    break
            if not overlap:
                new_state = bin_state + [(cx, cy, cz, w, h, d)]
                return True, new_state, cx, cy, cz
        
        return False, None, None, None
    
    # Bin states: list of lists of placed boxes
    bins = []  # Each element is a list of (x, y, z, w, h, d)
    
    # Placement result: for each original box index, store (bin_idx, x, y, z)
    placements = [None] * n
    
    # Time budget: 140 seconds max, leave 10s margin
    t_start = time.time()
    TIME_LIMIT = 140.0
    
    for box_idx in order:
        w, h, d = boxes[box_idx]
        if time.time() - t_start > TIME_LIMIT:
            break
        
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        # Try to fit in existing bins first
        for b_idx, bin_state in enumerate(bins):
            if time.time() - t_start > TIME_LIMIT:
                break
            success, new_state, cx, cy, cz = try_place(bin_state, box_idx)
            if success:
                # Score: prefer bins that are more full (to minimize number of bins)
                # and positions closer to origin
                pos_score = cx + cy + cz
                bin_fill = sum(pw * ph * pd for (px, py, pz, pw, ph, pd) in bin_state)
                # Prefer higher fill and lower position
                score = -bin_fill * 1000 + pos_score
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = (cx, cy, cz)
        
        if best_bin == -1:
            # Open a new bin
            bins.append([])
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
            # Place at origin
            bins[best_bin].append((0, 0, 0, w, h, d))
            placements[box_idx] = (best_bin, 0, 0, 0)
        else:
            # Place in existing bin
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placements[box_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2])
    
    # Handle any unplaced boxes (shouldn't happen if time limit not hit)
    for i in range(n):
        if placements[i] is None:
            # Force place in a new bin
            bins.append([])
            b_idx = len(bins) - 1
            bins[b_idx].append((0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            placements[i] = (b_idx, 0, 0, 0)
    
    # Build result: (bin, x, y, z, w, h, d) for each box in original order
    result = []
    for i in range(n):
        b, x, y, z = placements[i]
        w, h, d = boxes[i]
        result.append((b, x, y, z, w, h, d))
    
    return result
# EVOLVE-BLOCK-END
