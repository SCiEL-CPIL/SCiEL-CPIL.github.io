"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    def find_position(placed, w, h, d, W, H, D):
        """Find the best (minimize x+y+z) position for a box in a bin."""
        best_pos = None
        best_score = float('inf')
        
        # Generate candidate positions based on existing placed boxes
        candidates = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in placed:
            # Place to the right, above, or behind existing boxes
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
        
        # Also try some heuristic positions
        for cx in range(0, W, max(1, W // 10)):
            for cy in range(0, H, max(1, H // 10)):
                for cz in range(0, D, max(1, D // 10)):
                    candidates.append((cx, cy, cz))
        
        for (x, y, z) in candidates:
            if x + w > W or y + h > H or z + d > D:
                continue
            # Check for overlap
            overlap = False
            for (px, py, pz, pw, ph, pd) in placed:
                if (x < px + pw and x + w > px and
                    y < py + ph and y + h > py and
                    z < pz + pd and z + d > pz):
                    overlap = True
                    break
            if not overlap:
                score = x + y + z
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
        
        return best_pos
    
    def pack_into_bin(boxes_list, W, H, D):
        """Try to pack a list of boxes into a single bin. Returns list of (index, x, y, z) or None if not all fit."""
        placed = []
        result = []
        for (idx, w, h, d, vol) in boxes_list:
            pos = find_position(placed, w, h, d, W, H, D)
            if pos is None:
                return None
            x, y, z = pos
            placed.append((x, y, z, w, h, d))
            result.append((idx, x, y, z))
        return result
    
    # Greedy approach: try to fit as many boxes as possible into the first bin
    # Then try to fill the last bin as much as possible
    
    # Strategy: Use a multi-pass approach
    # 1. Sort by volume descending
    # 2. Greedily place boxes into bins
    # 3. Try to optimize the last bin's fill
    
    placements = {}  # index -> (bin, x, y, z, w, h, d)
    bins = []  # list of lists of (idx, x, y, z, w, h, d)
    
    current_bin = 0
    current_bin_placed = []
    current_bin_boxes = []
    
    for (idx, w, h, d, vol) in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try to place in current bin
        pos = find_position(current_bin_placed, w, h, d, W, H, D)
        if pos is not None:
            x, y, z = pos
            current_bin_placed.append((x, y, z, w, h, d))
            current_bin_boxes.append((idx, x, y, z, w, h, d))
        else:
            # Start a new bin
            if current_bin_boxes:
                bins.append(current_bin_boxes)
            current_bin += 1
            current_bin_placed = [(0, 0, 0, w, h, d)]
            current_bin_boxes = [(idx, 0, 0, 0, w, h, d)]
    
    if current_bin_boxes:
        bins.append(current_bin_boxes)
    
    # Now try to improve: move some boxes from earlier bins to the last bin to increase its fill
    # Or rearrange to reduce the number of bins
    
    # Simple improvement: try to consolidate
    # For now, just return the greedy result
    for b_idx, bin_boxes in enumerate(bins):
        for (idx, x, y, z, w, h, d) in bin_boxes:
            placements[idx] = (b_idx, x, y, z, w, h, d)
    
    # Build final result in original order
    result = []
    for i in range(n):
        if i in placements:
            result.append(placements[i])
        else:
            # Should not happen, but fallback
            result.append((0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
    
    return result
# EVOLVE-BLOCK-END
