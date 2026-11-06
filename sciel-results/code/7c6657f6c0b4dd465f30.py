"""3D Bin Packing with Backtracking Search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin within 120s
    
    container_vol = W * H * D
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # We'll use a backtracking search to find a good packing
    # State: list of bins, each bin is a list of placed boxes (x, y, z, w, h, d)
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box can be placed at (x, y, z) in the given bin."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def get_candidate_positions(bin_placements, w, h, d):
        """Generate candidate positions for placing a box in a bin."""
        candidates = set([(0, 0, 0)])
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Positions adjacent to existing boxes
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter valid candidates
        valid = []
        for (x, y, z) in candidates:
            if can_place(bin_placements, x, y, z, w, h, d):
                valid.append((x, y, z))
        
        # Sort by preference: minimize z, then y, then x (bottom-left-back)
        valid.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid
    
    def backtrack(box_idx, bins, placements, best_solution, best_score):
        """
        Recursive backtracking search.
        box_idx: index in indexed_boxes of the next box to place
        bins: list of lists of placed boxes
        placements: list of (bin_idx, x, y, z, w, h, d) for each original box
        best_solution: best placements found so far
        best_score: best score found so far
        Returns: (placements, score) if found, else None
        """
        if time.time() > deadline:
            return None
        
        if box_idx == n:
            # All boxes placed, calculate score
            occupied_bins = set()
            for p in placements:
                if p is not None:
                    occupied_bins.add(p[0])
            B = len(occupied_bins)
            if B == 0:
                return (placements, float('inf'))
            
            last_bin = max(occupied_bins)
            last_bin_vol = 0
            for p in placements:
                if p is not None and p[0] == last_bin:
                    last_bin_vol += p[4] * p[5] * p[6]
            
            if container_vol == 0:
                score = float(B)
            else:
                last_bin_fraction = last_bin_vol / container_vol
                score = B + 0.1 * (1 - last_bin_fraction)
            
            if score < best_score[0]:
                best_score[0] = score
                best_solution[0] = list(placements)
            
            return (placements, score)
        
        orig_idx, w, h, d, vol = indexed_boxes[box_idx]
        
        # Try placing in each existing bin
        for b_idx in range(len(bins)):
            candidates = get_candidate_positions(bins[b_idx], w, h, d)
            for (x, y, z) in candidates:
                # Place the box
                bins[b_idx].append((x, y, z, w, h, d))
                placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                
                result = backtrack(box_idx + 1, bins, placements, best_solution, best_score)
                
                # Undo the placement
                bins[b_idx].pop()
                placements[orig_idx] = None
                
                if result is not None:
                    # We found a complete solution, but we want to keep searching
                    # for better solutions within the time limit
                    pass
        
        # Try placing in a new bin
        if len(bins) < n:  # Don't create more bins than boxes
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
            placements[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
            
            result = backtrack(box_idx + 1, bins, placements, best_solution, best_score)
            
            # Undo
            bins.pop()
            placements[orig_idx] = None
        
        return None
    
    # Initialize
    placements = [None] * n
    bins = []
    best_solution = [None]
    best_score = [float('inf')]
    
    # Run backtracking search
    backtrack(0, bins, placements, best_solution, best_score)
    
    if best_solution[0] is not None:
        return best_solution[0]
    
    # Fallback: simple shelf packing if backtracking didn't find a solution
    result = []
    b = 0
    x = y = z = 0
    row_h = layer_d = 0
    
    for w, h, d in boxes:
        if x + w > W:
            x = 0
            y += row_h
            row_h = 0
        if y + h > H:
            y = 0
            z += layer_d
            layer_d = 0
            x = 0
            row_h = 0
        if z + d > D:
            b += 1
            x = y = z = 0
            row_h = layer_d = 0
        
        result.append((b, x, y, z, w, h, d))
        x += w
        row_h = max(row_h, h)
        layer_d = max(layer_d, d)
    
    return result
# EVOLVE-BLOCK-END
