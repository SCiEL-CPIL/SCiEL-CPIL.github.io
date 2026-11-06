"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe within 120s
    
    # Sort boxes by volume descending (largest first)
    indexed_boxes = [(i, boxes[i]) for i in range(n)]
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # We'll maintain a list of bins, each bin has a list of placed boxes (x,y,z,w,h,d)
    # For efficient placement, we use a "bottom-left-front" heuristic with a priority queue
    # of candidate positions (x, y, z) sorted by (z, y, x) to prefer bottom, then left, then front
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if a box of size (w,h,d) can be placed at (x,y,z) in the given bin."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Check overlap
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in a bin using bottom-left-front heuristic."""
        # Generate candidate positions from existing boxes
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Positions adjacent to existing boxes
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            # Also try combinations
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
        
        # Filter valid candidates and sort by (z, y, x) for bottom-left-front
        valid_candidates = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if can_place(bin_boxes, x, y, z, w, h, d):
                    valid_candidates.append((z, y, x, x, y, z))
        
        if not valid_candidates:
            return None
        
        valid_candidates.sort()
        return (valid_candidates[0][3], valid_candidates[0][4], valid_candidates[0][5])
    
    # First pass: greedy placement with FFD
    def greedy_pack(sorted_boxes):
        bins = []  # list of lists of (x,y,z,w,h,d)
        placements = []  # (bin_idx, x, y, z, w, h, d)
        
        for idx, (w, h, d) in sorted_boxes:
            placed = False
            # Try existing bins first
            for bi in range(len(bins)):
                pos = find_position(bins[bi], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    placements.append((bi, x, y, z, w, h, d))
                    placed = True
                    break
            
            if not placed:
                # Try to place in a new bin
                if can_place([], 0, 0, 0, w, h, d):
                    bins.append([(0, 0, 0, w, h, d)])
                    placements.append((len(bins)-1, 0, 0, 0, w, h, d))
                else:
                    # Should not happen if box fits in container
                    raise ValueError("Box does not fit in container")
        
        return placements, bins
    
    # Run greedy packing
    placements, bins = greedy_pack(indexed_boxes)
    
    # Check time
    if time.time() > deadline - 10:
        return placements
    
    # Local search: try to move boxes to reduce bin count
    # Strategy: try to consolidate bins by moving boxes from higher-numbered bins to lower ones
    
    def compute_score(placements):
        """Compute the score for a placement."""
        if not placements:
            return 0.0
        max_bin = max(p[0] for p in placements)
        B = max_bin + 1  # number of occupied bins (assuming consecutive from 0)
        
        # Actually, B is the number of occupied bins, not the max label + 1
        # Let's count distinct bin labels
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        last_bin = max(bin_labels)
        
        # Compute last-bin occupied volume
        last_bin_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == last_bin)
        container_vol = W * H * D
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0
        
        return B + 0.1 * (1 - occupied_frac)
    
    best_score = compute_score(placements)
    best_placements = placements[:]
    
    # Local search: try to move boxes between bins
    # We'll do a limited number of iterations
    max_iterations = 50
    for it in range(max_iterations):
        if time.time() > deadline - 15:
            break
        
        # Try to find a box in a higher bin that can be moved to a lower bin
        # to reduce the number of bins or improve the last bin fill
        
        # Get bin assignments
        bin_to_boxes = {}
        for pi, (bi, x, y, z, w, h, d) in enumerate(placements):
            if bi not in bin_to_boxes:
                bin_to_boxes[bi] = []
            bin_to_boxes[bi].append(pi)
        
        bin_labels = sorted(bin_to_boxes.keys())
        if len(bin_labels) <= 1:
            break
        
        # Try to move a box from the last bin to an earlier bin
        last_bin = bin_labels[-1]
        moved = False
        
        for pi in bin_to_boxes[last_bin]:
            bi, x, y, z, w, h, d = placements[pi]
            
            # Try to place this box in an earlier bin
            for target_bin in bin_labels[:-1]:
                # Get the boxes in the target bin
                target_boxes = []
                for p in placements:
                    if p[0] == target_bin:
                        target_boxes.append((p[1], p[2], p[3], p[4], p[5], p[6]))
                
                pos = find_position(target_boxes, w, h, d)
                if pos is not None:
                    # Move the box
                    nx, ny, nz = pos
                    placements[pi] = (target_bin, nx, ny, nz, w, h, d)
                    moved = True
                    break
            
            if moved:
                break
        
        if not moved:
            break
        
        # Update best
        score = compute_score(placements)
        if score < best_score:
            best_score = score
            best_placements = placements[:]
    
    return best_placements
# EVOLVE-BLOCK-END
