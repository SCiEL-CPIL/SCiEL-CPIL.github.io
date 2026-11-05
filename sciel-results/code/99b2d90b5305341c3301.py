"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # New Approach: Multi-Start First Fit with Bottom-Left-Back (BLB) placement
    # This differs from the previous Best Fit Decreasing (BFD) with Z-priority.
    
    def generate_candidates(bin_items, w, h, d):
        """
        Generate candidate positions for a box of size (w,h,d) in a bin.
        Candidates are derived from the faces of existing boxes.
        """
        candidates = set()
        # Origin
        candidates.add((0, 0, 0))
        
        for (bw, bh, bd, bx, by, bz) in bin_items:
            # Right face
            candidates.add((bx + bw, by, bz))
            # Back face
            candidates.add((bx, by + bh, bz))
            # Top face
            candidates.add((bx, by, bz + bd))
            
            # Corners (combinations)
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
            
            # Additional sliding positions? 
            # Sometimes placing at (bx, by, bz + bd) is same as top face.
            # We also check positions "inside" the bounding box of existing items?
            # No, BLB typically relies on the "skyline" or face projections.
            
        # Filter valid candidates
        valid_candidates = []
        for c in candidates:
            x, y, z = c
            if 0 <= x <= W - w and 0 <= y <= H - h and 0 <= z <= D - d:
                valid_candidates.append(c)
        
        return valid_candidates

    def can_place(bin_items, w, h, d, x, y, z):
        """Check if box (w,h,d) at (x,y,z) overlaps with any item in bin_items."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        
        for (bw, bh, bd, bx, by, bz) in bin_items:
            # Check overlap
            if (x < bx + bw and x + w > bx and
                y < by + bh and y + h > by and
                z < bz + bd and z + d > bz):
                return False
        return True

    def find_blb_position(bin_items, w, h, d):
        """
        Find the Bottom-Left-Back position.
        BLB: Minimize X, then Y, then Z.
        """
        candidates = generate_candidates(bin_items, w, h, d)
        
        best_pos = None
        best_key = None
        
        for (x, y, z) in candidates:
            if can_place(bin_items, w, h, d, x, y, z):
                # BLB Key: (x, y, z)
                key = (x, y, z)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (x, y, z)
                    
        return best_pos

    def run_ffd_blb(sorted_indices):
        """
        Run First Fit Decreasing with BLB placement.
        sorted_indices: list of original indices in the order to process.
        """
        bins = [] # List of lists of (w, h, d, x, y, z)
        placements = [None] * n
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            placed = False
            
            # First Fit: Try to place in the first bin where it fits
            for b_idx, bin_items in enumerate(bins):
                pos = find_blb_position(bin_items, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx].append((w, h, d, x, y, z))
                    placements[idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                # Start new bin
                bins.append([(w, h, d, 0, 0, 0)])
                placements[idx] = (len(bins)-1, 0, 0, 0, w, h, d)
                
        if not bins:
            return placements, 0, 0.0
            
        num_bins = len(bins)
        last_bin_idx = max(p[0] for p in placements if p is not None)
        
        last_bin_vol = 0
        for b_idx, x, y, z, w, h, d in placements:
            if b_idx == last_bin_idx:
                last_bin_vol += w * h * d
                
        container_vol = W * H * D
        fill_ratio = last_bin_vol / container_vol if container_vol > 0 else 0.0
        
        return placements, num_bins, fill_ratio

    best_result = None
    best_score = float('inf')
    
    # Base sorting: Volume descending
    indexed_boxes = list(range(n))
    base_sorted = sorted(indexed_boxes, key=lambda i: boxes[i][1]*boxes[i][2]*boxes[i][3], reverse=True)
    
    # Multi-start with slight randomization in sorting to escape local optima
    # We will run a few variants.
    variants = [base_sorted]
    
    # Variant 2: Sort by Max Dimension
    variant2 = sorted(indexed_boxes, key=lambda i: max(boxes[i][1], boxes[i][2], boxes[i][3]), reverse=True)
    variants.append(variant2)
    
    # Variant 3: Sort by Surface Area
    variant3 = sorted(indexed_boxes, key=lambda i: 2*(boxes[i][1]*boxes[i][2] + boxes[i][2]*boxes[i][3] + boxes[i][1]*boxes[i][3]), reverse=True)
    variants.append(variant3)
    
    # Variant 4: Randomized Shuffle of top 10 items from Volume Sort
    random.seed(42) # Deterministic for reproducibility in evaluation if needed, but task allows randomness? 
    # Actually, better to keep it deterministic for stability. 
    # Let's use a "Tie-breaking" randomization based on index.
    # We'll just stick to the 3 deterministic sorts for now, as they cover different aspects.
    
    # Add a "Mixed" sort: Volume, but if volumes are close, use Max Dim
    def mixed_key(i):
        vol = boxes[i][1]*boxes[i][2]*boxes[i][3]
        max_d = max(boxes[i][1], boxes[i][2], boxes[i][3])
        # Primary: Volume, Secondary: Max Dim
        return (-vol, -max_d)
        
    variant4 = sorted(indexed_boxes, key=mixed_key)
    variants.append(variant4)
    
    for v in variants:
        placements, num_bins, fill_ratio = run_ffd_blb(v)
        
        # Calculate score
        score = num_bins + 0.1 * (1 - fill_ratio)
        
        if score < best_score:
            best_score = score
            best_result = placements
            
    return best_result
# EVOLVE-BLOCK-END
