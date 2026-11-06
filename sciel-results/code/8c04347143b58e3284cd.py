"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using a Best-Fit Decreasing heuristic with
    a simplified empty-space management per bin.
    """
    W, H, D = container
    container_volume = W * H * D
    
    # Sort boxes by volume descending, but keep track of original indices
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    # Each bin will be represented by a list of placed boxes: (x, y, z, w, h, d)
    bins = []  # list of lists
    bin_volumes = []  # occupied volume per bin
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in the bin without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Check for overlap in all three dimensions
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """
        Find the best position for a box in a given bin.
        Uses a simple grid-based search from origin.
        Returns (x, y, z) or None.
        """
        # Try to find the lowest z, then lowest y, then lowest x position
        best_pos = None
        best_score = None
        
        # Generate candidate positions
        # We'll scan through possible positions
        # To make this efficient, we'll use the existing placed boxes to generate candidates
        
        # Candidate x positions: 0 and right edge of each existing box
        xs = [0]
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            xs.append(px + pw)
        xs = sorted(set([x for x in xs if x + w <= W]))
        
        # Candidate y positions: 0 and top edge of each existing box
        ys = [0]
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            ys.append(py + ph)
        ys = sorted(set([y for y in ys if y + h <= H]))
        
        # Candidate z positions: 0 and front edge of each existing box
        zs = [0]
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            zs.append(pz + pd)
        zs = sorted(set([z for z in zs if z + d <= D]))
        
        for z in zs:
            for y in ys:
                for x in xs:
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        # Score: prefer lower z, then lower y, then lower x
                        score = z * W * H + y * W + x
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    # Place each box
    placements = [None] * len(boxes)
    
    for idx, w, h, d in indexed_boxes:
        placed = False
        
        # Try to fit in existing bins first (best-fit: minimize wasted space)
        best_bin = -1
        best_pos = None
        best_waste = float('inf')
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                # Calculate how "tight" this fit is (lower is better)
                # Use the position as a proxy for how well it fits
                x, y, z = pos
                # A box placed at lower coordinates is generally better
                waste = z * W * H + y * W + x
                if waste < best_waste:
                    best_waste = waste
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            bin_volumes[best_bin] += w * h * d
            placements[idx] = (best_bin, x, y, z, w, h, d)
            placed = True
        else:
            # Need a new bin
            b_idx = len(bins)
            bins.append([])
            bin_volumes.append(0)
            x, y, z = 0, 0, 0
            bins[b_idx].append((x, y, z, w, h, d))
            bin_volumes[b_idx] = w * h * d
            placements[idx] = (b_idx, x, y, z, w, h, d)
            placed = True
    
    return placements
# EVOLVE-BLOCK-END
