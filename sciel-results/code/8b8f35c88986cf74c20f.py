"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Deadline guard: aim to finish well under 120s
    deadline = time.time() + 110.0
    
    # Index boxes with original indices
    indexed = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    
    # Sort by volume descending for Best Fit Decreasing
    indexed.sort(key=lambda x: x[4], reverse=True)
    
    # We'll maintain a list of bins, each with a list of placed boxes (x, y, z, w, h, d)
    # And we'll track "corner" points (candidate positions) for each bin
    # A corner is (x, y, z) where a box could potentially be placed
    
    bins = []  # list of lists of (x, y, z, w, h, d)
    bin_corners = []  # list of sets of (x, y, z)
    
    def can_place(bin_idx, px, py, pz, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) can be placed at (px, py, pz) in bin bin_idx."""
        if px + bw > W or py + bh > H or pz + bd > D:
            return False
        placed = bins[bin_idx]
        for (ox, oy, oz, ow, oh, od) in placed:
            # Check overlap: two boxes overlap if they intersect in all 3 dimensions
            if (px < ox + ow and px + bw > ox and
                py < oy + oh and py + bh > oy and
                pz < oz + od and pz + bd > oz):
                return False
        return True
    
    def get_corners(bin_idx):
        """Get all candidate corner positions for a bin."""
        corners = {(0, 0, 0)}
        for (ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            # For each placed box, add corners that are "exposed"
            # The key idea: a valid position must have its x aligned with the right edge of some box
            # and its y aligned with the top edge of some box (in the x-projection)
            # and its z aligned with the front edge of some box (in the xy-projection)
            
            # Simpler approach: just add all 8 corners of each box as candidates,
            # plus (0,0,0). This is a standard heuristic.
            corners.add((ox + ow, oy, oz))
            corners.add((ox, oy + oh, oz))
            corners.add((ox, oy, oz + od))
            corners.add((ox + ow, oy + oh, oz))
            corners.add((ox + ow, oy, oz + od))
            corners.add((ox, oy + oh, oz + od))
            corners.add((ox + ow, oy + oh, oz + od))
        return corners
    
    # For efficiency, we'll use a different approach:
    # For each box, try to place it in existing bins first, then open a new bin.
    # For each bin, we generate candidate positions and try them.
    
    # To make this faster, we'll limit the number of candidate positions per bin
    # by only considering positions that are "dominated" (i.e., we prefer lower x, y, z)
    
    placements = [None] * n  # Will store (bin, x, y, z, w, h, d) for each original box
    
    for i, (orig_idx, w, h, d, vol) in enumerate(indexed):
        if time.time() > deadline - 5.0:
            # Emergency: just place remaining boxes in new bins to ensure validity
            for j in range(i, n):
                orig_idx_j, w_j, h_j, d_j, _ = indexed[j]
                b = len(bins)
                bins.append([(0, 0, 0, w_j, h_j, d_j)])
                bin_corners.append({(0, 0, 0)})
                placements[orig_idx_j] = (b, 0, 0, 0, w_j, h_j, d_j)
            break
        
        placed = False
        # Try existing bins first
        for b in range(len(bins)):
            if placed:
                break
            # Get candidate positions
            corners = get_corners(b)
            # Sort corners by (x, y, z) for deterministic ordering
            corners = sorted(corners)
            for (px, py, pz) in corners:
                if can_place(b, px, py, pz, w, h, d):
                    # Place the box
                    bins[b].append((px, py, pz, w, h, d))
                    placements[orig_idx] = (b, px, py, pz, w, h, d)
                    placed = True
                    break
        
        if not placed:
            # Open a new bin
            b = len(bins)
            if w <= W and h <= H and d <= D:
                bins.append([(0, 0, 0, w, h, d)])
                bin_corners.append({(0, 0, 0)})
                placements[orig_idx] = (b, 0, 0, 0, w, h, d)
            else:
                # Box doesn't fit in any bin - this shouldn't happen for valid inputs
                # Place it in a new bin at (0,0,0) anyway (will be invalid but we need to return something)
                bins.append([(0, 0, 0, w, h, d)])
                bin_corners.append({(0, 0, 0)})
                placements[orig_idx] = (b, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
