"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D if container_vol > 0 else 1
    
    # Sort boxes by volume descending, then by max dimension, to place large boxes first
    # This helps in reducing fragmentation
    indexed_boxes = [(vol, max_dim, idx, box) for idx, box in enumerate(boxes) 
                     for vol in [box[0]*box[1]*box[2]] 
                     for max_dim in [max(box)]]
    indexed_boxes.sort(key=lambda x: (-x[0], -x[1], x[2]))
    
    # We'll manage a list of bins. Each bin has a 3D numpy boolean array for occupancy.
    bins = []
    # placements_by_idx: list of (bin_id, x, y, z, w, h, d) for each original box index
    placements_by_idx = [None] * len(boxes)
    
    def get_bin_volume(bin_id):
        return bins[bin_id]['vol']
    
    def try_place_in_bin(bin_id, w, h, d):
        """
        Try to place a box of dimensions (w, h, d) in the given bin.
        Returns (x, y, z) if placed, or None if not possible.
        Uses a simple scan for the first valid position (bottom-left-front).
        """
        occ = bins[bin_id]['occ']
        # Scan all possible positions
        for z in range(D - d + 1):
            for y in range(H - h + 1):
                for x in range(W - w + 1):
                    # Check if the sub-cube is free
                    if not occ[x:x+w, y:y+h, z:z+d].any():
                        occ[x:x+w, y:y+h, z:z+d] = True
                        bins[bin_id]['vol'] += w * h * d
                        return (x, y, z)
        return None
    
    # For efficiency with small grids, the above is fine. If grids are large, we might need better heuristics.
    # Given 40 boxes, the grid size is likely manageable.
    
    for _, _, orig_idx, (w, h, d) in indexed_boxes:
        placed = False
        
        # Try to place in existing bins first
        # Sort existing bins by how full they are (to pack tighter) or just iterate
        # To minimize number of bins, we should try to fit into the bin that is most full? 
        # Or the bin that has the most space? 
        # Actually, to minimize B (number of bins), we want to fill bins as much as possible.
        # A common heuristic is "Best Fit Decreasing" (BFD).
        # Let's try to find the bin where the box fits with the least remaining space (tightest fit).
        
        best_bin_id = -1
        best_remaining = float('inf')
        best_pos = None
        
        for b_id in range(len(bins)):
            pos = try_place_in_bin(b_id, w, h, d)
            if pos is not None:
                # We placed it. Now we need to check if this is the "best" fit.
                # But try_place_in_bin already modified the bin.
                # We need to undo if it's not the best fit.
                # This is inefficient. Let's restructure.
                
                # Undo placement
                x, y, z = pos
                occ = bins[b_id]['occ']
                occ[x:x+w, y:y+h, z:z+d] = False
                bins[b_id]['vol'] -= w * h * d
                
                # Calculate remaining space in this bin
                remaining = container_vol - bins[b_id]['vol']
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin_id = b_id
                    best_pos = pos
        
        if best_bin_id != -1:
            # Re-place in the best bin
            x, y, z = best_pos
            occ = bins[best_bin_id]['occ']
            occ[x:x+w, y:y+h, z:z+d] = True
            bins[best_bin_id]['vol'] += w * h * d
            placements_by_idx[orig_idx] = (best_bin_id, x, y, z, w, h, d)
            placed = True
        
        if not placed:
            # Open a new bin
            b_id = len(bins)
            occ = np.zeros((W, H, D), dtype=bool)
            bins.append({'occ': occ, 'vol': 0})
            
            # Place at origin (0,0,0)
            occ[0:w, 0:h, 0:d] = True
            bins[b_id]['vol'] = w * h * d
            placements_by_idx[orig_idx] = (b_id, 0, 0, 0, w, h, d)
    
    # Return placements in original order
    return placements_by_idx
# EVOLVE-BLOCK-END
