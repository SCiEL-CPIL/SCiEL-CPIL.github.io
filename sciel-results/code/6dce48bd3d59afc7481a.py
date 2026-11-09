"""Rewritten 3D bin packing using 3D grid occupancy and best-fit placement."""
import numpy as np

def pack(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(0, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Precompute container volume for scoring context (not used directly in placement)
    container_vol = W * H * D
    
    # Bin data: list of (occupancy_grid, used_volume)
    bins = []
    
    def get_bin_grid(b):
        if b < len(bins):
            return bins[b][0]
        return None
    
    def add_bin():
        occ = np.zeros((W, H, D), dtype=bool)
        bins.append((occ, 0))
        return len(bins) - 1
    
    def try_place_in_bin(b, w, h, d):
        """
        Try to find a feasible position in bin b for a box of size (w, h, d).
        Returns (x, y, z) if found, else None.
        Uses best-fit: minimize (x + y + z) among all feasible positions.
        """
        if b >= len(bins):
            return None
        occ, _ = bins[b]
        
        best_pos = None
        best_score = float('inf')
        
        # Iterate over all possible positions
        for x in range(0, W - w + 1):
            for y in range(0, H - h + 1):
                for z in range(0, D - d + 1):
                    # Check if this region is free
                    # Using slicing: occ[x:x+w, y:y+h, z:z+d]
                    if not np.any(occ[x:x+w, y:y+h, z:z+d]):
                        score = x + y + z  # Best-fit: prefer lower coordinates
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def place_in_bin(b, x, y, z, w, h, d):
        """Mark the region as occupied in bin b."""
        occ, vol = bins[b]
        occ[x:x+w, y:y+h, z:z+d] = True
        bins[b] = (occ, vol + w * h * d)
    
    def get_last_bin_occupied_fraction():
        """Return occupied volume fraction of the last (highest labeled) bin."""
        if not bins:
            return 0.0
        max_bin = max(b for b, _, _, _, _, _, _ in [(0,0,0,0,0,0,0)] if False) # placeholder
        # Actually, we track bins list; the last bin used is the one with the highest index that has content?
        # The task says: "largest label defines the last bin".
        # We need to find the max bin index used.
        # But in our loop, we just append bins. The "last bin" for scoring is the one with the highest index.
        # However, if we skip bins, we need to be careful.
        # For simplicity, we assume we use bins 0..n-1 consecutively.
        # The score uses the last bin's occupied volume fraction.
        # Let's just return the fraction of the last bin in our list.
        if not bins:
            return 0.0
        _, vol = bins[-1]
        return vol / container_vol if container_vol > 0 else 0.0
    
    placements = []
    
    for w, h, d in boxes:
        placed = False
        
        # Try existing bins first
        for b in range(len(bins)):
            pos = try_place_in_bin(b, w, h, d)
            if pos is not None:
                x, y, z = pos
                place_in_bin(b, x, y, z, w, h, d)
                placements.append((b, x, y, z, w, h, d))
                placed = True
                break
        
        # If not placed in any existing bin, create a new one
        if not placed:
            b = add_bin()
            # Try to place in the new bin
            pos = try_place_in_bin(b, w, h, d)
            if pos is not None:
                x, y, z = pos
                place_in_bin(b, x, y, z, w, h, d)
                placements.append((b, x, y, z, w, h, d))
            else:
                # This should not happen if w<=W, h<=H, d<=D
                # If it does, we have a problem. But assume boxes fit in container.
                # Fallback: place at (0,0,0) and hope for the best (invalid if overlapping, but this case shouldn't occur)
                place_in_bin(b, 0, 0, 0, w, h, d)
                placements.append((b, 0, 0, 0, w, h, d))
    
    return placements
