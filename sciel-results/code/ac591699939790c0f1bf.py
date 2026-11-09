"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Deadline guard: kill after 115s to be safe under 120s limit
    t0 = time.time()
    DEADLINE = 115.0

    # Sort boxes by volume descending, then by max dimension descending
    # This helps pack large items first for better bin utilization
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: (-(t[1][0]*t[1][1]*t[1][2]), -max(t[1])))
    sorted_boxes = [(i, w, h, d) for i, (w, h, d) in indexed]

    # We'll maintain a list of bins, each with a list of placed boxes (x,y,z,w,h,d)
    bins = []  # list of lists of (x, y, z, w, h, d)

    def get_bin_volume_used(bin_items):
        return sum(w*h*d for (x,y,z,w,h,d) in bin_items)

    def can_place(bin_items, x, y, z, w, h, d):
        # Check if box at (x,y,z) with dims (w,h,d) fits in container
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing items
        for (ox, oy, oz, ow, oh, od) in bin_items:
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def find_position(bin_items, w, h, d):
        # Try to find a position for box (w,h,d) in this bin
        # Strategy: try to place at lowest z, then lowest y, then lowest x
        # First, collect all candidate positions
        # Candidate x positions: 0, and right edge of each existing box
        # Candidate y positions: 0, and top edge of each existing box
        # Candidate z positions: 0, and front edge of each existing box
        
        if not bin_items:
            return (0, 0, 0)
        
        # Build candidate coordinates
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        for (ox, oy, oz, ow, oh, od) in bin_items:
            x_candidates.append(ox + ow)
            y_candidates.append(oy + oh)
            z_candidates.append(oz + od)
        
        # Also add W, H, D as boundaries (but we check bounds in can_place)
        x_candidates = sorted(set(x_candidates))
        y_candidates = sorted(set(y_candidates))
        z_candidates = sorted(set(z_candidates))
        
        # Try all combinations, preferring lower z, then lower y, then lower x
        # Sort z_candidates, y_candidates, x_candidates
        best_pos = None
        for z in sorted(z_candidates):
            if z + d > D:
                continue
            for y in sorted(y_candidates):
                if y + h > H:
                    continue
                for x in sorted(x_candidates):
                    if x + w > W:
                        continue
                    if can_place(bin_items, x, y, z, w, h, d):
                        return (x, y, z)
        
        # If no position found via candidate edges, try a grid search
        # This is a fallback; should rarely be needed
        # Actually, the edge-based approach should suffice for axis-aligned boxes
        # Let's try a more thorough search if needed
        return None

    def try_place_in_bin(bin_items, w, h, d):
        """Try to place box in given bin. Returns (x,y,z) or None."""
        pos = find_position(bin_items, w, h, d)
        return pos

    def new_bin():
        bins.append([])
        return len(bins) - 1

    # Greedy placement with Best-Fit Decreasing
    placements = [None] * n  # (bin_id, x, y, z, w, h, d)
    
    for idx, w, h, d in sorted_boxes:
        if time.time() > DEADLINE:
            # Emergency: place remaining boxes in new bins to ensure validity
            # This is a fallback; normally we shouldn't hit this
            for j in range(idx, n):
                orig_idx, ow, oh, od = sorted_boxes[j]
                b = len(bins)
                new_bin()
                bins[-1].append((0, 0, 0, ow, oh, od))
                placements[orig_idx] = (b, 0, 0, 0, ow, oh, od)
            break
        
        placed = False
        # Try existing bins first, sorted by current volume used (best fit)
        # For best-fit, we want the bin where the box fits with least remaining space
        # But since we just need a valid placement, let's try bins in order
        # Actually, let's try to find the bin that minimizes the "waste" after placement
        
        # Get list of bin indices sorted by current volume (ascending for best-fit? 
        # Actually best-fit means the bin that has the least space left after placing)
        # Let's just try each bin and pick the one where the box fits
        # To improve, sort bins by current fill ratio descending (most full first)
        
        bin_indices = list(range(len(bins)))
        # Sort by current volume used descending (most packed bins first)
        bin_indices.sort(key=lambda bi: -get_bin_volume_used(bins[bi]))
        
        best_bin = -1
        best_pos = None
        best_remaining = float('inf')
        
        for bi in bin_indices:
            if time.time() > DEADLINE:
                break
            pos = try_place_in_bin(bins[bi], w, h, d)
            if pos is not None:
                x, y, z = pos
                # Calculate remaining space in this bin after placement
                # Approximate: container volume - (current used + box volume)
                current_used = get_bin_volume_used(bins[bi])
                remaining = W * H * D - (current_used + w * h * d)
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = bi
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            placements[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # New bin
            b = len(bins)
            new_bin()
            bins[b].append((0, 0, 0, w, h, d))
            placements[idx] = (b, 0, 0, 0, w, h, d)
    
    # Renumber bins to be consecutive starting from 0
    # Map old bin labels to new consecutive labels
    old_to_new = {}
    new_labels = []
    for b in sorted(set(pl[p][0] for p in range(n))):
        old_to_new[b] = len(new_labels)
        new_labels.append(b)
    
    # Apply renumbering
    for i in range(n):
        if placements[i] is not None:
            b, x, y, z, w, h, d = placements[i]
            placements[i] = (old_to_new[b], x, y, z, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
