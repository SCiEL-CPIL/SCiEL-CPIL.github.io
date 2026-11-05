"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Each bin stores:
    #   'items': list of (x, y, z, w, h, d, orig_idx)
    #   'vol': total volume of items in bin
    bins = []
    
    def create_bin():
        return {'items': [], 'vol': 0}
    
    def fits(bin_items, x, y, z, w, h, d):
        """Check bounds and non-overlap with all items in bin."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            # Overlap check: intervals [x, x+w) and [ox, ox+ow) overlap iff
            # x < ox+ow and ox < x+w (similarly for y, z)
            if (x < ox + ow and ox < x + w and
                y < oy + oh and oy < y + h and
                z < oz + od and oz < z + d):
                return False
        return True
    
    def find_position(bin_data, w, h, d):
        """
        Find best position using candidate points from box boundaries.
        Returns (x, y, z) or None.
        """
        items = bin_data['items']
        
        if not items:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Build candidate x, y, z coordinates from existing box boundaries
        xs = {0}
        ys = {0}
        zs = {0}
        for (ox, oy, oz, ow, oh, od, _) in items:
            if ox + ow < W:
                xs.add(ox + ow)
            if oy + oh < H:
                ys.add(oy + oh)
            if oz + od < D:
                zs.add(oz + od)
        
        # Also add the max possible positions
        if W - w >= 0:
            xs.add(W - w)
        if H - h >= 0:
            ys.add(H - h)
        if D - d >= 0:
            zs.add(D - d)
        
        best_pos = None
        best_score = float('inf')
        
        for cx in xs:
            if cx + w > W:
                continue
            for cy in ys:
                if cy + h > H:
                    continue
                for cz in zs:
                    if cz + d > D:
                        continue
                    if fits(items, cx, cy, cz, w, h, d):
                        # Score: prefer positions that minimize wasted space
                        # Use a weighted sum that favors filling gaps
                        score = cx + cy + cz
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
        
        return best_pos
    
    deadline = time.time() + 145.0  # Safety margin
    
    for idx, (orig_i, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            # Emergency fallback: place each remaining box in its own new bin
            for rem_i, rw, rh, rd, rv in indexed_boxes[idx:]:
                bins.append({'items': [(0, 0, 0, rw, rh, rd, rem_i)], 'vol': rv})
            break
        
        best_bin_idx = -1
        best_pos = None
        best_bin_vol = -1
        
        # Try to fit in existing bins
        for b_idx, bin_data in enumerate(bins):
            pos = find_position(bin_data, w, h, d)
            if pos is not None:
                # Prefer bins with higher volume (fuller bins) to minimize B
                if bin_data['vol'] > best_bin_vol:
                    best_bin_vol = bin_data['vol']
                    best_bin_idx = b_idx
                    best_pos = pos
        
        if best_bin_idx == -1:
            # Create new bin
            new_bin = create_bin()
            new_bin['items'].append((0, 0, 0, w, h, d, orig_i))
            new_bin['vol'] = vol
            bins.append(new_bin)
        else:
            bin_data = bins[best_bin_idx]
            bin_data['items'].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, orig_i))
            bin_data['vol'] += vol
    
    # Construct final placements indexed by original box index
    placements = [None] * n
    
    for b_idx, bin_data in enumerate(bins):
        for item in bin_data['items']:
            x, y, z, w, h, d, orig_idx = item
            placements[orig_idx] = (b_idx, x, y, z, w, h, d)
    
    # Safety: ensure all boxes are placed
    for i in range(n):
        if placements[i] is None:
            new_bin_idx = len(bins)
            bw, bh, bd = boxes[i]
            bins.append({'items': [(0, 0, 0, bw, bh, bd, i)], 'vol': bw * bh * bd})
            placements[i] = (new_bin_idx, 0, 0, 0, bw, bh, bd)
    
    return placements
# EVOLVE-BLOCK-END
