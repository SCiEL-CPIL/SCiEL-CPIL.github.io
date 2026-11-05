"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Each bin has: grid (3D boolean), items list, volume
    bins = []  # list of dicts with 'grid', 'items', 'vol'
    
    def create_bin():
        return {
            'grid': np.zeros((W, H, D), dtype=bool),
            'items': [],
            'vol': 0
        }
    
    def can_place(grid, x, y, z, w, h, d):
        """Check if box fits at position using grid."""
        if x + w > W or y + h > H or z + d > D:
            return False
        return not np.any(grid[x:x+w, y:y+h, z:z+d])
    
    def place_in_grid(grid, x, y, z, w, h, d):
        """Mark box as occupied in grid."""
        grid[x:x+w, y:y+h, z:z+d] = True
    
    def find_best_position(bin_data, w, h, d):
        """
        Find the best position using grid-based candidate generation.
        Generate candidates from existing box boundaries and scan for valid spots.
        """
        items = bin_data['items']
        grid = bin_data['grid']
        
        # Start with (0,0,0)
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Generate candidates from box boundaries
        for (bx, by, bz, bw, bh, bd, _) in items:
            # Positions adjacent to each face
            if bx + bw < W:
                candidates.add((bx + bw, by, bz))
            if by + bh < H:
                candidates.add((bx, by + bh, bz))
            if bz + bd < D:
                candidates.add((bx, by, bz + bd))
            
            # Corner combinations
            if bx + bw < W and by + bh < H:
                candidates.add((bx + bw, by + bh, bz))
            if bx + bw < W and bz + bd < D:
                candidates.add((bx + bw, by, bz + bd))
            if by + bh < H and bz + bd < D:
                candidates.add((bx, by + bh, bz + bd))
            if bx + bw < W and by + bh < H and bz + bd < D:
                candidates.add((bx + bw, by + bh, bz + bd))
        
        # Also add some systematic scan points for better coverage
        # Scan along axes at key positions
        for x in [0] + [bx + bw for (_, _, _, bw, _, _, _) in [(it[0], it[1], it[2], it[3], it[4], it[5], it[6]) for it in items] if bx + bw < W]:
            for y in [0] + [by + bh for (_, by, _, _, bh, _, _) in [(it[0], it[1], it[2], it[3], it[4], it[5], it[6]) for it in items] if by + bh < H]:
                for z in [0] + [bz + bd for (_, _, bz, _, _, bd, _) in [(it[0], it[1], it[2], it[3], it[4], it[5], it[6]) for it in items] if bz + bd < D]:
                    if x + w <= W and y + h <= H and z + d <= D:
                        candidates.add((x, y, z))
        
        best_pos = None
        best_score = float('inf')
        
        for cx, cy, cz in candidates:
            if can_place(grid, cx, cy, cz, w, h, d):
                # Prefer positions that minimize wasted space:
                # Lower x+y+z is better (packs toward origin)
                # Also prefer positions that leave less fragmented space
                score = cx + cy + cz
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
        
        return best_pos
    
    deadline = time.time() + 145.0  # Safety margin
    
    for idx, (orig_i, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            # Emergency fallback
            for rem_i, rw, rh, rd, rv in indexed_boxes[idx:]:
                new_bin = create_bin()
                place_in_grid(new_bin['grid'], 0, 0, 0, rw, rh, rd)
                new_bin['items'].append((0, 0, 0, rw, rh, rd, rem_i))
                new_bin['vol'] = rw * rh * rd
                bins.append(new_bin)
            break
        
        best_bin_idx = -1
        best_pos = None
        best_bin_vol = -1
        
        # Try to fit in existing bins
        for b_idx, bin_data in enumerate(bins):
            pos = find_best_position(bin_data, w, h, d)
            if pos is not None:
                # Prefer bins with higher volume (fuller bins) to minimize B
                if bin_data['vol'] > best_bin_vol:
                    best_bin_vol = bin_data['vol']
                    best_bin_idx = b_idx
                    best_pos = pos
        
        if best_bin_idx == -1:
            # Create new bin
            new_bin = create_bin()
            place_in_grid(new_bin['grid'], 0, 0, 0, w, h, d)
            new_bin['items'].append((0, 0, 0, w, h, d, orig_i))
            new_bin['vol'] = vol
            bins.append(new_bin)
        else:
            bin_data = bins[best_bin_idx]
            place_in_grid(bin_data['grid'], best_pos[0], best_pos[1], best_pos[2], w, h, d)
            bin_data['items'].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, orig_i))
            bin_data['vol'] += vol
    
    # Construct final placements
    placements = [None] * n
    
    for b_idx, bin_data in enumerate(bins):
        for item in bin_data['items']:
            x, y, z, w, h, d, orig_idx = item
            placements[orig_idx] = (b_idx, x, y, z, w, h, d)
    
    # Safety: ensure all boxes are placed
    for i in range(n):
        if placements[i] is None:
            # This should not happen, but handle gracefully
            new_bin_idx = len(bins)
            bins.append({
                'grid': np.zeros((W, H, D), dtype=bool),
                'items': [(0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2], i)],
                'vol': boxes[i][0] * boxes[i][1] * boxes[i][2]
            })
            placements[i] = (new_bin_idx, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return placements
# EVOLVE-BLOCK-END
