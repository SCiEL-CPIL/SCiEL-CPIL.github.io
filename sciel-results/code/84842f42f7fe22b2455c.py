"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Precompute rotations, prefer orientations with smaller Z (height) for shelf packing
    def get_rotations_pref_height(w, h, d):
        seen = set()
        rots = []
        for perm in permutations([w, h, d]):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        # Sort rotations by Z (last element) ascending, then X, then Y
        # This prefers laying the box flat
        rots.sort(key=lambda r: (r[2], r[0], r[1]))
        return rots
    
    box_rotations = [get_rotations_pref_height(w, h, d) for w, h, d in boxes]
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Sort boxes by volume descending? Or by height?
    # For shelf packing, sorting by height (min dimension) or volume is common.
    # Let's sort by volume descending to place big boxes first.
    sorted_indices = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    
    placements = [None] * n
    # shelves: list of dicts
    # Each shelf: { 'z_start': float, 'max_h': float, 'items': [(x, y, w, d, box_idx)] }
    shelves = []
    
    def get_2d_candidates(shelf):
        """Generate candidate x,y positions for a new box in a shelf."""
        cands = [(0, 0)]
        for (x, y, w, d, idx) in shelf['items']:
            cands.append((x + w, y))
            cands.append((x, y + d))
        # Add some random/other points? No, keep it deterministic and fast.
        # Also check boundaries
        return cands

    def can_place_2d(x, y, w, d, shelf):
        if x < 0 or y < 0:
            return False
        if x + w > W or y + d > D:
            return False
        for (ex, ey, ew, ed, idx) in shelf['items']:
            if (x < ex + ew and ex < x + w and
                y < ey + ed and ey < y + d):
                return False
        return True

    for idx in sorted_indices:
        w, h, d = boxes[idx]
        placed = False
        # Try to place in existing shelves, then create new
        # We should try to place in the shelf that fits best?
        # Or just the first one that fits?
        # Let's try all existing shelves, pick the one that minimizes the increase in shelf height or just the first fit.
        best_shelf_idx = -1
        best_pos = None
        best_rot = None
        best_cost = float('inf')
        
        # Try existing shelves
        for s_idx, shelf in enumerate(shelves):
            # If the box is taller than the current max height, we would need to expand the shelf.
            # Expanding a shelf increases the total height used, which is bad if we have many shelves.
            # But it might be better than starting a new shelf.
            # Let's just try to fit. If it fits, we might need to update shelf['max_h'].
            
            for rot in box_rotations[idx]:
                rw, rh, rd = rot
                # rh is the height (Z)
                # Check if it fits in 2D
                for (px, py) in get_2d_candidates(shelf):
                    if can_place_2d(px, py, rw, rd, shelf):
                        # Cost heuristic:
                        # 1. Prefer shelves that are already tall (no height increase)
                        # 2. Prefer lower shelves?
                        # 3. Prefer tighter fit?
                        
                        # If rh > shelf['max_h'], we must expand.
                        # Cost = increase in height * 1000 + px + py
                        height_increase = max(0, rh - shelf['max_h'])
                        cost = height_increase * 10000 + px * 10 + py
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_shelf_idx = s_idx
                            best_pos = (px, py)
                            best_rot = (rw, rh, rd)
                            
        if best_shelf_idx != -1:
            shelf = shelves[best_shelf_idx]
            px, py = best_pos
            rw, rh, rd = best_rot
            # Update shelf max height
            if rh > shelf['max_h']:
                shelf['max_h'] = rh
            # Add to shelf items
            shelf['items'].append((px, py, rw, rd, idx))
            placements[idx] = (0, px, py, shelf['z_start'], rw, rh, rd)
            placed = True
        else:
            # Create new shelf
            # Determine Z start: sum of max_h of all previous shelves
            z_start = sum(s['max_h'] for s in shelves)
            
            # Check if new shelf fits in container height
            if z_start + box_rotations[idx][0][1] > H: # Using min height rotation
                # Might not fit at all, or needs a different rotation
                # Try all rotations to see if any fits in remaining height
                found = False
                for rot in box_rotations[idx]:
                    rw, rh, rd = rot
                    if z_start + rh <= H:
                        # Create shelf with this height
                        new_shelf = {'z_start': z_start, 'max_h': rh, 'items': []}
                        shelves.append(new_shelf)
                        # Place at 0,0
                        new_shelf['items'].append((0, 0, rw, rd, idx))
                        placements[idx] = (0, 0, 0, z_start, rw, rh, rd)
                        found = True
                        break
                if not found:
                    placements[idx] = (-1, 0, 0, 0, w, h, d)
                continue
            else:
                # Use the preferred rotation (smallest height)
                rw, rh, rd = box_rotations[idx][0]
                new_shelf = {'z_start': z_start, 'max_h': rh, 'items': []}
                shelves.append(new_shelf)
                new_shelf['items'].append((0, 0, rw, rd, idx))
                placements[idx] = (0, 0, 0, z_start, rw, rh, rd)
                placed = True

    # Fill in any unplaced boxes
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
# EVOLVE-BLOCK-END
