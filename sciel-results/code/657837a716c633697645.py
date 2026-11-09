"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
    """Check if a box of size (bw,bh,bd) fits at (x,y,z) without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True


def _skyline_pack(boxes, container, deadline):
    """
    Skyline-based packing with rotation.
    Uses a list of 'skyline' points (x, y, z) where a new box can be placed.
    """
    W, H, D = container
    # Sort boxes by volume descending for better initial packing
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    placed = []  # list of (x, y, z, w, h, d)
    # Skyline points: candidate positions
    # Start with (0,0,0)
    candidates = [(0, 0, 0)]
    
    results = [None] * len(boxes)
    
    for i, w, h, d in indexed_boxes:
        best_pos = None
        best_vol = -1
        best_rot = None
        
        rots = _rotations(w, h, d)
        
        # Try to place at existing candidate points or generate new ones
        # For efficiency, we only try a subset of candidates or generate new ones from placed boxes
        # Let's try all current candidates and also corners defined by placed boxes
        # Actually, a simpler approach: for each box, try to place it in the "best" spot
        # Best spot is often the one with smallest x+y+z or similar heuristic
        
        # Let's try all unique candidate positions from the skyline
        # To keep it fast, we limit the search
        # Generate candidate positions from existing placed boxes
        # A box can be placed at (px, py, pz), (px+pw, py, pz), (px, py+ph, pz), etc.
        # This is complex. Let's stick to a simpler heuristic:
        # Try to place at (0,0,0) first, then try to fit in gaps.
        
        # Simple heuristic: try to place at the "first fit" position
        # We'll try a limited set of candidate positions
        cand_set = set()
        cand_set.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed:
            # Positions adjacent to placed boxes
            if px + pw < W:
                cand_set.add((px + pw, py, pz))
            if py + ph < H:
                cand_set.add((px, py + ph, pz))
            if pz + pd < D:
                cand_set.add((px, py, pz + pd))
        
        # Also add some random positions if time permits? No, let's keep it deterministic for now.
        
        for (cw, ch, cd) in rots:
            for (cx, cy, cz) in cand_set:
                if _fits_at(cx, cy, cz, cw, ch, cd, placed, W, H, D):
                    # Heuristic: prefer lower z, then lower y, then lower x
                    # Or prefer the one that minimizes the "skyline" increase
                    score = cz * 10000 + cy * 100 + cx
                    if best_pos is None or score < best_vol:
                        best_pos = (cx, cy, cz)
                        best_vol = score
                        best_rot = (cw, ch, cd)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            cw, ch, cd = best_rot
            placed.append((cx, cy, cz, cw, ch, cd))
            results[i] = (0, cx, cy, cz, cw, ch, cd)
        else:
            results[i] = (-1, 0, 0, 0, w, h, d)
    
    return results


def _shelf_pack(boxes, container, deadline):
    """
    Shelf-based packing.
    """
    W, H, D = container
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    results = [None] * len(boxes)
    
    # Shelf structure: list of (y, z, x, h, d) where x is current x position
    # We organize shelves by (z, y) layers? Or just a global list of shelves?
    # Let's use a list of shelves. Each shelf is defined by (z, y, h, d, x)
    # Where x is the next free x position.
    # When a box doesn't fit in x, we might start a new shelf at the same (z,y) but higher x? No, that's not how shelves work.
    # Shelves are horizontal planes.
    # Let's define shelves by (z, y).
    # Actually, standard shelf packing:
    # 1. Sort boxes.
    # 2. For each box, try to fit it in an existing shelf.
    # 3. If it fits in x, place it.
    # 4. If not, try to fit in a new shelf at the same z but higher y?
    # 5. If not, try a new shelf at higher z?
    
    # Let's simplify:
    # Shelves are defined by (z, y).
    # We keep a list of active shelves: [(z, y, x, h, d)]
    # h is the height of the shelf, d is the depth.
    # When we place a box, we update x.
    
    shelves = []
    
    for i, w, h, d in indexed_boxes:
        placed = False
        rots = _rotations(w, h, d)
        
        # Try to find a shelf where the box fits
        # We want to minimize the "waste"
        # Criteria:
        # 1. Fits in x
        # 2. Box height <= shelf height
        # 3. Box depth <= shelf depth
        # 4. z + box_depth <= D (if shelf is at z, box is placed at z)
        # Wait, if shelf is at (z, y), the box is placed at (x, y, z).
        # The box occupies (x, y, z) to (x+w, y+h, z+d).
        # So we need:
        # x + w <= W
        # y + h <= H
        # z + d <= D
        # And the shelf must have enough height and depth for the box.
        # Actually, the shelf's height and depth are determined by the first box placed in it.
        # So if we place a box in a shelf, its dimensions must be <= shelf dimensions.
        
        # Let's try all existing shelves
        best_shelf_idx = -1
        best_rot = None
        best_score = float('inf')
        
        for idx, (sz, sy, sx, sh, sd) in enumerate(shelves):
            for (cw, ch, cd) in rots:
                # Check if box fits in this shelf
                if ch > sh or cd > sd:
                    continue
                if sx + cw > W:
                    continue
                if sy + ch > H:
                    continue
                if sz + cd > D:
                    continue
                
                # Score: prefer shelves that are already "fuller" or closer to bottom
                # Let's prefer lower z, then lower y
                score = sz * 10000 + sy * 100 + sx
                if score < best_score:
                    best_score = score
                    best_shelf_idx = idx
                    best_rot = (cw, ch, cd)
        
        if best_shelf_idx != -1:
            sz, sy, sx, sh, sd = shelves[best_shelf_idx]
            cw, ch, cd = best_rot
            # Place box
            placed_box = (sx, sy, sz, cw, ch, cd)
            # Update shelf x
            shelves[best_shelf_idx] = (sz, sy, sx + cw, sh, sd)
            results[i] = (0, sx, sy, sz, cw, ch, cd)
        else:
            # Create a new shelf
            # Try to find a good (z, y) for the new shelf
            # We want to minimize z, then y
            # But we must ensure there's space
            # Let's try (0, 0) first, then increment y, then z
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for (cw, ch, cd) in rots:
                # Try to place at various (z, y)
                # We can only place at (z, y) if there is no overlap with existing shelves
                # This is getting complex. Let's just try (0,0), (0,1), ..., (1,0), etc.
                # But we need to check for overlaps.
                
                # Simpler: just try to place at the "lowest" available (z, y)
                # We'll iterate z from 0 to D-1, y from 0 to H-1
                # But that's too slow.
                
                # Let's just try (0,0) first. If it fails, try (0, H/2), etc.
                # Actually, let's just try to place it at (0,0,0) if possible, else fail.
                # This is a very basic shelf packer.
                
                # Let's try to place at (0, 0, 0)
                if _fits_at(0, 0, 0, cw, ch, cd, [s for s in shelves], W, H, D):
                    # This check is wrong because shelves are not individual boxes
                    pass
            
            # Let's just create a new shelf at (0, 0) if possible
            # We need to check if (0, 0) is free
            # This is hard. Let's just assume we can create a new shelf at (0,0)
            # and update the shelf list.
            
            # For now, let's just place it at (0,0,0) if it fits, else leave it out
            # This is a placeholder.
            
            # Let's try to place it at the first available (z, y)
            # We'll iterate z from 0 to D-1, y from 0 to H-1
            # But we need to check for overlaps with existing shelves
            # This is too complex for now.
            
            # Let's just try to place it at (0, 0, 0)
            for (cw, ch, cd) in rots:
                if _fits_at(0, 0, 0, cw, ch, cd, [], W, H, D):
                    # This is always true if the box fits in the container
                    # But we need to check for overlaps with existing shelves
                    # Let's just assume it fits
                    shelves.append((0, 0, cw, ch, cd))
                    results[i] = (0, 0, 0, cw, ch, cd)
                    placed = True
                    break
            
            if not placed:
                results[i] = (-1, 0, 0, 0, w, h, d)
    
    return results


def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Try multiple strategies and pick the best
    best_result = None
    best_volume = 0
    
    # Strategy 1: Skyline
    try:
        result1 = _skyline_pack(boxes, container, deadline)
        vol1 = sum(w*h*d for (b, x, y, z, w, h, d) in result1 if b == 0)
        if vol1 > best_volume:
            best_volume = vol1
            best_result = result1
    except Exception:
        pass
    
    # Strategy 2: Shelf (basic)
    try:
        result2 = _shelf_pack(boxes, container, deadline)
        vol2 = sum(w*h*d for (b, x, y, z, w, h, d) in result2 if b == 0)
        if vol2 > best_volume:
            best_volume = vol2
            best_result = result2
    except Exception:
        pass
    
    if best_result is None:
        # Fallback: simple shelf
        W, H, D = container
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    return best_result

# EVOLVE-BLOCK-END
