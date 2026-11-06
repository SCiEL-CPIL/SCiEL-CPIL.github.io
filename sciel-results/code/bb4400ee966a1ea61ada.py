"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Generate all rotations for each box
    # Each rotation: (w, h, d)
    def get_rotations(w, h, d):
        rots = set()
        for perm in permutations([w, h, d]):
            rots.add(perm)
        return list(rots)
    
    # Precompute rotations for all boxes
    all_rotations = []
    for w, h, d in boxes:
        all_rotations.append(get_rotations(w, h, d))
    
    # Sort boxes by volume descending, then by max dimension descending
    box_vols = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: (-box_vols[i], -max(boxes[i])))
    
    # Try multiple strategies and keep the best
    best_placements = None
    best_volume = 0
    
    # Strategy 1: Shelf packing with rotation selection
    # Strategy 2: Extreme point method
    # Strategy 3: Layer decomposition
    
    def try_shelf_packing(sorted_indices, rotations_map):
        """
        Shelf packing: place boxes in rows (shelves) along x, then stack shelves along y,
        then stack layers along z. For each box, choose the rotation that minimizes height
        in the current shelf context.
        """
        placements = [None] * n
        x, y, z = 0, 0, 0
        shelf_h = 0  # height of current shelf
        layer_d = 0  # depth of current layer
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            rots = rotations_map[idx]
            
            # Choose rotation: minimize height (h component) to keep shelves thin
            # But also consider width and depth fit
            best_rot = None
            best_score = float('inf')
            
            for rw, rh, rd in rots:
                # Check if fits in container
                if rw > W or rh > H or rd > D:
                    continue
                # Score: prefer smaller height (to pack more shelves), then smaller depth
                score = rh * 1000 + rd * 100 + rw
                if score < best_score:
                    best_score = score
                    best_rot = (rw, rh, rd)
            
            if best_rot is None:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                continue
            
            rw, rh, rd = best_rot
            
            # Check if we need a new shelf
            if x + rw > W:
                # Move to next shelf
                y += shelf_h
                x = 0
                shelf_h = 0
            
            # Check if we need a new layer
            if y + rh > H:
                z += layer_d
                y = 0
                x = 0
                shelf_h = 0
                layer_d = 0
            
            # Check if box fits in container depth
            if z + rd > D:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                continue
            
            # Place the box
            placements[idx] = (0, x, y, z, rw, rh, rd)
            x += rw
            shelf_h = max(shelf_h, rh)
            layer_d = max(layer_d, rd)
        
        # Calculate filled volume
        vol = 0
        for p in placements:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        
        return placements, vol
    
    def try_extreme_point(sorted_indices, rotations_map):
        """
        Extreme point method: maintain a set of candidate positions (extreme points),
        place each box at the best extreme point.
        """
        placements = [None] * n
        # Extreme points: list of (x, y, z)
        extreme_points = [(0, 0, 0)]
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            rots = rotations_map[idx]
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rw, rh, rd in rots:
                if rw > W or rh > H or rd > D:
                    continue
                
                # Try all extreme points
                for px, py, pz in extreme_points:
                    nx, ny, nz = px + rw, py + rh, pz + rd
                    if nx > W or ny > H or nz > D:
                        continue
                    
                    # Score: prefer positions closer to origin (to leave room)
                    score = px * 10000 + py * 100 + pz
                    if score < best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is None:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                continue
            
            px, py, pz = best_pos
            rw, rh, rd = best_rot
            
            placements[idx] = (0, px, py, pz, rw, rh, rd)
            
            # Add new extreme points
            new_points = [
                (px + rw, py, pz),
                (px, py + rh, pz),
                (px, py, pz + rd),
                (px + rw, py + rh, pz),
                (px + rw, py, pz + rd),
                (px, py + rh, pz + rd),
                (px + rw, py + rh, pz + rd),
            ]
            
            for np_ in new_points:
                if np_ not in extreme_points:
                    extreme_points.append(np_)
            
            # Limit extreme points to avoid explosion
            if len(extreme_points) > 500:
                # Keep only the most promising ones (closest to origin)
                extreme_points.sort(key=lambda p: p[0] * 10000 + p[1] * 100 + p[2])
                extreme_points = extreme_points[:500]
        
        vol = 0
        for p in placements:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        
        return placements, vol
    
    def try_layer_decomposition(sorted_indices, rotations_map):
        """
        Layer decomposition: discretize height into slabs, pack 2D shelves in each slab.
        """
        # Determine slab heights from box dimensions
        all_h = set()
        for idx in range(n):
            for rw, rh, rd in rotations_map[idx]:
                all_h.add(rh)
        
        if not all_h:
            return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes], 0
        
        # Use unique heights as slab boundaries
        heights = sorted(all_h)
        
        # Try different slab height strategies
        best_pl = None
        best_vol = 0
        
        for slab_h in heights[:10]:  # Limit to top 10 smallest heights
            if slab_h <= 0 or slab_h > H:
                continue
            
            placements = [None] * n
            z = 0
            remaining = list(sorted_indices)
            
            while z + slab_h <= H and remaining:
                # Pack 2D shelf in this slab
                x, y = 0, 0
                shelf_h = 0
                placed_in_slab = []
                
                for idx in remaining[:]:
                    w, h, d = boxes[idx]
                    rots = rotations_map[idx]
                    
                    best_rot = None
                    for rw, rh, rd in rots:
                        if rh > slab_h:
                            continue
                        if rw > W or rd > D:
                            continue
                        best_rot = (rw, rh, rd)
                        break
                    
                    if best_rot is None:
                        continue
                    
                    rw, rh, rd = best_rot
                    
                    if x + rw > W:
                        y += shelf_h
                        x = 0
                        shelf_h = 0
                    
                    if y + rd > D:
                        break
                    
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    placed_in_slab.append(idx)
                    x += rw
                    shelf_h = max(shelf_h, rh)
                
                # Remove placed boxes from remaining
                remaining = [i for i in remaining if i not in set(placed_in_slab)]
                z += slab_h
            
            # Mark remaining as not placed
            for idx in remaining:
                if placements[idx] is None:
                    placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            
            vol = 0
            for p in placements:
                if p[0] == 0:
                    vol += p[4] * p[5] * p[6]
            
            if vol > best_vol:
                best_vol = vol
                best_pl = placements
            
            if time.time() > deadline:
                break
        
        if best_pl is None:
            return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes], 0
        
        return best_pl, best_vol
    
    # Try each strategy
    strategies = [
        ("shelf", try_shelf_packing),
        ("extreme", try_extreme_point),
        ("layer", try_layer_decomposition),
    ]
    
    for name, func in strategies:
        if time.time() > deadline:
            break
        
        try:
            placements, vol = func(sorted_indices, all_rotations)
            if vol > best_volume:
                best_volume = vol
                best_placements = placements
        except Exception:
            pass
    
    # If no placements found, return all as not placed
    if best_placements is None:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
# EVOLVE-BLOCK-END
