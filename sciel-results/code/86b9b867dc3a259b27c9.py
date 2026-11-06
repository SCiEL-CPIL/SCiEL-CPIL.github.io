"""3D Container Loading with improved shelf packing and multi-strategy search."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    deadline = time.time() + 115.0  # 5s safety margin before 120s kill
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    # Precompute orientations for all boxes
    box_orientations = [get_orientations(w, h, d) for w, h, d in boxes]
    
    def shelf_pack_3d(ordered_indices):
        """
        3D shelf packing with better position search:
        - Y-axis is the stacking axis (shelves at different y levels)
        - Each shelf has a fixed height
        - Within a shelf, boxes are placed in the X-Z plane
        - We use a more thorough search for positions
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        
        # Shelves: list of dicts {'y': float, 'h': float, 'boxes': [(x, z, ow, od, oh)]}
        shelves = []
        current_y = 0
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            # Try to place on existing shelves
            placed = False
            for shelf in shelves:
                if shelf['y'] + shelf['h'] > H:
                    continue
                # Try each orientation
                for ow, oh, od in orients:
                    if oh > shelf['h']:
                        continue
                    if ow > W or od > D:
                        continue
                    # Try to find a position (x, z) in this shelf where the box fits
                    # Use a grid search with finer resolution
                    found_pos = False
                    # Try x positions: 0, then check against existing boxes
                    x_candidates = [0]
                    for px, pz, pow_, pod, poh in shelf['boxes']:
                        if px + pow_ <= W - ow:
                            x_candidates.append(px + pow_)
                    x_candidates = sorted(set(x_candidates))
                    
                    for x in x_candidates:
                        if x + ow > W:
                            continue
                        z_candidates = [0]
                        for px, pz, pow_, pod, poh in shelf['boxes']:
                            if pz + pod <= D - od:
                                z_candidates.append(pz + pod)
                        z_candidates = sorted(set(z_candidates))
                        
                        for z in z_candidates:
                            if z + od > D:
                                continue
                            # Check if this position is free
                            free = True
                            for px, pz, pow_, pod, poh in shelf['boxes']:
                                # Check overlap in X and Z
                                if not (x + ow <= px or px + pow_ <= x):
                                    if not (z + od <= pz or pz + pod <= z):
                                        free = False
                                        break
                            if free:
                                # Place the box here
                                placements[idx] = (0, x, int(shelf['y']), z, ow, oh, od)
                                shelf['boxes'].append((x, z, ow, od, oh))
                                total_volume += ow * oh * od
                                placed = True
                                found_pos = True
                                break
                        if found_pos:
                            break
                    if placed:
                        break
                if placed:
                    break
            
            if placed:
                continue
            
            # If not placed on existing shelves, try to start a new shelf
            best_orient = None
            for ow, oh, od in orients:
                if current_y + oh <= H and ow <= W and od <= D:
                    # Prefer orientations that minimize height to fit more shelves
                    if best_orient is None or oh < best_orient[1]:
                        best_orient = (ow, oh, od)
            
            if best_orient:
                ow, oh, od = best_orient
                new_shelf = {'y': current_y, 'h': oh, 'boxes': [(0, 0, ow, od, oh)]}
                shelves.append(new_shelf)
                current_y += oh
                placements[idx] = (0, 0, int(current_y - oh), 0, ow, oh, od)
                total_volume += ow * oh * od
            # Else: box cannot fit in remaining height, leave it out
        
        return placements, total_volume
    
    # Generate different sorting strategies
    indexed_boxes = list(range(n))
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]),  # Max dim
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],  # Sum of dims
        lambda i: boxes[i][1],  # Height
        lambda i: boxes[i][0],  # Width
        lambda i: boxes[i][2],  # Depth
        lambda i: -min(boxes[i][0], boxes[i][1], boxes[i][2]),  # Min dim (ascending)
        lambda i: boxes[i][0] * boxes[i][1],  # Base area
        lambda i: boxes[i][1] * boxes[i][2],  # Side area
        lambda i: boxes[i][0] * boxes[i][2],  # Top area
        lambda i: (boxes[i][0] * boxes[i][1] * boxes[i][2]) / max(1, min(boxes[i][0], boxes[i][1], boxes[i][2])),  # Volume/min
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]) / max(1, min(boxes[i][0], boxes[i][1], boxes[i][2])),  # Aspect ratio
    ]
    
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_volume = 0
    
    for sort_key in strategies:
        if time.time() > deadline - 10:
            break
        
        ordered = sorted(indexed_boxes, key=sort_key, reverse=True)
        p, v = shelf_pack_3d(ordered)
        if v > best_volume:
            best_placements = p
            best_volume = v
    
    return best_placements
# EVOLVE-BLOCK-END
