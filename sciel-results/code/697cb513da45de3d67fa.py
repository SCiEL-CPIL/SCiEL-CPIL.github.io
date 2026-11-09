import time
import random
import itertools
from collections import defaultdict

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    W, H, D = container
    container_volume = W * H * D
    
    # Pre-compute all possible rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        box_rotations.append(list(rots))
        
    # Spatial Hash Parameters
    avg_w = sum(b[0] for b in boxes) / len(boxes)
    avg_h = sum(b[1] for b in boxes) / len(boxes)
    avg_d = sum(b[2] for b in boxes) / len(boxes)
    cell_size = max(1, int((avg_w + avg_h + avg_d) / 3))
    
    def get_cells(x, y, z, w, h, d):
        """Return set of cell coordinates that the box overlaps."""
        cells = set()
        x_min = int(x // cell_size)
        x_max = int((x + w - 1e-9) // cell_size)
        y_min = int(y // cell_size)
        y_max = int((y + h - 1e-9) // cell_size)
        z_min = int(z // cell_size)
        z_max = int((z + d - 1e-9) // cell_size)
        
        for cx in range(x_min, x_max + 1):
            for cy in range(y_min, y_max + 1):
                for cz in range(z_min, z_max + 1):
                    cells.add((cx, cy, cz))
        return cells

    def can_place(placed, spatial_hash, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        
        # Get cells for the new box
        new_cells = get_cells(x, y, z, w, h, d)
        
        # Check all boxes in those cells
        for cell in new_cells:
            if cell in spatial_hash:
                for idx in spatial_hash[cell]:
                    px, py, pz, pw, ph, pd = placed[idx]
                    # Check overlap
                    if not (x + w <= px + 1e-9 or px + pw <= x + 1e-9 or 
                            y + h <= py + 1e-9 or py + ph <= y + 1e-9 or 
                            z + d <= pz + 1e-9 or pz + pd <= z + 1e-9):
                        return False
        return True

    def solve_with_order(order):
        placed = [] # list of (x, y, z, w, h, d)
        spatial_hash = defaultdict(list)
        placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(len(boxes))]
        candidates = {(0, 0, 0)}
        
        for i in order:
            box_id = i
            w_orig, h_orig, d_orig = boxes[box_id]
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rot in box_rotations[box_id]:
                rw, rh, rd = rot
                for cx, cy, cz in candidates:
                    if can_place(placed, spatial_hash, cx, cy, cz, rw, rh, rd):
                        # Heuristic: minimize x+y+z to pack tightly
                        score = cx + cy + cz
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = rot
            
            if best_pos:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed_idx = len(placed)
                placed.append((bx, by, bz, bw, bh, bd))
                placements[box_id] = (0, bx, by, bz, bw, bh, bd)
                
                # Update Spatial Hash
                cells = get_cells(bx, by, bz, bw, bh, bd)
                for cell in cells:
                    spatial_hash[cell].append(placed_idx)
                
                # Update Candidates
                new_corners = [
                    (bx + bw, by, bz),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                    (bx, by + bh, bz),
                    (bx, by + bh, bz + bd),
                    (bx, by, bz + bd)
                ]
                for nc in new_corners:
                    if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                        candidates.add(nc)
            else:
                pass
        
        # Calculate volume filled
        vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        return placements, vol

    # Try multiple sorting strategies
    n = len(boxes)
    indices = list(range(n))
    
    best_placements = None
    best_vol = 0
    
    # Strategy 1: Volume Descending
    order_vol = sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    p, v = solve_with_order(order_vol)
    if v > best_vol:
        best_vol = v
        best_placements = p
        
    # Strategy 2: Max Dimension Descending
    order_max = sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    p, v = solve_with_order(order_max)
    if v > best_vol:
        best_vol = v
        best_placements = p
        
    # Strategy 3: Random Restarts
    max_restarts = 150
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        random.shuffle(indices)
        p, v = solve_with_order(indices)
        if v > best_vol:
            best_vol = v
            best_placements = p
            
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        
    return best_placements
