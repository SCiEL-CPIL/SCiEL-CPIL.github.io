import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Pre-calculate orientations for each box
    # We want to try all 6 permutations of dimensions
    box_orientations = []
    for w, h, d in boxes:
        perms = set(itertools.permutations((w, h, d)))
        box_orientations.append(list(perms))
        
    # Sort boxes initially by volume descending to prioritize large boxes
    # But we will randomize order in restarts
    volumes = [w*h*d for w, h, d in boxes]
    
    best_fill = 0
    best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Grid resolution: We use a coarse grid to approximate the skyline.
    # A finer grid is better but slower. 
    # Let's use a grid that is at least 1 unit, but maybe capped for performance.
    # Actually, for 3D skyline, we can just store the max height at each (x,y) grid point.
    # Resolution: Let's try to keep grid size manageable.
    # If W, H, D are large, we need to be careful.
    # Let's assume dimensions are integers or small enough.
    # If they are floats, we need to scale or use a different method.
    # The task implies standard container loading, usually integer or simple floats.
    # Let's assume we can discretize.
    
    # To be safe and fast, we'll use a "skyline" approach with a list of blocks
    # or a simple grid. A grid is easier to implement correctly for 3D.
    
    # Let's estimate grid size. If W, H, D are up to 100, 100^3 = 1M cells. 
    # If up to 1000, 1B cells is too big.
    # Let's check typical sizes. If they are large, we need a different approach.
    # However, 60 boxes is small.
    
    # Alternative: Store placed boxes and check overlap. O(N^2) per box placement.
    # N=60, so 3600 checks per box, 216,000 total. Very fast.
    # Let's use the simple overlap check method to avoid grid resolution issues.
    
    def can_place(x, y, z, w, h, d, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            # Check overlap
            if not (x + w <= px or px + pw <= x or 
                    y + h <= py or py + ph <= y or 
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def find_position(w, h, d, placed):
        # Try to find the lowest z, then lowest y, then lowest x
        # This is a heuristic.
        # We can iterate through potential positions.
        # To be efficient, we can just check (0,0,0), then scan.
        # But scanning the whole volume is O(W*H*D).
        
        # Better heuristic: 
        # 1. Check if it fits at (0,0,0)
        # 2. Try to place it on top of existing boxes?
        
        # Simple approach for small N:
        # Generate candidate positions based on existing boxes + 0
        # x candidates: 0, and box_x + box_w for each box
        # y candidates: 0, and box_y + box_h for each box
        # z candidates: 0, and box_z + box_d for each box
        
        x_cands = [0]
        y_cands = [0]
        z_cands = [0]
        
        for px, py, pz, pw, ph, pd in placed:
            if px + pw <= W: x_cands.append(px + pw)
            if py + ph <= H: y_cands.append(py + ph)
            if pz + pd <= D: z_cands.append(pz + pd)
            
        # Sort to prioritize lower positions
        x_cands = sorted(set(x_cands))
        y_cands = sorted(set(y_cands))
        z_cands = sorted(set(z_cands))
        
        best_pos = None
        best_score = float('inf')
        
        for z in z_cands:
            if z + d > D:
                continue
            for y in y_cands:
                if y + h > H:
                    continue
                for x in x_cands:
                    if x + w > W:
                        continue
                    if can_place(x, y, z, w, h, d, placed):
                        # Heuristic: minimize z, then y, then x
                        score = z * H * W + y * W + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            
        return best_pos

    def run_pack(order):
        placed = []
        placements = [None] * n
        current_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            # Try all orientations
            best_pos = None
            best_orient = None
            best_score = float('inf')
            
            for ow, oh, od in orients:
                pos = find_position(ow, oh, od, placed)
                if pos is not None:
                    x, y, z = pos
                    # Same scoring: minimize z, y, x
                    score = z * H * W + y * W + x
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orient = (ow, oh, od)
            
            if best_pos is not None:
                x, y, z = best_pos
                ow, oh, od = best_orient
                placed.append((x, y, z, ow, oh, od))
                placements[idx] = (0, x, y, z, ow, oh, od)
                current_vol += ow * oh * od
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        return current_vol, placements

    # Multi-start strategy
    # 1. Sorted by volume desc
    # 2. Random permutations
    
    orders = []
    
    # Order 1: Volume descending
    orders.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))
    
    # Order 2-10: Random
    # We will generate random orders on the fly to save memory, but let's pre-generate a few
    for _ in range(10):
        random_order = list(range(n))
        random.shuffle(random_order)
        orders.append(random_order)
    
    for i, order in enumerate(orders):
        if time.time() - start_time > DEADLINE:
            break
            
        vol, placements = run_pack(order)
        
        if vol > best_fill:
            best_fill = vol
            best_placements = placements
            
    return best_placements
# EVOLVE-BLOCK-END
