"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # safety margin before 120s limit
    n = len(boxes)
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    orientations = [get_orientations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending for better packing
    order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def try_pack(order_list, seed=0):
        """Try to pack boxes in given order using shelf-based approach with rotation."""
        rng = random.Random(seed)
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
        # Shelf-based packing: divide container into horizontal shelves (layers in z)
        # Each shelf has a fixed height, and boxes are placed left-to-right within rows
        
        # We'll use a 3D shelf approach:
        # - Process boxes in order
        # - For each box, try to find a position in existing shelves or create new shelf
        # - Within a shelf, place boxes in rows
        
        # State: list of shelves, each shelf is (z_start, z_height, list of rows)
        # Each row: (y_start, y_height, x_cursor)
        
        shelves = []  # each shelf: [z_start, z_height, [(y_start, y_height, x_cursor), ...]]
        
        for idx in order_list:
            w, h, d = boxes[idx]
            placed = False
            
            # Try all orientations
            for ow, oh, od in orientations[idx]:
                if ow > W or oh > H or od > D:
                    continue
                
                # Try to place in existing shelves
                for s_idx, shelf in enumerate(shelves):
                    z_start, z_height, rows = shelf
                    # Box must fit in shelf height
                    if oh > z_height:
                        continue
                    
                    # Try to place in existing rows
                    for r_idx, row in enumerate(rows):
                        y_start, y_height, x_cursor = row
                        if oh > y_height:
                            continue
                        if x_cursor + ow <= W:
                            # Place here
                            x_pos = x_cursor
                            y_pos = y_start
                            z_pos = z_start
                            placements[idx] = (0, x_pos, y_pos, z_pos, ow, oh, od)
                            rows[r_idx] = (y_start, y_height, x_cursor + ow)
                            placed = True
                            break
                    
                    if placed:
                        break
                    
                    # Try creating a new row in this shelf
                    # Find max y used in this shelf
                    max_y = 0
                    for row in rows:
                        max_y = max(max_y, row[0] + row[1])
                    if max_y + oh <= z_height:
                        y_pos = max_y
                        z_pos = z_start
                        x_pos = 0
                        placements[idx] = (0, x_pos, y_pos, z_pos, ow, oh, od)
                        rows.append((y_pos, oh, ow))
                        placed = True
                        break
                
                if placed:
                    break
            
            if not placed:
                # Create a new shelf
                # Find max z used
                max_z = 0
                for shelf in shelves:
                    max_z = max(max_z, shelf[0] + shelf[1])
                
                # Try all orientations again for new shelf
                for ow, oh, od in orientations[idx]:
                    if ow > W or oh > H or od > D:
                        continue
                    if max_z + od > D:
                        continue
                    
                    z_pos = max_z
                    y_pos = 0
                    x_pos = 0
                    placements[idx] = (0, x_pos, y_pos, z_pos, ow, oh, od)
                    shelves.append((z_pos, od, [(y_pos, oh, ow)]))
                    placed = True
                    break
            
            if not placed:
                # Can't place this box
                continue
        
        # Calculate filled volume
        filled = sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n) if placements[i][0] == 0)
        total_vol = W * H * D
        return filled / total_vol, placements
    
    # Try multiple orders and seeds
    best_score = 0
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # First try the volume-sorted order
    score, pl = try_pack(order, seed=0)
    if score > best_score:
        best_score = score
        best_placements = pl
    
    # Try a few more random perturbations of the order if time permits
    base_order = list(order)
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Slightly perturb the order
        perturbed = list(base_order)
        for _ in range(3):
            i = random.randint(0, len(perturbed)-1)
            j = random.randint(0, len(perturbed)-1)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        score, pl = try_pack(perturbed, seed=attempt)
        if score > best_score:
            best_score = score
            best_placements = pl
    
    return best_placements
# EVOLVE-BLOCK-END
