"""3D Container Loading: Hybrid Skyline + Best-Fit with Local Search."""
import time
import random
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Preprocess boxes
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        # Generate unique rotations
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        
        valid_rots = [r for r in rots if r[0] <= W and r[1] <= H and r[2] <= D]
        if not valid_rots:
            valid_rots = [(w, h, d)]
            
        # Sort rotations to prefer "flatter" or more compact ones? 
        # Actually, for Best Fit, we usually just iterate.
        # Let's sort by max dimension descending to try big ones first? Or ascending?
        # Usually, trying the orientation that fits best in the "tightest" dimension is good.
        # Let's just keep them as a list.
        box_data.append((i, w, h, d, valid_rots, w*h*d))

    def check_overlap(x, y, z, bw, bh, bd, placed):
        """Check if a box at (x,y,z) with dims (bw,bh,bd) overlaps any in placed."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return True
        return False

    def find_best_position(placed, bw, bh, bd):
        """
        Find the best position for a box of dims (bw, bh, bd).
        Uses candidate points generated from existing boxes' faces.
        Returns (x, y, z) or None.
        """
        # Generate candidate coordinates
        x_cands = {0}
        y_cands = {0}
        z_cands = {0}
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_cands.add(px)
            x_cands.add(px + pw)
            y_cands.add(py)
            y_cands.add(py + ph)
            z_cands.add(pz)
            z_cands.add(pz + pd)
            
        # Filter candidates that are within bounds
        x_cands = [x for x in x_cands if x + bw <= W]
        y_cands = [y for y in y_cands if y + bh <= H]
        z_cands = [z for z in z_cands if z + bd <= D]
        
        if not x_cands or not y_cands or not z_cands:
            return None

        # Sort candidates to prioritize "Bottom-Left-Far" (low z, low x, low y)
        # Actually, standard BLF is usually (z, x, y) or similar.
        # Let's try to minimize (z, x, y) to pack tightly.
        best_pos = None
        best_score = None
        
        # To speed up, we can limit the number of candidates checked if the set is huge,
        # but for N=60, the set size is at most 2*60+1 = 121. 
        # 121^3 = ~1.7M checks. This is too slow for many restarts in Python.
        # Optimization: Only check candidates that are "close" to the existing packing?
        # Or use a simpler heuristic:
        # 1. Try to place on the "floor" (z=0) first?
        # 2. Use a grid?
        
        # Let's optimize the candidate generation.
        # Instead of all combinations, we can iterate through placed boxes and check specific spots:
        # - Next to box in X
        # - Next to box in Y
        # - Next to box in Z
        # - Under box in Z (if space)
        # This reduces candidates to O(N) per box.
        
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed:
            # X-adjacent
            if px + pw + bw <= W:
                candidates.add((px + pw, py, pz))
            # Y-adjacent
            if py + ph + bh <= H:
                candidates.add((px, py + ph, pz))
            # Z-adjacent (on top)
            if pz + pd + bh <= H: # Wait, bh is height (Y), bd is depth (Z).
                # Box dims: bw (x), bh (y), bd (z).
                # Placed dims: pw (x), ph (y), pd (z).
                # "On top" means increasing Y.
                if pz + pd + bd <= D and py + ph + bh <= H: # This is not strictly "on top", this is just a new spot.
                    candidates.add((px, py + ph, pz))
                # "Behind" means increasing Z
                if px + pw + bw <= W and py + ph + bh <= H:
                    candidates.add((px, py, pz + pd))
            # Also try the "corner" spots relative to the box
            # (px+pw, py+ph, pz), etc?
            # Let's add the 8 corners of the placed box's bounding box expansion?
            # No, just the faces is usually enough for BFD.
            
            # Add spots "under" if we are filling from top down? No, we fill bottom up.
            # So we mainly care about x, y, z extensions.
            
            # Let's also add the spot (0, 0, pz+pd) etc?
            # The set {0, px, px+pw} for each axis is the standard approach.
            # Let's stick to the face-based candidates for speed, but ensure we include (0,0,0).
            
        # If the face-based set is empty or fails, fall back to the full grid?
        # For now, let's use the face-based candidates. It's O(N^2) per box -> O(N^3) total.
        # 60^3 = 216k operations per simulation. This is fast enough for ~1000 simulations.
        
        for (cx, cy, cz) in candidates:
            if cx + bw > W or cy + bh > H or cz + bd > D:
                continue
            if check_overlap(cx, cy, cz, bw, bh, bd, placed):
                continue
            # Score: prefer lower Z, then lower X, then lower Y?
            # Or lower Y (height) first?
            # Standard BLF: Bottom (low Y), Left (low X), Far (low Z? or high Z?)
            # "Far" usually means furthest in the depth direction (Z), so high Z? 
            # Or is it "Back"? 
            # Let's try (z, x, y) ascending. This packs towards the origin.
            score = (cz, cx, cy)
            if best_score is None or score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
                
        if best_pos is None:
            # Fallback: try a few random spots? Or just fail.
            # Let's try the full grid if N is small, but we assumed N=60.
            # For robustness, let's just return None.
            pass
            
        return best_pos

    def simulate_packing(order_indices, rotation_strategy="random"):
        """
        Simulates packing boxes in the given order.
        Returns (filled_volume, placements_dict).
        """
        placed = []
        placements = {}
        filled_vol = 0.0
        
        for idx in order_indices:
            orig_idx, w, h, d, rots, vol = box_data[idx]
            
            # Choose rotation
            if rotation_strategy == "random":
                chosen_rot = random.choice(rots)
            else:
                chosen_rot = rots[0] # Default first rotation
            
            rw, rh, rd = chosen_rot
            
            pos = find_best_position(placed, rw, rh, rd)
            
            if pos is not None:
                x, y, z = pos
                placed.append((x, y, z, rw, rh, rd))
                placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                filled_vol += rw * rh * rd
            # If not placed, leave it out. We can try other rotations if the first fails?
            # Let's try all rotations if the first one fails.
            else:
                placed_this = False
                for rot in rots:
                    rw, rh, rd = rot
                    pos = find_best_position(placed, rw, rh, rd)
                    if pos is not None:
                        x, y, z = pos
                        placed.append((x, y, z, rw, rh, rd))
                        placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                        filled_vol += rw * rh * rd
                        placed_this = True
                        break
                if not placed_this:
                    placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any missing indices (shouldn't happen if order covers all, but just in case)
        for i in range(n):
            if i not in placements:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
                
        return filled_vol, placements

    # Local Search: Improve the placement by swapping rotations or moving boxes?
    # For now, let's stick to Multi-Start with different orders.
    
    best_vol = -1
    best_placements = None
    
    # Strategy 1: Volume Descending
    order1 = sorted(range(n), key=lambda i: box_data[i][5], reverse=True)
    
    # Strategy 2: Max Dimension Descending
    order2 = sorted(range(n), key=lambda i: max(box_data[i][1], box_data[i][2], box_data[i][3]), reverse=True)
    
    # Strategy 3: Surface Area Descending
    def surface_area(i):
        w, h, d = box_data[i][1], box_data[i][2], box_data[i][3]
        return 2*(w*h + h*d + d*w)
    order3 = sorted(range(n), key=surface_area, reverse=True)
    
    # Strategy 4: Random
    order4 = list(range(n))
    random.shuffle(order4)
    
    orders = [order1, order2, order3, order4]
    
    # Run initial strategies
    for order in orders:
        if time.time() > deadline - 10:
            break
        vol, pls = simulate_packing(order)
        if vol > best_vol:
            best_vol = vol
            best_placements = pls

    # Multi-Start with perturbations
    # Base order: Best performing order? Or just Volume Descending?
    base_order = order1
    
    restarts = 0
    while time.time() < deadline - 5:
        # Perturb the base order
        order = list(base_order)
        
        # Randomly swap a few elements
        for _ in range(3):
            i = random.randint(0, n - 2)
            j = random.randint(0, n - 2)
            order[i], order[j] = order[j], order[i]
            
        # Also try randomizing the rotation choice
        vol, pls = simulate_packing(order, rotation_strategy="random")
        
        if vol > best_vol:
            best_vol = vol
            best_placements = pls
            
        restarts += 1
        # Limit restarts to avoid infinite loop if time is not the constraint
        if restarts > 200:
            break

    # Finalize result
    final_placements = [None] * n
    if best_placements:
        for i in range(n):
            final_placements[i] = best_placements.get(i, (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
    else:
        for i in range(n):
            final_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return final_placements
# EVOLVE-BLOCK-END
