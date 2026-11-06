"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 120s limit, keep 10s margin for safety/overhead
    deadline = start_time + 110.0
    
    # Preprocess: store box dimensions and original indices
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        box_data.append((i, w, h, d))
        
    # Initial result: all boxes left out
    final_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    def get_rotations(w, h, d):
        # Generate unique rotations
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)

    def run_bfd(order):
        """
        Runs the Best-Fit Decreasing heuristic for a given order.
        Returns (filled_volume, placements_dict)
        """
        placed = []
        placements = {}
        filled_vol = 0.0
        
        for idx in order:
            orig_idx, w, h, d = box_data[idx]
            rots = get_rotations(w, h, d)
            
            best_pos = None
            best_rot = None
            best_score = None
            
            for rw, rh, rd in rots:
                # Candidate x-coordinates: 0 and right faces of all placed boxes
                x_candidates = set([0])
                y_candidates = set([0])
                z_candidates = set([0])
                
                for (px, py, pz, pw, ph, pd) in placed:
                    x_candidates.add(px + pw)
                    y_candidates.add(py + ph)
                    z_candidates.add(pz + pd)
                    
                # Filter candidates that fit within container bounds
                x_candidates = [x for x in x_candidates if x + rw <= W]
                y_candidates = [y for y in y_candidates if y + rh <= H]
                z_candidates = [z for z in z_candidates if z + rd <= D]
                
                if not x_candidates or not y_candidates or not z_candidates:
                    continue

                # Heuristic: Minimize z (bottom), then x (left), then y (far/back)
                # Score tuple: (z, x, -y) -> min
                local_best_score = None
                local_best_coord = None
                
                for z in sorted(z_candidates):
                    for x in sorted(x_candidates):
                        for y in y_candidates:
                            # Check overlap
                            overlap = False
                            for (px, py, pz, pw, ph, pd) in placed:
                                if not (x + rw <= px or px + pw <= x or 
                                        y + rh <= py or py + ph <= y or 
                                        z + rd <= pz or pz + pd <= z):
                                    overlap = True
                                    break
                            
                            if not overlap:
                                score = (z, x, -y)
                                if local_best_score is None or score < local_best_score:
                                    local_best_score = score
                                    local_best_coord = (x, y, z)
                                    
                if local_best_coord is not None:
                    # Compare with global best for this box (across rotations)
                    # We prefer the first rotation that fits? Or the one that leaves the most space?
                    # Standard BFD usually just takes the first valid rotation in a specific order or the one that fits best.
                    # Let's stick to: first valid rotation that has a valid position.
                    # Actually, to be consistent with the previous simple loop, we just break on first valid rotation.
                    # But here we are checking all rotations. Let's keep the one with the "best" position score.
                    if best_score is None or local_best_score < best_score:
                        best_score = local_best_score
                        best_pos = local_best_coord
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed.append((x, y, z, rw, rh, rd))
                placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                filled_vol += rw * rh * rd
                
            if time.time() > deadline:
                break
                
        return filled_vol, placements

    def run_layered(order):
        """
        Time-sliced layer decomposition strategy.
        Discretizes container height H into slabs.
        Packs boxes into each slab using a 2D shelf heuristic on the (W, D) plane.
        """
        # Determine number of layers
        # We can try a few different layer heights or just use a fixed grid.
        # Let's try to pack greedily by height.
        # Sort order is already given.
        
        # We will maintain a list of "layers" (slabs)
        # Each layer has:
        #   y_start: float
        #   y_end: float
        #   shelves: list of (z_start, z_end, x_current) -- 2D shelves in WxD plane?
        #   Actually, standard shelf packing in 3D usually fixes one axis.
        #   Let's fix Y as the vertical axis for layers.
        #   Within a layer, we pack in X and Z.
        #   We can use 2D shelf packing on the X-Z plane for each layer.
        
        layers = [] # List of dicts: {'y_start':, 'y_end':, 'shelves': [(z, x_end, max_h_in_layer? No, h is fixed for layer?)]}
        # Wait, if we fix layer height, all boxes in that layer must have h <= layer_height.
        # This might waste space if boxes have varying heights.
        # Alternative: "Greedy Layer" where we pick the next box, determine its height,
        # and if it doesn't fit in current layer, start a new layer.
        
        # Let's try a simpler approach:
        # 1. Sort boxes by height descending? No, order is fixed by 'order' param.
        # 2. Iterate through 'order'.
        # 3. For each box, try to fit it in existing layers (shelves).
        # 4. If it fits, place it.
        # 5. If not, start a new layer at the current max_y.
        
        # State:
        # layers: list of (y_start, y_end, shelves)
        # shelves: list of (z_start, z_end, x_current)
        # Note: In a layer, all boxes have the same y range?
        # If we start a new layer when a box doesn't fit in height, then yes.
        # But what if a box is short and could fit in a previous layer's remaining height?
        # This requires more complex bookkeeping.
        
        # Let's stick to a simpler "Shelf" approach that is fast.
        # 2D Shelf Packing on X-Z plane, stacking in Y.
        # This is effectively what the previous "Shelf" idea was, but let's implement it properly.
        
        shelves = [] # List of (y, z, x_current, max_h)
        # y: start y of shelf
        # z: start z of shelf
        # x_current: current x position in this shelf
        # max_h: max height allowed in this shelf (determined by first box? or container H?)
        # Let's say max_h is H for all shelves initially, but we can refine.
        # Actually, if we place a box of height h in a shelf, all subsequent boxes in that shelf
        # must have height <= h? No, they can be shorter, but they occupy the same y range.
        # So the shelf height is determined by the tallest box placed in it so far.
        
        filled_vol = 0.0
        placements = {}
        
        for idx in order:
            orig_idx, w, h, d = box_data[idx]
            rots = get_rotations(w, h, d)
            
            best_place = None
            best_score = None
            
            for rw, rh, rd in rots:
                # Try to fit in existing shelves
                for s_idx, (s_y, s_z, s_x, s_max_h) in enumerate(shelves):
                    # Check if fits in height
                    if rh <= s_max_h:
                        # Check if fits in x
                        if s_x + rw <= W:
                            # Check if fits in z (it does by definition of shelf)
                            # Check for overlap with other boxes in the same shelf?
                            # Shelves are separated in Z or Y?
                            # My definition: shelves are at specific (y, z).
                            # Boxes in the same shelf share the same y range and z range?
                            # No, usually shelves are horizontal strips.
                            # Let's redefine:
                            # A shelf is defined by y_start, y_end, z_start, z_end.
                            # Boxes in a shelf are placed along X.
                            # So they don't overlap in X.
                            # They share the same Y and Z ranges.
                            
                            # So if we place box at (s_x, s_y, s_z) with size (rw, rh, rd),
                            # we need:
                            # 1. rh <= s_max_h (height fits in shelf thickness)
                            # 2. rd <= shelf_z_thickness? 
                            #    If the shelf is defined by a specific z range, then yes.
                            #    But if we just define shelves by (y, z_start), and they extend to z_end?
                            #    Let's say shelves extend in Z from s_z to s_z + rd?
                            #    No, that's not a shelf.
                            
                            # Let's use a simpler "2D Shelf" on X-Z, stacking in Y.
                            # Shelves are defined by (y_start, y_end).
                            # Within a shelf, we pack in X and Z.
                            # This is equivalent to 2D packing in X-Z plane for each shelf.
                            # To keep it fast, we use "Shelf" within the shelf:
                            # Sub-shelves defined by z_start.
                            
                            # This is getting complicated. Let's go back to the simple 3D Shelf:
                            # Shelves are horizontal (constant Y range).
                            # Within a shelf, we use 2D Shelves (constant Z range).
                            
                            # Let's try a different fast heuristic:
                            # "Bottom-Left-Far" but restricted to a few candidate points.
                            # Or just use the BFD but with a time limit per box?
                            
                            # Given the time constraints and complexity, let's just use the BFD 
                            # but try to optimize the candidate generation.
                            
                            pass
                
                # If the layered approach is too complex to implement correctly and fast,
                # let's fall back to a Multi-Start BFD with different sorting orders.
                
                pass
            
            # Fallback: Just use BFD logic here if layered fails to place
            # Actually, let's just implement the Multi-Start BFD strategy properly.
            # The "Layered" idea might be too risky to implement correctly in the remaining time.
            # Let's stick to the "Multiple Sorting Strategies" idea which was measured to improve score.
            
            break # Break out of rotation loop to use BFD fallback
            
        return 0.0, {} # Placeholder, we will use BFD instead

    # Strategy: Multi-Start BFD with different sorting orders
    # 1. Volume Descending
    # 2. Max Dimension Descending
    # 3. Volume / Max Dimension Ratio (Flatness)
    # 4. Random Shuffles
    
    best_vol = 0.0
    best_placements = {}
    
    # Define sorting keys
    def key_volume(x):
        i, w, h, d = x
        return w*h*d
    
    def key_max_dim(x):
        i, w, h, d = x
        return max(w, h, d)
        
    def key_flatness(x):
        i, w, h, d = x
        return max(w, h, d) / (w*h*d) # Higher ratio = flatter? No, lower ratio = flatter?
        # Actually, we want to pack flat boxes well.
        # Let's just try a few standard ones.
        
    orders = []
    
    # Order 1: Volume Descending
    orders.append(sorted(range(n), key=lambda i: box_data[i][1]*box_data[i][2]*box_data[i][3], reverse=True))
    
    # Order 2: Max Dimension Descending
    orders.append(sorted(range(n), key=lambda i: max(box_data[i][1], box_data[i][2], box_data[i][3]), reverse=True))
    
    # Order 3: Sum of dimensions
    orders.append(sorted(range(n), key=lambda i: box_data[i][1]+box_data[i][2]+box_data[i][3], reverse=True))
    
    # Order 4-10: Random Shuffles
    for _ in range(7):
        rand_order = list(range(n))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        vol, placements = run_bfd(order)
        
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
            
    # Update final_placements with best result
    for idx, (b, x, y, z, w, h, d) in best_placements.items():
        final_placements[idx] = (b, x, y, z, w, h, d)
        
    return final_placements
# EVOLVE-BLOCK-END
