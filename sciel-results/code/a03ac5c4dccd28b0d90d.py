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
    # 150s limit, keep 10s margin for safety/overhead
    deadline = start_time + 140.0
    
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

    def can_place(x, y, z, bw, bh, bd, placed):
        """
        Check if a box of size (bw, bh, bd) can be placed at (x, y, z)
        without overlapping any box in 'placed'.
        """
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return False
        return True

    def find_best_fit(x, y, z, bw, bh, bd, placed):
        """
        Calculate a 'badness' score for placing a box at (x, y, z).
        Lower is better.
        Heuristic: Minimize distance to existing boxes (compactness) 
        and push towards back/bottom/left.
        """
        if not placed:
            return (z, x, -y) # Just use BLF for the first box
        
        min_dist = float('inf')
        for (px, py, pz, pw, ph, pd) in placed:
            # Check if this box is "close" (sharing a face or near)
            # Simple Euclidean distance to the nearest face of any placed box
            # Actually, a common heuristic is to minimize the sum of distances to boundaries
            # or maximize contact with other boxes.
            
            # Let's use a simpler metric: 
            # 1. Prefer lower z (bottom)
            # 2. Prefer lower x (left)
            # 3. Prefer higher y (back)
            # 4. Prefer positions that are "supported" (z is close to floor or top of other boxes)
            
            # A good "Best Fit" proxy is minimizing the volume of the "gap" left behind,
            # but that's complex. 
            # Let's stick to a weighted sum:
            # score = 10*z + 5*x - 5*y + 0.1 * (distance to nearest placed box)
            
            # Distance to nearest placed box:
            dx = max(px - (x + bw), x - (px + pw), 0)
            dy = max(py - (y + bh), y - (py + ph), 0)
            dz = max(pz - (z + bd), z - (pz + pd), 0)
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < min_dist:
                min_dist = dist
        
        # Weighted score: Bottom (z) is most important, then Left (x), then Back (-y)
        # And some penalty for being far from other boxes (to keep it compact)
        score = 10.0 * z + 5.0 * x - 5.0 * y + 0.5 * min_dist
        return score

    def run_packing_pass(order, deadline):
        """
        Run a single packing pass with the given box order.
        Returns (total_volume, placements_dict)
        """
        placed = []
        placements = {}
        total_vol = 0
        
        for idx in order:
            if time.time() > deadline:
                break
                
            orig_idx, w, h, d = box_data[idx]
            rots = get_rotations(w, h, d)
            placed_this = False
            
            best_pos = None
            best_score = float('inf')
            
            for rw, rh, rd in rots:
                # Candidate positions:
                # 0, and right/back/top faces of all placed boxes
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

                for z in z_candidates:
                    for x in x_candidates:
                        for y in y_candidates:
                            if can_place(x, y, z, rw, rh, rd, placed):
                                score = find_best_fit(x, y, z, rw, rh, rd, placed)
                                if score < best_score:
                                    best_score = score
                                    best_pos = (x, y, z, rw, rh, rd)
                
            if best_pos is not None:
                x, y, z, rw, rh, rd = best_pos
                placed.append((x, y, z, rw, rh, rd))
                placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                total_vol += rw * rh * rd
                placed_this = True
            else:
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
                
        return total_vol, placements

    # Multi-start strategy
    best_total_vol = 0
    best_placements = None
    
    # Try a few different orderings
    # 1. Volume Descending
    order1 = sorted(range(n), key=lambda i: box_data[i][1]*box_data[i][2]*box_data[i][3], reverse=True)
    vol1, plac1 = run_packing_pass(order1, deadline)
    if vol1 > best_total_vol:
        best_total_vol = vol1
        best_placements = plac1
    
    # 2. Randomized orderings
    # If we have time, try a few random shuffles
    # Limit the number of restarts based on time
    max_restarts = 5
    for r in range(max_restarts):
        if time.time() > deadline - 5: # Keep 5s margin
            break
            
        random_order = list(range(n))
        random.shuffle(random_order)
        vol_r, plac_r = run_packing_pass(random_order, deadline)
        if vol_r > best_total_vol:
            best_total_vol = vol_r
            best_placements = plac_r
    
    # Construct final output
    if best_placements is not None:
        for i in range(n):
            if i in best_placements:
                final_placements[i] = best_placements[i]
            else:
                final_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                
    return final_placements
# EVOLVE-BLOCK-END
