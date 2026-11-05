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
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)

    def can_place(x, y, z, bw, bh, bd, placed):
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return False
        return True

    def find_best_fit(bw, bh, bd):
        """
        Finds the best position (x, y, z) for a box of size (bw, bh, bd)
        using a Best-Fit heuristic: minimize z, then x, then y.
        """
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
            
        x_candidates = [x for x in x_candidates if x + bw <= W]
        y_candidates = [y for y in y_candidates if y + bh <= H]
        z_candidates = [z for z in z_candidates if z + bd <= D]
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None

        best_score = None
        best_coord = None
        
        for z in sorted(z_candidates):
            for x in sorted(x_candidates):
                for y in sorted(y_candidates):
                    if can_place(x, y, z, bw, bh, bd, placed):
                        score = (z, x, y)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_coord = (x, y, z)
                            
        return best_coord

    def run_packing_pass(order_key):
        """
        Runs a full packing pass with a specific ordering strategy.
        Returns the final placements and total volume packed.
        """
        local_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        local_placed = []
        
        # Sort boxes based on order_key
        indexed_boxes = list(box_data)
        if order_key == 'volume':
            indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
        elif order_key == 'max_dim':
            indexed_boxes.sort(key=lambda x: max(x[1], x[2], x[3]), reverse=True)
        elif order_key == 'surface':
            indexed_boxes.sort(key=lambda x: 2*(x[1]*x[2] + x[2]*x[3] + x[1]*x[3]), reverse=True)
        else: # random
            random.shuffle(indexed_boxes)

        for idx, (w, h, d) in indexed_boxes:
            if time.time() > deadline:
                break
                
            rots = get_rotations(w, h, d)
            placed_this = False
            
            for rw, rh, rd in rots:
                pos = find_best_fit(rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    local_placed.append((x, y, z, rw, rh, rd))
                    local_placements[idx] = (0, x, y, z, rw, rh, rd)
                    placed_this = True
                    break
                    
        total_vol = sum(rw*rh*rd for (_, _, _, rw, rh, rd) in local_placed)
        return local_placements, total_vol

    # Try multiple strategies and keep the best one
    best_placements = None
    best_vol = 0
    
    strategies = ['volume', 'max_dim', 'surface', 'random', 'random']
    
    for strat in strategies:
        if time.time() > deadline:
            break
        placements, vol = run_packing_pass(strat)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
            
    if best_placements is not None:
        return best_placements
    return final_placements
# EVOLVE-BLOCK-END
