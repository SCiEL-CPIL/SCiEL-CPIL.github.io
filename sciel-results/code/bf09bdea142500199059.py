"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
import sys

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 150s limit, keep 10s margin for safety/overhead
    deadline = start_time + 140.0
    
    # Preprocess: store box dimensions and original indices
    # box_data[i] = (index, w, h, d, vol)
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        box_data.append((i, w, h, d, w*h*d))
        
    # Initial result: all boxes left out
    # Format: (bin, x, y, z, w, h, d)
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

    def can_place(placed, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) at (x, y, z) overlaps with any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return False
        return True

    def find_best_fit(placed, bw, bh, bd):
        """
        Find the 'best fit' position for a box.
        Best fit is defined as the position that minimizes the sum of distances
        to the closest placed box on each axis (encouraging tight packing).
        Returns (x, y, z) or None.
        """
        if bw > W or bh > H or bd > D:
            return None

        # Candidate coordinates are 0 and the max extent of placed boxes
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
            
        # Filter candidates that fit within container bounds
        x_candidates = [x for x in x_candidates if x + bw <= W]
        y_candidates = [y for y in y_candidates if y + bh <= H]
        z_candidates = [z for z in z_candidates if z + bd <= D]
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None

        best_score = None
        best_coord = None
        
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if can_place(placed, x, y, z, bw, bh, bd):
                        # Score: Prefer positions that are "lower" (z), "left" (x), "back" (y)
                        # This prioritizes filling the bottom layer first.
                        score = (z, x, y)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_coord = (x, y, z)
                            
        return best_coord

    def greedy_packing(box_indices):
        """
        Run a greedy packing pass on the given list of box indices.
        Returns (placed_list, placements_dict, total_volume)
        placed_list: list of (x, y, z, w, h, d, box_idx)
        placements_dict: dict mapping box_idx -> (x, y, z, w, h, d)
        """
        placed = []
        placements = {}
        total_vol = 0
        
        # Sort by volume descending
        sorted_indices = sorted(box_indices, key=lambda i: box_data[i][4], reverse=True)
        
        for idx in sorted_indices:
            _, w, h, d, vol = box_data[idx]
            rots = get_rotations(w, h, d)
            placed_this = False
            
            for rw, rh, rd in rots:
                pos = find_best_fit(placed, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    placed.append((x, y, z, rw, rh, rd, idx))
                    placements[idx] = (x, y, z, rw, rh, rd)
                    total_vol += vol
                    placed_this = True
                    break
                    
        return placed, placements, total_vol

    # --- Simulated Annealing Setup ---
    
    # Initialize with greedy packing
    all_indices = list(range(n))
    current_placed, current_placements, current_vol = greedy_packing(all_indices)
    
    best_vol = current_vol
    best_placements = current_placements.copy()
    
    # SA Parameters
    T = 100.0 # Initial temperature (arbitrary scale, we use volume diff)
    T_min = 1.0
    alpha = 0.995 # Cooling rate
    max_steps = 20000 # Max steps before stopping if no improvement or time runs out
    
    step_count = 0
    no_improve_count = 0
    max_no_improve = 500
    
    # Precompute rotations for all boxes to save time
    box_rots = {}
    for i, (idx, w, h, d, vol) in enumerate(box_data):
        box_rots[i] = get_rotations(w, h, d)

    while time.time() < deadline and step_count < max_steps:
        if T < T_min:
            break
            
        # Check for stagnation
        if no_improve_count > max_no_improve:
            break
            
        # Choose a random move
        # Move types:
        # 1. Relocate: Pick a placed box, remove it, try to re-insert it (possibly in different rotation/pos)
        # 2. Swap: Pick two placed boxes, remove both, try to re-insert them in swapped order
        
        if len(current_placed) < 2:
            # Can't swap, only relocate or break
            if len(current_placed) == 0:
                break
            move_type = 'relocate'
        else:
            # 70% relocate, 30% swap
            move_type = 'relocate' if random.random() < 0.7 else 'swap'
            
        new_placed = current_placed.copy()
        new_placements = current_placements.copy()
        new_vol = current_vol
        
        if move_type == 'relocate':
            # Pick a random box from placed
            idx_to_move = random.choice(new_placed)
            box_idx = idx_to_move[5]
            
            # Remove it
            new_placed.remove(idx_to_move)
            del new_placements[box_idx]
            new_vol -= box_data[box_idx][4]
            
            # Try to re-insert it
            _, w, h, d, vol = box_data[box_idx]
            rots = box_rots[box_idx]
            # Shuffle rotations to encourage different orientation
            random.shuffle(rots)
            
            inserted = False
            for rw, rh, rd in rots:
                pos = find_best_fit(new_placed, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    new_placed.append((x, y, z, rw, rh, rd, box_idx))
                    new_placements[box_idx] = (x, y, z, rw, rh, rd)
                    new_vol += vol
                    inserted = True
                    break
                    
            if not inserted:
                # Box was lost. This is a big penalty.
                # In SA, we might accept this if T is high, but usually we want to keep volume high.
                # Let's treat "lost box" as a very bad move (delta = -vol)
                pass
                
        elif move_type == 'swap':
            # Pick two distinct boxes
            i1, i2 = random.sample(range(len(new_placed)), 2)
            b1 = new_placed[i1]
            b2 = new_placed[i2]
            idx1 = b1[5]
            idx2 = b2[5]
            
            # Remove both
            new_placed.remove(b1)
            new_placed.remove(b2)
            del new_placements[idx1]
            del new_placements[idx2]
            new_vol -= box_data[idx1][4] + box_data[idx2][4]
            
            # Re-insert b2 first, then b1 (swapped order)
            # Note: find_best_fit depends on current 'placed' state, so order matters
            
            # Insert b2
            _, w2, h2, d2, vol2 = box_data[idx2]
            rots2 = box_rots[idx2]
            random.shuffle(rots2)
            ins2 = False
            for rw, rh, rd in rots2:
                pos = find_best_fit(new_placed, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    new_placed.append((x, y, z, rw, rh, rd, idx2))
                    new_placements[idx2] = (x, y, z, rw, rh, rd)
                    new_vol += vol2
                    ins2 = True
                    break
            
            # Insert b1
            _, w1, h1, d1, vol1 = box_data[idx1]
            rots1 = box_rots[idx1]
            random.shuffle(rots1)
            ins1 = False
            for rw, rh, rd in rots1:
                pos = find_best_fit(new_placed, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    new_placed.append((x, y, z, rw, rh, rd, idx1))
                    new_placements[idx1] = (x, y, z, rw, rh, rd)
                    new_vol += vol1
                    ins1 = True
                    break
            
            # If either failed to insert, the move is invalid (volume loss)
            # We handle this via the delta calculation below.

        delta = new_vol - current_vol
        
        # Acceptance criterion
        if delta > 0:
            # Always accept improvement
            current_placed = new_placed
            current_placements = new_placements
            current_vol = new_vol
            no_improve_count = 0
            
            if current_vol > best_vol:
                best_vol = current_vol
                best_placements = current_placements.copy()
                
        else:
            # Deterioration
            if random.random() < math.exp(delta / T):
                current_placed = new_placed
                current_placements = new_placements
                current_vol = new_vol
                no_improve_count += 1
            else:
                no_improve_count += 1
                
        T *= alpha
        step_count += 1
        
        # Periodic check for time
        if time.time() > deadline:
            break

    # Construct final output from best_placements
    # best_placements is a dict {box_idx: (x, y, z, w, h, d)}
    for i in range(n):
        if i in best_placements:
            x, y, z, w, h, d = best_placements[i]
            final_placements[i] = (0, x, y, z, w, h, d)
        else:
            # Left out
            # Use original dimensions for left-out boxes as per standard convention? 
            # The prompt says "marks the rest as left out". 
            # The seed program uses boxes[i] dims for left out.
            w, h, d = boxes[i]
            final_placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return final_placements
# EVOLVE-BLOCK-END
