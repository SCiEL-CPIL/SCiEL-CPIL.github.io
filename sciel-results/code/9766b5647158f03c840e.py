"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 150s limit, keep 10s margin for safety/overhead
    deadline = start_time + 140.0
    
    # Preprocess: calculate volume and sort by volume descending
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        vol = w * h * d
        box_data.append((vol, i, w, h, d))
    
    # Sort by volume descending (best-first)
    box_data.sort(key=lambda x: -x[0])
    
    # Get all 6 unique rotations for a box
    def get_rotations(w, h, d):
        rotations = set()
        for perm in permutations([w, h, d]):
            rotations.add(perm)
        return list(rotations)
    
    def find_best_position(bw, bh, bd, placed_boxes, W, H, D):
        """Find the position that minimizes the max coordinate (closest to origin)."""
        best_pos = None
        best_score = None
        
        # Generate candidate positions:
        # 1. Origin
        # 2. Along edges of existing placed boxes
        candidate_positions = set()
        candidate_positions.add((0, 0, 0))
        
        # Add positions adjacent to existing boxes
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            # Right of box
            candidate_positions.add((px + pw, py, pz))
            # Above box
            candidate_positions.add((px, py + ph, pz))
            # Front of box
            candidate_positions.add((px, py, pz + pd))
            # Combinations
            candidate_positions.add((px + pw, py + ph, pz))
            candidate_positions.add((px + pw, py, pz + pd))
            candidate_positions.add((px, py + ph, pz + pd))
            candidate_positions.add((px + pw, py + ph, pz + pd))
        
        # Also add some grid-based candidates (coarse)
        step = max(1, min(W, H, D) // 10)
        for gx in range(0, W, step):
            for gy in range(0, H, step):
                for gz in range(0, D, step):
                    candidate_positions.add((gx, gy, gz))
        
        # Filter and evaluate candidates
        for (cx, cy, cz) in candidate_positions:
            if cx + bw > W or cy + bh > H or cz + bd > D:
                continue
            # Check overlap
            overlap = False
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                if (cx < px + pw and cx + bw > px and
                    cy < py + ph and cy + bh > py and
                    cz < pz + pd and cz + bd > pz):
                    overlap = True
                    break
            if overlap:
                continue
            
            # Score: prefer positions closer to origin (minimize max extent)
            score = cx + cy + cz
            if best_score is None or score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
        
        return best_pos

    def greedy_pack(box_order, W, H, D, deadline):
        """Perform a single greedy packing pass."""
        placements = [None] * n
        placed_boxes = []
        placed_indices = set()
        
        for vol, idx, w, h, d in box_order:
            if time.time() > deadline:
                break
            
            rotations = get_rotations(w, h, d)
            best_placement = None
            best_rot_score = None
            
            for (rw, rh, rd) in rotations:
                pos = find_best_position(rw, rh, rd, placed_boxes, W, H, D)
                if pos is not None:
                    # Score the placement: prefer lower positions
                    score = pos[0] + pos[1] + pos[2]
                    if best_rot_score is None or score < best_rot_score:
                        best_rot_score = score
                        best_placement = (pos, (rw, rh, rd))
            
            if best_placement is not None:
                (px, py, pz), (pw, ph, pd) = best_placement
                placed_boxes.append((px, py, pz, pw, ph, pd))
                placements[idx] = (0, px, py, pz, pw, ph, pd)
                placed_indices.add(idx)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any remaining unplaced boxes
        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                
        return placements, placed_indices

    def local_search(placements, placed_indices, W, H, D, deadline):
        """Attempt to improve the solution by swapping or moving boxes."""
        # Reconstruct placed_boxes list from placements for quick access
        placed_boxes = []
        for i in range(n):
            if i in placed_indices:
                p = placements[i]
                placed_boxes.append((i, p[1], p[2], p[3], p[4], p[5], p[6]))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to move each placed box to a better position
            # We iterate in random order to avoid bias
            indices_to_try = list(placed_indices)
            random.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if time.time() > deadline:
                    break
                
                current_p = placements[idx]
                if current_p[0] != 0:
                    continue
                    
                cur_x, cur_y, cur_z, cur_w, cur_h, cur_d = current_p[1:7]
                
                # Remove current box from placed_boxes
                # We need to find it in the list
                # For efficiency in this small N, linear search is okay
                box_in_list = None
                for b in placed_boxes:
                    if b[0] == idx:
                        box_in_list = b
                        break
                if box_in_list is None:
                    continue
                    
                placed_boxes.remove(box_in_list)
                
                # Try all rotations
                w, h, d = boxes[idx]
                rotations = get_rotations(w, h, d)
                best_new_pos = None
                best_new_rot = None
                best_score = None
                
                for (rw, rh, rd) in rotations:
                    pos = find_best_position(rw, rh, rd, placed_boxes, W, H, D)
                    if pos is not None:
                        score = pos[0] + pos[1] + pos[2]
                        if best_score is None or score < best_score:
                            best_score = score
                            best_new_pos = pos
                            best_new_rot = (rw, rh, rd)
                            
                if best_new_pos is not None:
                    # Check if this is an improvement (lower score than current)
                    current_score = cur_x + cur_y + cur_z
                    if best_score < current_score:
                        # Accept the move
                        nx, ny, nz = best_new_pos
                        nw, nh, nd = best_new_rot
                        placements[idx] = (0, nx, ny, nz, nw, nh, nd)
                        placed_boxes.append((idx, nx, ny, nz, nw, nh, nd))
                        improved = True
                else:
                    # Couldn't place it anywhere? That shouldn't happen if we just removed it,
                    # unless the only place was the one it was in. 
                    # But find_best_position might miss the original spot if it's not a "corner" 
                    # in the new set. So we must restore it if we can't find a better spot.
                    placed_boxes.append(box_in_list)

        return placements, placed_indices

    # 1. Initial Greedy Pass
    initial_placements, initial_placed = greedy_pack(box_data, W, H, D, deadline)
    
    # 2. Local Search Refinement
    # We use the remaining time for local search
    if time.time() < deadline:
        final_placements, final_placed = local_search(initial_placements, initial_placed, W, H, D, deadline)
    else:
        final_placements = initial_placements
        final_placed = initial_placed
        
    return final_placements
# EVOLVE-BLOCK-END
