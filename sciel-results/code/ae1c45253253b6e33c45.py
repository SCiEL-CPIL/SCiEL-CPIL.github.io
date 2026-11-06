"""
Strategy: Skyline/Floor-Map based packing with multi-strategy greedy and randomized restarts.
Instead of checking overlap against all placed boxes (O(n^2) per placement), we maintain
a 2D grid of minimum heights. This allows O(1) overlap checks and efficient placement
at the lowest valid z-coordinate, fundamentally reducing search space complexity.
"""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0  # 5s safety margin for 120s limit
    
    # Precompute orientations for each box
    all_orients = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = list(set((a, b, c) for a in (w, h, d) for b in (w, h, d) for c in (w, h, d) if a == w or a == h or a == d and b == w or b == h or b == d and c == w or c == h or c == d))
        # More robust: just use all unique permutations
        orients = list(set(tuple(sorted((w, h, d)))) and list(set((a, b, c) for a in range(3) for b in range(3) for c in range(3) if len({a,b,c})==3)))
        dims = [w, h, d]
        orients = set()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != j and i != k and j != k:
                        orients.add((dims[i], dims[j], dims[k]))
        all_orients.append(list(orphients) if False else list(orients))
        
    # Precompute volumes
    volumes = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D
    
    # Skyline representation: 2D grid where grid[x][y] = minimum z-height of any box covering (x,y)
    # We'll use a list of (x, y, z) skyline points instead of a full grid for efficiency
    
    def pack_with_skyline(order, strategy='blf'):
        """
        Pack boxes in given order using skyline-based placement.
        strategy: 'blf' = bottom-left-farthest, 'bl' = bottom-left
        Returns (filled_volume, result_list)
        """
        # Skyline: list of (x, y, z) representing the top surface
        # A box can be placed at (x, y) at height z if z >= max height of all skyline points covering (x,y)
        # and the box fits within container bounds and doesn't overlap with existing boxes
        
        # We maintain a list of placed boxes and a set of skyline points
        placed_boxes = []  # (x, y, z, w, h, d)
        skyline = [(0, 0, 0)]  # Start with origin
        
        result = [None] * n
        filled = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_orient = None
            best_cost = float('inf')
            
            # Try all orientations
            for bw, bh, bd in all_orients[idx]:
                if bw > W or bh > H or bd > D:
                    continue
                
                # Generate candidate positions from skyline
                # A valid position is where the box can sit on the "floor" (z=0 or on top of other boxes)
                # Candidate x,y positions come from existing box edges
                candidate_positions = set()
                candidate_positions.add((0, 0))
                
                for (px, py, pz, pw, ph, pd) in placed_boxes:
                    # Right edge
                    candidate_positions.add((px + pw, py))
                    candidate_positions.add((px + pw, py + ph))
                    # Back edge  
                    candidate_positions.add((px, py + ph))
                    candidate_positions.add((px + pw, py + ph))
                    # Also try positions along the footprint
                    if px + pw < W:
                        candidate_positions.add((px + pw, py))
                    if py + ph < H:
                        candidate_positions.add((px, py + ph))
                
                for (cx, cy) in candidate_positions:
                    if cx + bw > W or cy + bh > H:
                        continue
                    
                    # Find the minimum z-height at this position
                    # The box sits at z = max of all heights of boxes whose footprint overlaps (cx, cy, cx+bw, cy+bh)
                    min_z = 0
                    for (px, py, pz, pw, ph, pd) in placed_boxes:
                        # Check if footprints overlap
                        if cx < px + pw and cx + bw > px and cy < py + ph and cy + bh > py:
                            # They overlap in xy, so this box must be on top of it
                            min_z = max(min_z, pz + pd)
                    
                    if min_z + bd > D:
                        continue
                    
                    # Check for actual overlap with all placed boxes (safety check)
                    valid = True
                    for (px, py, pz, pw, ph, pd) in placed_boxes:
                        if not (cx + bw <= px or px + pw <= cx or
                                cy + bh <= py or py + ph <= cy or
                                min_z + bd <= pz or pz + pd <= min_z):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Compute cost
                    if strategy == 'blf':
                        # Bottom-Left-Farthest: minimize x+y, then maximize z
                        cost = (cx + cy, -min_z)
                    else:
                        # Bottom-Left: minimize x+y+z
                        cost = (cx + cy + min_z,)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (cx, cy, min_z)
                        best_orient = (bw, bh, bd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed_boxes.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled += bw * bh * bd
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
        
        return filled, result
    
    # Initial Order: Volume Descending
    initial_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    best_filled, best_result = pack_with_skyline(initial_order, 'blf')
    
    # Try multiple strategies and randomized restarts
    strategies = ['blf', 'bl']
    sort_keys = [
        lambda i: volumes[i],
        lambda i: -boxes[i][0] - boxes[i][1] - boxes[i][2],  # largest dimension
        lambda i: -max(boxes[i]),
        lambda i: -min(boxes[i]),
    ]
    
    # Deterministic multi-strategy
    for strategy in strategies:
        for sort_key in sort_keys:
            if time.time() > deadline - 15.0:
                break
            order = sorted(range(n), key=sort_key, reverse=True)
            filled, res = pack_with_skyline(order, strategy)
            if filled > best_filled:
                best_filled = filled
                best_result = res
    
    # Randomized restarts with simulated annealing on order
    if time.time() < deadline - 15.0:
        current_order = initial_order[:]
        current_filled, current_result = pack_with_skyline(current_order, 'blf')
        
        T = 500.0
        cooling = 0.998
        min_T = 1.0
        
        while T > min_T and time.time() < deadline - 10.0:
            # Pick two random indices to swap
            i, j = random.sample(range(n), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            new_filled, new_result = pack_with_skyline(current_order, 'blf')
            delta = new_filled - current_filled
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_filled = new_filled
                current_result = new_result
                if current_filled > best_filled:
                    best_filled = current_filled
                    best_result = current_result
            else:
                current_order[i], current_order[j] = current_order[j], current_order[i]
            
            T *= cooling
    
    # Fallback
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result

# EVOLVE-BLOCK-END
