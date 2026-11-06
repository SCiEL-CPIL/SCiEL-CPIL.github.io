"""
Strategy: Simulated Annealing for 3D Container Loading.

The current program uses a greedy approach with a fixed or locally swapped order.
This approach is structurally different: it treats the problem as a stochastic
optimization of the placement state itself. 

Key differences:
1. State Representation: A list of (box_idx, x, y, z, orientation) for currently placed boxes.
2. Optimization: Simulated Annealing. We start with a greedy seed, then perform random moves:
   - Move a box to a new valid position/orientation.
   - Remove a box.
   - Add a random unplaced box.
   - Swap two boxes' positions (if valid).
3. Objective: Maximize total volume of placed boxes.
4. Escape Mechanism: SA allows temporary worsening of the score (removing a box or moving to a "worse" spot) 
   to escape local minima, which greedy methods cannot do.
"""

import time
import random
import itertools
import math

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    deadline = time.time() + 110.0  # Safety margin
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    
    # Helper to check if a box fits at a position without overlap
    # placed: list of (idx, x, y, z, w, h, d)
    def is_valid(placed, idx, x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        
        # Check overlap with all placed boxes
        # Optimization: Only check boxes that are "close" or just iterate all (N=60 is small)
        for (p_idx, px, py, pz, pw, ph, pd) in placed:
            if p_idx == idx:
                continue
            # Check overlap conditions
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        return True

    def get_volume(placed):
        return sum(p[4]*p[5]*p[6] for p in placed)

    # Initial Greedy Packing to get a good starting state
    def greedy_pack(order):
        placed = []
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_score = None
            
            # Try all orientations
            for (bw, bh, bd) in orient_cache[idx]:
                # Try a set of candidate positions
                # 1. (0,0,0)
                # 2. Adjacent to existing boxes (x+w, y, z), (x, y+h, z), etc.
                candidates = [(0, 0, 0)]
                for (_, px, py, pz, pw, ph, pd) in placed:
                    candidates.append((px + pw, py, pz))
                    candidates.append((px, py + ph, pz))
                    candidates.append((px, py, pz + pd))
                    candidates.append((px + pw, py + ph, pz))
                    candidates.append((px + pw, py, pz + pd))
                    candidates.append((px, py + ph, pz + pd))
                    candidates.append((px + pw, py + ph, pz + pd))
                
                for (cx, cy, cz) in candidates:
                    if is_valid(placed, idx, cx, cy, cz, bw, bh, bd):
                        # Score: prefer lower z, then y, then x (bottom-left-farthest)
                        score = (cz, cy, cx)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
        return placed

    # Generate initial orders
    indexed = list(range(n))
    orders = [
        sorted(indexed, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True),
        sorted(indexed, key=lambda i: max(boxes[i]), reverse=True),
        sorted(indexed, key=lambda i: boxes[i][0]+boxes[i][1]+boxes[i][2], reverse=True),
    ]
    
    best_placed = []
    best_vol = 0
    for order in orders:
        p = greedy_pack(order)
        v = get_volume(p)
        if v > best_vol:
            best_vol = v
            best_placed = p[:]

    current_placed = best_placed[:]
    current_vol = best_vol
    
    # Simulated Annealing
    T = 100.0 # Initial temperature
    cooling = 0.995 # Cooling rate
    min_T = 0.1
    
    # Pre-compute box volumes for quick calculation
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    
    # Map of currently placed indices to their placement details
    # For faster lookup in SA moves
    placed_map = {p[0]: p for p in current_placed}
    unplaced_set = set(indexed) - set(placed_map.keys())
    
    while T > min_T and time.time() < deadline:
        # Generate a neighbor state
        move_type = random.random()
        new_placed = current_placed[:] # Shallow copy, we will modify specific entries
        # Actually, modifying the list directly is safer for SA if we revert properly.
        # Let's work on a copy of the list of tuples.
        
        if move_type < 0.5:
            # Move a placed box to a new position
            if not current_placed:
                T *= cooling
                continue
            idx_to_move = random.choice(current_placed)[0]
            # Remove it temporarily
            temp_placed = [p for p in current_placed if p[0] != idx_to_move]
            
            w, h, d = boxes[idx_to_move]
            moved = False
            for (bw, bh, bd) in orient_cache[idx_to_move]:
                # Try random positions within bounds
                # To make it effective, try positions near existing boxes or random
                attempts = 0
                while attempts < 5:
                    cx = random.randint(0, max(0, W - bw))
                    cy = random.randint(0, max(0, H - bh))
                    cz = random.randint(0, max(0, D - bd))
                    if is_valid(temp_placed, idx_to_move, cx, cy, cz, bw, bh, bd):
                        new_placed = temp_placed + [(idx_to_move, cx, cy, cz, bw, bh, bd)]
                        moved = True
                        break
                    attempts += 1
                if moved:
                    break
            if not moved:
                # Revert to original if no valid move found (or keep original)
                continue
                
        elif move_type < 0.8:
            # Remove a random box
            if not current_placed:
                T *= cooling
                continue
            idx_to_remove = random.choice(current_placed)[0]
            new_placed = [p for p in current_placed if p[0] != idx_to_remove]
            
        else:
            # Add a random unplaced box
            if not unplaced_set:
                T *= cooling
                continue
            idx_to_add = random.choice(list(unplaced_set))
            w, h, d = boxes[idx_to_add]
            added = False
            for (bw, bh, bd) in orient_cache[idx_to_add]:
                # Try to place at a valid spot
                # Heuristic: try (0,0,0) first, then random
                candidates = [(0,0,0)]
                for _ in range(10):
                    cx = random.randint(0, max(0, W - bw))
                    cy = random.randint(0, max(0, H - bh))
                    cz = random.randint(0, max(0, D - bd))
                    candidates.append((cx, cy, cz))
                
                for (cx, cy, cz) in candidates:
                    if is_valid(current_placed, idx_to_add, cx, cy, cz, bw, bh, bd):
                        new_placed = current_placed + [(idx_to_add, cx, cy, cz, bw, bh, bd)]
                        added = True
                        break
                if added:
                    break
            if not added:
                continue

        # Evaluate new state
        new_vol = get_volume(new_placed)
        delta = new_vol - current_vol
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept move
            current_placed = new_placed
            current_vol = new_vol
            
            if current_vol > best_vol:
                best_vol = current_vol
                best_placed = current_placed[:]
                
            # Update placed_map and unplaced_set for next iteration
            placed_map = {p[0]: p for p in current_placed}
            unplaced_set = set(indexed) - set(placed_map.keys())
        else:
            # Reject move, keep current_placed
            pass
            
        T *= cooling

    # Finalize result
    result = [None] * n
    for p in best_placed:
        idx, x, y, z, w, h, d = p
        result[idx] = (0, x, y, z, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return result

# EVOLVE-BLOCK-END
