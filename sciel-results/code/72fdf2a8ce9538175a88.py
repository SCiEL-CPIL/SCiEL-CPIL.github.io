"""3D Container Loading - Skyline-based approach with multiple orientations and local search."""
import time
import random
import math
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    t0 = time.time()
    DEADLINE = 115  # safety margin before 120s limit
    
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
    def get_orientations(w, h, d):
        """Return all unique orientations (permutations) of a box."""
        orients = set(permutations([w, h, d]))
        return list(orients)
    
    def can_place(x, y, z, bw, bh, bd, placed):
        """Check if a box at (x,y,z) with dims (bw,bh,bd) doesn't overlap any placed box."""
        for px, py, pz, pw, ph, pd in placed:
            if x < px + pw and x + bw > px and y < py + ph and y + bh > py and z < pz + pd and z + bd > pz:
                return False
        return True
    
    def find_best_position(bw, bh, bd, placed):
        """Find the best (lowest z, then lowest y, then lowest x) position for a box."""
        # Generate candidate x positions: 0 and right edges of placed boxes
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for px, py, pz, pw, ph, pd in placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
        
        # Filter to valid positions within container
        x_candidates = sorted([x for x in x_candidates if x + bw <= W])
        y_candidates = sorted([y for y in y_candidates if y + bh <= H])
        z_candidates = sorted([z for z in z_candidates if z + bd <= D])
        
        best_pos = None
        best_score = float('inf')
        
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if x + bw <= W and y + bh <= H and z + bd <= D:
                        if can_place(x, y, z, bw, bh, bd, placed):
                            score = z * 10000 + y * 100 + x
                            if score < best_score:
                                best_score = score
                                best_pos = (x, y, z)
        
        return best_pos
    
    def place_boxes_order(order, boxes, container):
        """Place boxes in the given order, trying all orientations. Returns (placements, placed_list, filled_vol)."""
        W, H, D = container
        placements = [None] * len(boxes)
        placed = []
        filled_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            best_orient = None
            best_pos = None
            best_score = float('inf')
            
            for ow, oh, od in orients:
                pos = find_best_position(ow, oh, od, placed)
                if pos is not None:
                    x, y, z = pos
                    score = z * 10000 + y * 100 + x
                    if score < best_score:
                        best_score = score
                        best_orient = (ow, oh, od)
                        best_pos = pos
            
            if best_pos is not None:
                x, y, z = best_pos
                ow, oh, od = best_orient
                placements[idx] = (0, x, y, z, ow, oh, od)
                placed.append((x, y, z, ow, oh, od))
                filled_vol += ow * oh * od
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        return placements, placed, filled_vol
    
    def perturb_order(order, n, rng):
        """Apply a random perturbation to the order."""
        order = list(order)
        op = rng.random()
        if op < 0.3:
            # Swap two elements
            i, j = rng.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]
        elif op < 0.5:
            # Move one element
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            elem = order.pop(i)
            order.insert(j, elem)
        elif op < 0.7:
            # Reverse a segment
            i, j = rng.sample(range(n), 2)
            if i > j:
                i, j = j, i
            order[i:j+1] = reversed(order[i:j+1])
        else:
            # Shuffle a small segment
            i = rng.randint(0, n - 1)
            length = min(rng.randint(2, 5), n - i)
            seg = order[i:i+length]
            rng.shuffle(seg)
            order[i:i+length] = seg
        return order
    
    # Initial ordering: sort by volume descending, then by max dimension
    initial_order = list(range(n))
    initial_order.sort(key=lambda i: (-boxes[i][0] * boxes[i][1] * boxes[i][2],
                                       -max(boxes[i])))
    
    # First pass: place in initial order
    placements, placed, filled_vol = place_boxes_order(initial_order, boxes, container)
    best_placements = placements
    best_filled = filled_vol
    best_order = list(initial_order)
    
    # Iterated Local Search
    rng = random.Random(42)
    current_order = list(initial_order)
    current_filled = filled_vol
    
    max_its = 5000
    its = 0
    
    while time.time() - t0 < DEADLINE:
        its += 1
        if its > max_its:
            break
        
        # Perturb
        new_order = perturb_order(current_order, n, rng)
        
        # Place
        new_placements, _, new_filled = place_boxes_order(new_order, boxes, container)
        
        # Accept if better or with some probability (simulated annealing-like)
        if new_filled > current_filled:
            current_order = new_order
            current_filled = new_filled
            if new_filled > best_filled:
                best_filled = new_filled
                best_placements = new_placements
                best_order = list(new_order)
        elif rng.random() < 0.1:
            # Occasionally accept worse solutions to escape local optima
            current_order = new_order
            current_filled = new_filled
    
    return best_placements
