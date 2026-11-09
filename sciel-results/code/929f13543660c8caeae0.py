"""3D Container Loading – multi-start shelf-based packer with spatial index."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 125.0

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return elapsed() > DEADLINE

    def rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots

    # 3D spatial hash for fast overlap queries
    # We store placed boxes in a dict keyed by (x,y,z) cells they occupy (coarse)
    # For simplicity and speed, we use a list of placed boxes and do linear scan,
    # but we also maintain a 3D occupancy grid for quick rejection.
    
    class PlacedBoxes:
        def __init__(self, W, H, D):
            self.W = W
            self.H = H
            self.D = D
            self.boxes = []  # list of (x,y,z,w,h,d)
            # Coarse grid: cell size 1, but we only store boxes, not cells
            # For speed, we'll use a simple list and linear check, but with early exit
        
        def add(self, box):
            self.boxes.append(box)
        
        def can_place(self, x, y, z, w, h, d):
            if x + w > self.W or y + h > self.H or z + d > self.D:
                return False
            for px, py, pz, pw, ph, pd in self.boxes:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return False
            return True
        
        def candidate_positions(self):
            """Generate candidate (x,y,z) positions based on existing placements."""
            candidates = set()
            candidates.add((0, 0, 0))
            for px, py, pz, pw, ph, pd in self.boxes:
                # Right
                if px + pw < self.W:
                    candidates.add((px + pw, py, pz))
                # Above
                if py + ph < self.H:
                    candidates.add((px, py + ph, pz))
                # Front
                if pz + pd < self.D:
                    candidates.add((px, py, pz + pd))
                # Diagonals
                if px + pw < self.W and py + ph < self.H:
                    candidates.add((px + pw, py + ph, pz))
                if px + pw < self.W and pz + pd < self.D:
                    candidates.add((px + pw, py, pz + pd))
                if py + ph < self.H and pz + pd < self.D:
                    candidates.add((px, py + ph, pz + pd))
                if px + pw < self.W and py + ph < self.H and pz + pd < self.D:
                    candidates.add((px + pw, py + ph, pz + pd))
                # Also try positions slightly offset to fill gaps
                if px + pw < self.W:
                    candidates.add((px + pw, 0, 0))
                if py + ph < self.H:
                    candidates.add((0, py + ph, 0))
                if pz + pd < self.D:
                    candidates.add((0, 0, pz + pd))
            
            valid = []
            for (x, y, z) in candidates:
                if 0 <= x < self.W and 0 <= y < self.H and 0 <= z < self.D:
                    valid.append((x, y, z))
            return valid
        
        def volume_filled(self):
            return sum(w*h*d for (_,_,_,w,h,d) in self.boxes)

    def try_place_box(w, h, d, placed, deadline_check):
        rots = rotations(w, h, d)
        best = None
        best_score = float('inf')
        
        positions = placed.candidate_positions()
        
        for (rw, rh, rd) in rots:
            for (x, y, z) in positions:
                if not placed.can_place(x, y, z, rw, rh, rd):
                    continue
                # Score: prefer low y, low x, low z, and compact
                sc = y * 1.0 + x * 0.5 + z * 0.3
                if sc < best_score:
                    best_score = sc
                    best = (x, y, z, rw, rh, rd)
                if deadline_check():
                    return best
            if deadline_check():
                return best
        
        return best

    def pack_with_order(order, boxes, W, H, D, deadline_check):
        placed = PlacedBoxes(W, H, D)
        result = [None] * len(boxes)
        
        for idx in order:
            if deadline_check():
                break
            w, h, d = boxes[idx]
            pos = try_place_box(w, h, d, placed, deadline_check)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.add((x, y, z, rw, rh, rd))
                result[idx] = (0, x, y, z, rw, rh, rd)
        
        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        
        return result

    def volume_filled(placements):
        return sum(w*h*d for (b, x, y, z, w, h, d) in placements if b == 0)

    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    # Strategy 1: Sort by volume descending
    order1 = sorted(indexed_boxes, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))
    order1 = [t[0] for t in order1]
    result1 = pack_with_order(order1, boxes, W, H, D, check_deadline)
    best_result = result1
    best_vol = volume_filled(result1)
    
    if check_deadline():
        return best_result
    
    # Strategy 2: Sort by max dimension descending
    order2 = sorted(indexed_boxes, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))
    order2 = [t[0] for t in order2]
    if order2 != order1:
        result2 = pack_with_order(order2, boxes, W, H, D, check_deadline)
        vol2 = volume_filled(result2)
        if vol2 > best_vol:
            best_vol = vol2
            best_result = result2
    
    if check_deadline():
        return best_result
    
    # Strategy 3: Sort by surface area descending
    order3 = sorted(indexed_boxes, key=lambda t: (-2*(t[1]*t[2] + t[2]*t[3] + t[1]*t[3]), -t[1]*t[2]*t[3]))
    order3 = [t[0] for t in order3]
    if order3 != order1 and order3 != order2:
        result3 = pack_with_order(order3, boxes, W, H, D, check_deadline)
        vol3 = volume_filled(result3)
        if vol3 > best_vol:
            best_vol = vol3
            best_result = result3
    
    if check_deadline():
        return best_result
    
    # Strategy 4: Sort by height (smallest dimension) descending – encourages flat stacking
    order4 = sorted(indexed_boxes, key=lambda t: (-min(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))
    order4 = [t[0] for t in order4]
    if order4 != order1 and order4 != order2 and order4 != order3:
        result4 = pack_with_order(order4, boxes, W, H, D, check_deadline)
        vol4 = volume_filled(result4)
        if vol4 > best_vol:
            best_vol = vol4
            best_result = result4
    
    if check_deadline():
        return best_result
    
    # Random restarts
    rng = random.Random(42)
    for restart in range(30):
        if check_deadline():
            break
        shuffled = list(indexed_boxes)
        rng.shuffle(shuffled)
        # Partial sort: top half by volume
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        # Perturb
        for _ in range(len(shuffled) // 3):
            i = rng.randint(0, len(shuffled) - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        order_r = [t[0] for t in shuffled]
        result_r = pack_with_order(order_r, boxes, W, H, D, check_deadline)
        vol_r = volume_filled(result_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = result_r
    
    return best_result
# EVOLVE-BLOCK-END
