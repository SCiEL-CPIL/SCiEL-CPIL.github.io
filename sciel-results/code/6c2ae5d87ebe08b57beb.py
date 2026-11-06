"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations
from collections import defaultdict
import numpy as np

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _pack_boxes(boxes, container, order_indices, time_limit, rotation_sort_key=None):
    """
    Attempt to place boxes into the container using Extreme Point heuristic with spatial index.
    Returns (result, filled_volume) where result is a list of placements.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        if rotation_sort_key is not None:
            rots.sort(key=rotation_sort_key)
        else:
            # Default: prefer orientations that align large dims with container's large dims
            # Sort by how well the box fits: prefer smaller z, then y, then x
            rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    # Spatial index: for fast overlap checking
    # Use a dictionary mapping (x, y, z) corner to list of box indices that start there
    # Actually, let's use a simpler approach: maintain a list and do O(n) check but with early exit
    # For 60 boxes, O(n^2) is fine: 60*60 = 3600 checks max
    
    # Candidate points for placement (Extreme Points)
    candidates = set([(0, 0, 0)])
    
    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False

    def point_is_inside_existing(x, y, z):
        """Check if point (x,y,z) is strictly inside any existing box."""
        for ex, ey, ez, ew, eh, ed in placed:
            if ex < x < ex + ew and ey < y < ey + eh and ez < z < ez + ed:
                return True
        return False

    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            if not point_is_inside_existing(x, y, z):
                candidates.add((x, y, z))

    for idx in order_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        # Sort candidates by Z (lowest), then Y (lowest), then X (leftmost)
        # This promotes compact packing towards the origin
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            found = False
            for cx, cy, cz in valid_candidates:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                if not overlaps_any(cx, cy, cz, bw, bh, bd):
                    placed.append((cx, cy, cz, bw, bh, bd))
                    result[idx] = (0, cx, cy, cz, bw, bh, bd)
                    filled_volume += bw * bh * bd
                    
                    add_candidate(cx + bw, cy, cz)
                    add_candidate(cx, cy + bh, cz)
                    add_candidate(cx, cy, cz + bd)
                    
                    # Remove the used candidate
                    candidates.discard((cx, cy, cz))
                    
                    placed_this = True
                    found = True
                    break
            
            if found:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_volume

def _pack_boxes_best_fit(boxes, container, order_indices, time_limit):
    """
    Best-fit placement: for each box, try all candidate points and rotations,
    and pick the placement that minimizes the "gap" (max distance to container walls).
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    placed = []
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    candidates = set([(0, 0, 0)])
    
    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False

    def point_is_inside_existing(x, y, z):
        for ex, ey, ez, ew, eh, ed in placed:
            if ex < x < ex + ew and ey < y < ey + eh and ez < z < ez + ed:
                return True
        return False

    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            if not point_is_inside_existing(x, y, z):
                candidates.add((x, y, z))

    for idx in order_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        best_placement = None
        best_score = float('inf')
        
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        
        for bw, bh, bd in rots:
            for cx, cy, cz in valid_candidates:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                if not overlaps_any(cx, cy, cz, bw, bh, bd):
                    # Score: prefer placements that are close to existing boxes (compactness)
                    # and close to the origin
                    score = cx + cy + cz
                    # Also penalize if it creates a large gap
                    if score < best_score:
                        best_score = score
                        best_placement = (cx, cy, cz, bw, bh, bd)
        
        if best_placement is not None:
            cx, cy, cz, bw, bh, bd = best_placement
            placed.append((cx, cy, cz, bw, bh, bd))
            result[idx] = (0, cx, cy, cz, bw, bh, bd)
            filled_volume += bw * bh * bd
            
            add_candidate(cx + bw, cy, cz)
            add_candidate(cx, cy + bh, cz)
            add_candidate(cx, cy, cz + bd)
            candidates.discard((cx, cy, cz))
            placed_this = True
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_volume

def _shake_improve(boxes, container, result, time_limit, max_shakes=50):
    """
    Try to improve packing by removing and re-inserting boxes.
    """
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Extract current placements
    placed = []
    for i in range(n):
        if result[i][0] == 0:
            _, x, y, z, w, h, d = result[i]
            placed.append((i, x, y, z, w, h, d))
    
    current_filled = sum(w * h * d for _, _, _, _, w, h, d in placed)
    
    # Try shaking: remove a box, then re-insert it (and maybe others)
    for shake_iter in range(max_shakes):
        if time.time() > deadline:
            break
        
        # Pick a random placed box to remove
        if not placed:
            break
        
        removed_idx = random.choice(placed)
        box_i = removed_idx[0]
        x, y, z, w, h, d = removed_idx[1:7]
        
        # Remove from placed
        placed = [p for p in placed if p[0] != box_i]
        
        # Update result
        result[box_i] = (-1, 0, 0, 0, w, h, d)
        
        # Try to re-insert this box and any other unplaced boxes
        unplaced = [i for i in range(n) if result[i][0] == -1]
        # Sort by volume descending
        unplaced.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
        
        new_placed = list(placed)
        new_result = [r for r in result]
        new_filled = current_filled - w * h * d
        
        candidates = set([(0, 0, 0)])
        # Rebuild candidates from existing placed boxes
        for _, px, py, pz, pw, ph, pd in new_placed:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
        
        def overlaps_any_local(px, py, pz, pw, ph, pd):
            for _, ex, ey, ez, ew, eh, ed in new_placed:
                if (px < ex + ew and px + pw > ex and
                    py < ey + eh and py + ph > ey and
                    pz < ez + ed and pz + pd > ez):
                    return True
            return False

        def point_inside_local(x, y, z):
            for _, ex, ey, ez, ew, eh, ed in new_placed:
                if ex < x < ex + ew and ey < y < ey + eh and ez < z < ez + ed:
                    return True
            return False

        def add_candidate_local(x, y, z):
            if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
                if not point_inside_local(x, y, z):
                    candidates.add((x, y, z))

        # Try to place the removed box and others
        for bidx in unplaced:
            if time.time() > deadline:
                break
            
            bw, bh, bd = boxes[bidx]
            rots = _get_rotations(bw, bh, bd)
            rots.sort(key=lambda r: (r[2], r[1], r[0]))
            
            placed_this = False
            valid_cands = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
            valid_cands.sort(key=lambda p: (p[2], p[1], p[0]))
            
            for rbw, rbh, rbd in rots:
                found = False
                for cx, cy, cz in valid_cands:
                    if cx + rbw > W or cy + rbh > H or cz + rbd > D:
                        continue
                    if not overlaps_any_local(cx, cy, cz, rbw, rbh, rbd):
                        new_placed.append((bidx, cx, cy, cz, rbw, rbh, rbd))
                        new_result[bidx] = (0, cx, cy, cz, rbw, rbh, rbd)
                        new_filled += rbw * rbh * rbd
                        add_candidate_local(cx + rbw, cy, cz)
                        add_candidate_local(cx, cy + rbh, cz)
                        add_candidate_local(cx, cy, cz + rbd)
                        candidates.discard((cx, cy, cz))
                        placed_this = True
                        found = True
                        break
                if found:
                    break
            
            if not placed_this:
                new_result[bidx] = (-1, 0, 0, 0, bw, bh, bd)
        
        # If improvement, accept
        if new_filled > current_filled:
            result = new_result
            placed = new_placed
            current_filled = new_filled
    
    return result, current_filled

def load(boxes, container):
    """
    Main entry point: place boxes into container.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Multiple sorting strategies
    order1 = list(range(n))
    order1.sort(key=lambda i: -box_volumes[i])  # Volume descending
    
    order2 = list(range(n))
    order2.sort(key=lambda i: -max(boxes[i]))  # Max dimension descending
    
    order3 = list(range(n))
    order3.sort(key=lambda i: -min(boxes[i]))  # Min dimension descending
    
    order4 = list(range(n))
    order4.sort(key=lambda i: -sum(boxes[i]))  # Sum of dimensions
    
    order5 = list(range(n))
    order5.sort(key=lambda i: box_volumes[i])  # Volume ascending
    
    random.seed(42)
    order6 = list(range(n))
    random.shuffle(order6)  # Random
    
    order7 = list(range(n))
    order7.sort(key=lambda i: -max(boxes[i]) / max(min(boxes[i]), 1e-9))  # Aspect ratio
    
    order8 = list(range(n))
    order8.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))  # Surface area
    
    orders = [order1, order2, order3, order4, order5, order6, order7, order8]
    
    best_result = None
    best_filled = -1
    
    # Run multiple strategies with basic packing
    for order in orders:
        if time.time() > deadline:
            break
        res, filled = _pack_boxes(boxes, container, order, 110.0)
        if filled > best_filled:
            best_filled = filled
            best_result = res
    
    # Try best-fit packing with some orders
    for order in [order1, order2, order4]:
        if time.time() > deadline:
            break
        res, filled = _pack_boxes_best_fit(boxes, container, order, 100.0)
        if filled > best_filled:
            best_filled = filled
            best_result = res
    
    # If we have time, try shake improvement on the best result
    if best_result is not None and time.time() < deadline - 30.0:
        improved_result, improved_filled = _shake_improve(boxes, container, best_result, deadline - time.time(), max_shakes=30)
        if improved_filled > best_filled:
            best_filled = improved_filled
            best_result = improved_result
    
    # Try more random orders if time remains
    if best_result is not None and time.time() < deadline - 15.0:
        for _ in range(10):
            if time.time() > deadline:
                break
            random_order = list(range(n))
            random.shuffle(random_order)
            res, filled = _pack_boxes(boxes, container, random_order, 100.0)
            if filled > best_filled:
                best_filled = filled
                best_result = res
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    
    return best_result
# EVOLVE-BLOCK-END
