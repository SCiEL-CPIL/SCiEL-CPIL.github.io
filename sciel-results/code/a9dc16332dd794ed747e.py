"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations
import numpy as np

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _skyline_pack(boxes, container, order, time_limit, strategy="bottom_left"):
    """
    Pack boxes using a skyline-based approach with multiple strategies.
    strategy: "bottom_left" (min x+y+z), "bottom_front" (min z then y then x),
              "best_fit" (minimize max gap), "max_contact" (maximize contact with existing)
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
        # Sort: prefer smaller z for stacking, then smaller y, then smaller x
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    placed = []  # (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    # Extreme points (candidate positions)
    candidates = set([(0, 0, 0)])
    
    def overlaps(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False
    
    def inside_any(x, y, z):
        for ex, ey, ez, ew, eh, ed in placed:
            if ex < x < ex + ew and ey < y < ey + eh and ez < z < ez + ed:
                return True
        return False
    
    def add_cand(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            if not inside_any(x, y, z):
                candidates.add((x, y, z))
    
    def score_placement(cx, cy, cz, bw, bh, bd):
        """Score a placement; lower is better."""
        if strategy == "bottom_left":
            return cx + cy + cz
        elif strategy == "bottom_front":
            return (cz, cy, cx)
        elif strategy == "best_fit":
            # Minimize the maximum remaining space in each dimension
            return max(W - cx - bw, H - cy - bh, D - cz - bd)
        elif strategy == "max_contact":
            # Maximize contact with existing boxes (more contact = better)
            contact = 0
            for ex, ey, ez, ew, eh, ed in placed:
                # Check face contact
                if abs(cx - (ex + ew)) < 1e-9 and not (cy + bh <= ey or cy >= ey + eh) and not (cz + bd <= ez or cz >= ez + ed):
                    contact += min(bh, eh) * min(bd, ed)
                if abs(cy - (ey + eh)) < 1e-9 and not (cx + bw <= ex or cx >= ex + ew) and not (cz + bd <= ez or cz >= ez + ed):
                    contact += min(bw, ew) * min(bd, ed)
                if abs(cz - (ez + ed)) < 1e-9 and not (cx + bw <= ex or cx >= ex + ew) and not (cy + bh <= ey or cy >= ey + eh):
                    contact += min(bw, ew) * min(bh, eh)
            return -contact  # Negate so lower is better
        else:
            return cx + cy + cz
    
    for idx in order:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        valid_cands = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        
        best_place = None
        best_score_val = float('inf')
        
        for bw, bh, bd in rots:
            for cx, cy, cz in valid_cands:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                if overlaps(cx, cy, cz, bw, bh, bd):
                    continue
                s = score_placement(cx, cy, cz, bw, bh, bd)
                if strategy == "bottom_front":
                    # Tuple comparison
                    if s < best_score_val:
                        best_score_val = s
                        best_place = (cx, cy, cz, bw, bh, bd)
                else:
                    if s < best_score_val:
                        best_score_val = s
                        best_place = (cx, cy, cz, bw, bh, bd)
        
        if best_place is not None:
            cx, cy, cz, bw, bh, bd = best_place
            placed.append((cx, cy, cz, bw, bh, bd))
            result[idx] = (0, cx, cy, cz, bw, bh, bd)
            filled_volume += bw * bh * bd
            
            add_cand(cx + bw, cy, cz)
            add_cand(cx, cy + bh, cz)
            add_cand(cx, cy, cz + bd)
            candidates.discard((cx, cy, cz))
    
    return result, filled_volume

def _local_search_improve(boxes, container, result, time_limit, max_iters=200):
    """
    Local search: try removing and re-inserting boxes in different orders to improve packing.
    Uses a simple hill-climbing approach.
    """
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Work on a copy
    cur_result = [r for r in result]
    cur_filled = sum(w * h * d for r in cur_result if r[0] == 0 for w, h, d in [r[4:7]])
    
    box_volumes = [w * h * d for w, h, d in boxes]
    
    best_result = [r for r in cur_result]
    best_filled = cur_filled
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random loaded box to remove
        loaded = [i for i in range(n) if cur_result[i][0] == 0]
        if not loaded:
            break
        
        remove_idx = random.choice(loaded)
        
        # Remove it
        saved_placement = cur_result[remove_idx]
        cur_result[remove_idx] = (-1, 0, 0, 0, boxes[remove_idx][0], boxes[remove_idx][1], boxes[remove_idx][2])
        cur_filled -= saved_placement[4] * saved_placement[5] * saved_placement[6]
        
        # Try to reinsert it and other unplaced boxes
        unplaced = [i for i in range(n) if cur_result[i][0] == -1]
        # Sort by volume descending for reinsertion
        unplaced.sort(key=lambda i: -box_volumes[i])
        
        # Rebuild placed list
        placed = []
        for i in range(n):
            if cur_result[i][0] == 0:
                _, x, y, z, w, h, d = cur_result[i]
                placed.append((x, y, z, w, h, d))
        
        candidates = set([(0, 0, 0)])
        for px, py, pz, pw, ph, pd in placed:
            for cx, cy, cz in [(px + pw, py, pz), (px, py + ph, pz), (px, py, pz + pd)]:
                if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D:
                    inside = False
                    for ex, ey, ez, ew, eh, ed in placed:
                        if ex < cx < ex + ew and ey < cy < ey + eh and ez < cz < ez + ed:
                            inside = True
                            break
                    if not inside:
                        candidates.add((cx, cy, cz))
        
        def overlaps_local(px, py, pz, pw, ph, pd):
            for ex, ey, ez, ew, eh, ed in placed:
                if (px < ex + ew and px + pw > ex and
                    py < ey + eh and py + ph > ey and
                    pz < ez + ed and pz + pd > ez):
                    return True
            return False
        
        # Try to place unplaced boxes
        for bidx in unplaced:
            if time.time() > deadline:
                break
            bw, bh, bd = boxes[bidx]
            rots = _get_rotations(bw, bh, bd)
            rots.sort(key=lambda r: (r[2], r[1], r[0]))
            
            valid_cands = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
            valid_cands.sort(key=lambda p: (p[2], p[1], p[0]))
            
            placed_this = False
            for rbw, rbh, rbd in rots:
                for cx, cy, cz in valid_cands:
                    if cx + rbw > W or cy + rbh > H or cz + rbd > D:
                        continue
                    if overlaps_local(cx, cy, cz, rbw, rbh, rbd):
                        continue
                    # Place it
                    placed.append((cx, cy, cz, rbw, rbh, rbd))
                    cur_result[bidx] = (0, cx, cy, cz, rbw, rbh, rbd)
                    cur_filled += rbw * rbh * rbd
                    
                    for nx, ny, nz in [(cx + rbw, cy, cz), (cx, cy + rbh, cz), (cx, cy, cz + rbd)]:
                        if 0 <= nx <= W and 0 <= ny <= H and 0 <= nz <= D:
                            inside = False
                            for ex, ey, ez, ew, eh, ed in placed:
                                if ex < nx < ex + ew and ey < ny < ey + eh and ez < nz < ez + ed:
                                    inside = True
                                    break
                            if not inside:
                                candidates.add((nx, ny, nz))
                    candidates.discard((cx, cy, cz))
                    placed_this = True
                    break
                if placed_this:
                    break
        
        # Accept if improved
        if cur_filled > best_filled:
            best_filled = cur_filled
            best_result = [r for r in cur_result]
    
    return best_result, best_filled

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
    deadline = start_time + 115.0  # Safety margin
    
    box_volumes = [w * h * d for w, h, d in boxes]
    random.seed(42)
    
    # Generate multiple orderings
    orders = []
    
    # 1. Volume descending
    o = list(range(n))
    o.sort(key=lambda i: -box_volumes[i])
    orders.append(o)
    
    # 2. Max dimension descending
    o = list(range(n))
    o.sort(key=lambda i: -max(boxes[i]))
    orders.append(o)
    
    # 3. Sum of dimensions descending
    o = list(range(n))
    o.sort(key=lambda i: -sum(boxes[i]))
    orders.append(o)
    
    # 4. Surface area descending
    o = list(range(n))
    o.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
    orders.append(o)
    
    # 5. Aspect ratio (max/min) descending
    o = list(range(n))
    o.sort(key=lambda i: -max(boxes[i]) / max(min(boxes[i]), 1e-9))
    orders.append(o)
    
    # 6. Random
    o = list(range(n))
    random.shuffle(o)
    orders.append(o)
    
    # 7. Volume ascending
    o = list(range(n))
    o.sort(key=lambda i: box_volumes[i])
    orders.append(o)
    
    # 8. Min dimension descending
    o = list(range(n))
    o.sort(key=lambda i: -min(boxes[i]))
    orders.append(o)
    
    best_result = None
    best_filled = -1
    
    strategies = ["bottom_left", "bottom_front", "best_fit", "max_contact"]
    
    # Try each ordering with each strategy
    for order in orders:
        if time.time() > deadline - 10:
            break
        for strat in strategies:
            if time.time() > deadline - 5:
                break
            res, filled = _skyline_pack(boxes, container, order, 110.0, strategy=strat)
            if filled > best_filled:
                best_filled = filled
                best_result = [r for r in res]
    
    # Local search improvement on best result
    if best_result is not None and time.time() < deadline - 20:
        improved, improved_filled = _local_search_improve(boxes, container, best_result, deadline - time.time() - 5, max_iters=100)
        if improved_filled > best_filled:
            best_filled = improved_filled
            best_result = improved
    
    # More random orders if time remains
    if best_result is not None and time.time() < deadline - 15:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            random_order = list(range(n))
            random.shuffle(random_order)
            for strat in ["bottom_left", "best_fit"]:
                if time.time() > deadline - 5:
                    break
                res, filled = _skyline_pack(boxes, container, random_order, 100.0, strategy=strat)
                if filled > best_filled:
                    best_filled = filled
                    best_result = [r for r in res]
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    
    return best_result
# EVOLVE-BLOCK-END
