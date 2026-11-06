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

def _get_rotation_orderings(w, h, d, container):
    """
    Get multiple rotation orderings for a box, each optimized for different packing goals.
    Returns a list of rotation lists, each sorted differently.
    """
    rots = _get_rotations(w, h, d)
    W, H, D = container
    
    # Ordering 1: Prefer lowest z (stack flat)
    order1 = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    
    # Ordering 2: Prefer smallest max dimension (compact)
    order2 = sorted(rots, key=lambda r: (max(r), r[2], r[1], r[0]))
    
    # Ordering 3: Prefer dimensions that match container proportions
    # Try to align largest box dim with largest container dim
    def _match_score(r):
        # Sort both and compare
        br = sorted(r, reverse=True)
        bc = sorted([W, H, D], reverse=True)
        # Penalize mismatch
        penalty = sum(abs(br[i] - bc[i]) for i in range(3))
        return (penalty, r[2], r[1], r[0])
    order3 = sorted(rots, key=_match_score)
    
    # Ordering 4: Prefer tallest (for containers where height matters)
    order4 = sorted(rots, key=lambda r: (-r[2], r[1], r[0]))
    
    # Ordering 5: Prefer smallest footprint
    order5 = sorted(rots, key=lambda r: (r[0] * r[1], r[2], r[1], r[0]))
    
    return [order1, order2, order3, order4, order5]

def _try_place(boxes, container, seed=0, rot_ordering_idx=0):
    """
    Attempt to place boxes into the container using Extreme Point heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute rotations for each box with specified ordering
    box_rotations = []
    for w, h, d in boxes:
        orderings = _get_rotation_orderings(w, h, d, container)
        # Use the specified ordering index, wrapping around
        rots = orderings[rot_ordering_idx % len(orderings)]
        box_rotations.append(rots)
    
    # Sort boxes by volume (largest first) for better packing
    box_indices = list(range(n))
    box_volumes = [w * h * d for w, h, d in boxes]
    box_indices.sort(key=lambda i: -box_volumes[i])
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    # Candidate points for placement (Extreme Points)
    candidates = [(0, 0, 0)]
    
    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False

    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))

    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        # Filter and sort candidates
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
                    
                    if (cx, cy, cz) in candidates:
                        candidates.remove((cx, cy, cz))
                    
                    placed_this = True
                    found = True
                    break
            
            if found:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _try_place_multi(boxes, container, num_trials=10):
    """
    Try multiple different sorting strategies and rotation orderings and return the best result.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_result = None
    best_filled = -1
    
    # Different sorting strategies
    strategies = [
        lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume desc
        lambda i: -boxes[i][0],  # Width desc
        lambda i: -boxes[i][1],  # Height desc
        lambda i: -boxes[i][2],  # Depth desc
        lambda i: -max(boxes[i]),  # Max dim desc
        lambda i: -min(boxes[i]),  # Min dim desc
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume asc
        lambda i: -sum(boxes[i]),  # Sum desc
        lambda i: -max(boxes[i]) / max(min(boxes[i]), 1e-9),  # Aspect ratio (largest/smallest) desc
        lambda i: -(max(boxes[i]) - min(boxes[i])),  # Range desc
        lambda i: -boxes[i][0] * boxes[i][1],  # Footprint area desc
        lambda i: -boxes[i][1] * boxes[i][2],  # Side area desc
    ]
    
    # Number of rotation orderings available
    num_rot_orderings = 5
    
    for s in range(num_trials):
        if time.time() > deadline:
            break
        
        # Use a different strategy or random perturbation
        if s < len(strategies):
            sort_key = strategies[s]
            rot_idx = s % num_rot_orderings
        else:
            sort_key = lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2]
            rot_idx = s % num_rot_orderings
        
        # Call _try_place with the specified rotation ordering
        result = _try_place(boxes, container, seed=s, rot_ordering_idx=rot_idx)
        
        # Calculate filled volume for this result
        filled_volume = 0
        for bin_id, x, y, z, w, h, d in result:
            if bin_id == 0:
                if x >= 0 and y >= 0 and z >= 0 and \
                   x + w <= W and y + h <= H and z + d <= D:
                    filled_volume += w * h * d
        
        if filled_volume > best_filled:
            best_filled = filled_volume
            best_result = result
    
    return best_result if best_result is not None else [(-1, 0, 0, 0, 0, 0, 0)] * n

def _evaluate_result(placement, container):
    """Evaluate the fraction of container volume filled."""
    W, H, D = container
    total_volume = W * H * D
    if total_volume == 0:
        return 0.0
    
    filled = 0
    for bin_id, x, y, z, w, h, d in placement:
        if bin_id == 0:
            # Check if placement is valid
            if x >= 0 and y >= 0 and z >= 0 and \
               x + w <= W and y + h <= H and z + d <= D:
                filled += w * h * d
    
    return filled / total_volume

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
    
    # Use multi-trial approach for better packing with more trials
    # Try 15 trials to cover more strategy/rotation combinations
    placement = _try_place_multi(boxes, container, num_trials=15)
    
    # Verify validity just in case (though _try_place should be correct)
    # If invalid, return empty/failed placements
    # For now, trust the overlap check in _try_place
    
    return placement
# EVOLVE-BLOCK-END
