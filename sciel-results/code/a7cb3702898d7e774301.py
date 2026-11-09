"""3D Container Loading - multi-strategy greedy with local improvement."""
import time
import itertools
import random
import math
from collections import defaultdict

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    """Generate candidate positions from placed boxes."""
    W, H, D = container
    points = set()
    points.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        candidates = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px + pw, py, pz + pd),
            (px, py + ph, pz + pd),
            (px + pw, py + ph, pz + pd),
            (px + pw, py + ph, pz),  # duplicate, but set handles it
        ]
        for c in candidates:
            if 0 <= c[0] < W and 0 <= c[1] < H and 0 <= c[2] < D:
                points.add(c)
    return list(points)

def _face_positions(placed, container):
    """Generate candidate positions based on face alignments."""
    W, H, D = container
    points = set()
    points.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        # Align to faces of placed boxes
        for x in [px, px + pw]:
            for y in [py, py + ph]:
                for z in [pz, pz + pd]:
                    if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                        points.add((x, y, z))
    return list(points)

def _score_position_compact(pos, box_dims, placed, container):
    """Score: prefer positions closer to origin (compactness)."""
    x, y, z = pos
    W, H, D = container
    # Weighted distance from origin, emphasizing z (bottom) most
    return z * (W * H) + y * W + x

def _score_position_low(pos, box_dims, placed, container):
    """Score: prefer lowest z, then lowest y, then lowest x."""
    x, y, z = pos
    W, H, D = container
    # Strong preference for low z
    return z * 1000000 + y * 1000 + x

def _score_position_balanced(pos, box_dims, placed, container):
    """Score: balance between compactness and filling gaps."""
    x, y, z = pos
    W, H, D = container
    # Distance from center, but weighted toward bottom
    cx, cy, cz = W / 2, H / 2, D / 2
    dist = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    # Penalize high z more
    z_penalty = z * 100
    return dist + z_penalty

def _greedy_load(boxes, container, deadline, seed=42, score_func=_score_position_compact, max_candidates=800):
    """Greedy placement with extreme points and rotation selection."""
    rng = random.Random(seed)
    W, H, D = container
    placed = []
    result = [None] * len(boxes)
    
    # Sort boxes by volume descending
    box_indices = list(range(len(boxes)))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in box_indices]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    # Precompute rotations
    box_rots = [_rotations(*boxes[i]) for i in range(len(boxes))]
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Get candidate positions from both extreme points and face positions
        candidates = _extreme_points(placed, container)
        face_cands = _face_positions(placed, container)
        all_cands = list(set(candidates + face_cands))
        
        # Limit candidates for speed
        if len(all_cands) > max_candidates:
            all_cands.sort(key=lambda p: score_func(p, (w, h, d), placed, container))
            all_cands = all_cands[:max_candidates]
        
        for rot in box_rots[idx]:
            rw, rh, rd = rot
            for pos in all_cands:
                if not _fits(rot, pos, container):
                    continue
                if not _no_overlap(rot, pos, placed):
                    continue
                sc = score_func(pos, rot, placed, container)
                if sc < best_score:
                    best_score = sc
                    best_pos = pos
                    best_rot = rot
        
        if best_pos is not None:
            x, y, z = best_pos
            placed.append((x, y, z, best_rot[0], best_rot[1], best_rot[2]))
            result[idx] = (0, x, y, z, best_rot[0], best_rot[1], best_rot[2])
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _shelf_greedy(boxes, container, deadline, seed=42):
    """Shelf-based greedy packing."""
    rng = random.Random(seed)
    W, H, D = container
    placements = [None] * len(boxes)
    
    # Sort by height (y-dimension) descending for shelf packing
    box_indices = list(range(len(boxes)))
    box_indices.sort(key=lambda i: boxes[i][1], reverse=True)
    
    shelves = []  # list of (y_start, y_end, x_cursor, boxes_on_shelf)
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        best_shelf = None
        best_x = None
        best_rot = None
        best_y = None
        
        # Try all rotations
        for rot in _rotations(w, h, d):
            rw, rh, rd = rot
            
            # Try to place on existing shelves
            for shelf in shelves:
                y_start, y_end, x_cursor, shelf_boxes = shelf
                if y_start + rh <= H and x_cursor + rw <= W:
                    # Check overlap with boxes on this shelf
                    can_place = True
                    for (bx, by, bz, bw, bh, bd) in shelf_boxes:
                        if (x_cursor < bx + bw and x_cursor + rw > bx and
                            y_start < by + bh and y_start + rh > by and
                            0 < bz + bd and 0 + rd > bz):  # z overlap check
                            can_place = False
                            break
                    if can_place:
                        if best_shelf is None or y_start < best_y:
                            best_shelf = shelf
                            best_x = x_cursor
                            best_rot = rot
                            best_y = y_start
            
            # Try new shelf
            if not shelves or shelves[-1][1] + rh <= H:
                new_y = shelves[-1][1] if shelves else 0
                if new_y + rh <= H:
                    if best_shelf is None or new_y < best_y:
                        best_shelf = None  # new shelf
                        best_x = 0
                        best_rot = rot
                        best_y = new_y
        
        if best_rot is not None:
            rw, rh, rd = best_rot
            if best_shelf is None:
                # New shelf
                shelves.append((best_y, best_y + rh, best_x + rw, []))
            else:
                best_shelf[2] = best_x + rw
                best_shelf[1] = max(best_shelf[1], best_y + rh)
            
            # Find z position (lowest available)
            z = 0
            placed_z = []
            for (bx, by, bz, bw, bh, bd) in [s[3] for s in shelves if s[3]]:
                placed_z.append(bz)
            for pz in sorted(placed_z):
                if z + rd <= D:
                    break
                z = pz + rd
            
            if best_shelf is not None:
                best_shelf[3].append((best_x, best_y, z, rw, rh, rd))
            else:
                shelves[-1][3].append((best_x, best_y, z, rw, rh, rd))
            
            placements[idx] = (0, best_x, best_y, z, rw, rh, rd)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements

def _local_search(boxes, container, placements, deadline, max_iterations=50):
    """Local search: try to improve by moving boxes to better positions."""
    W, H, D = container
    placed = [(x, y, z, w, h, d) for (bin_, x, y, z, w, h, d) in placements if bin_ == 0]
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try to move each placed box to a better position
        for i, (bin_, x, y, z, w, h, d) in enumerate(placements):
            if bin_ != 0:
                continue
            
            # Remove this box temporarily
            placed_idx = None
            for j, (px, py, pz, pw, ph, pd) in enumerate(placed):
                if (px, py, pz, pw, ph, pd) == (x, y, z, w, h, d):
                    placed_idx = j
                    break
            
            if placed_idx is None:
                continue
            
            temp_placed = placed[:placed_idx] + placed[placed_idx+1:]
            
            # Find best new position
            best_pos = None
            best_rot = None
            best_score = _score_position_compact((x, y, z), (w, h, d), temp_placed, container)
            
            candidates = _extreme_points(temp_placed, container)
            if len(candidates) > 300:
                candidates.sort(key=lambda p: _score_position_compact(p, (w, h, d), temp_placed, container))
                candidates = candidates[:300]
            
            for rot in _rotations(w, h, d):
                for pos in candidates:
                    if not _fits(rot, pos, container):
                        continue
                    if not _no_overlap(rot, pos, temp_placed):
                        continue
                    sc = _score_position_compact(pos, rot, temp_placed, container)
                    if sc < best_score - 1e-6:
                        best_score = sc
                        best_pos = pos
                        best_rot = rot
            
            if best_pos is not None:
                # Move the box
                placements[i] = (0, best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2])
                placed[placed_idx] = (best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2])
                improved = True
        
        if not improved:
            break
    
    return placements

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    W, H, D = container
    container_volume = W * H * D
    
    best_result = None
    best_volume = -1
    
    def update_best(result):
        nonlocal best_result, best_volume
        vol = sum(w*h*d for (bin_, x, y, z, w, h, d) in result if bin_ == 0)
        if vol > best_volume:
            best_volume = vol
            best_result = result
    
    # Strategy 1: Greedy with compact score
    result1 = _greedy_load(boxes, container, deadline, seed=42, score_func=_score_position_compact)
    update_best(result1)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 2: Greedy with low score
    result2 = _greedy_load(boxes, container, deadline, seed=123, score_func=_score_position_low)
    update_best(result2)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 3: Greedy with balanced score
    result3 = _greedy_load(boxes, container, deadline, seed=456, score_func=_score_position_balanced)
    update_best(result3)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 4: Shelf greedy
    result4 = _shelf_greedy(boxes, container, deadline, seed=789)
    update_best(result4)
    
    if time.time() > deadline:
        return best_result
    
    # Local search on best result
    if best_result is not None:
        improved = _local_search(boxes, container, best_result, deadline, max_iterations=30)
        update_best(improved)
    
    return best_result
