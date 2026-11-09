"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def _get_all_orientations(w, h, d):
    """Return unique orientations of a box (w,h,d)."""
    seen = set()
    orientations = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            orientations.append(perm)
    return orientations

def _fits(placed, x, y, z, w, h, d):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box."""
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _candidate_points(placed, W, H, D):
    """Generate candidate (x,y,z) positions to try for placement."""
    candidates = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placed:
        # Points adjacent to each face of each placed box
        candidates.append((px + pw, py, pz))
        candidates.append((px, py + ph, pz))
        candidates.append((px, py, pz + pd))
    # Also try origin-aligned points
    return candidates

def _blb_solve(boxes, container, time_limit=10.0):
    """
    Bottom-Left-Back heuristic with multiple sorting strategies and restarts.
    Returns list of placements: (bin, x, y, z, w, h, d) per box.
    """
    W, H, D = container
    container_vol = W * H * D
    
    # Precompute all orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        box_orientations.append(_get_all_orientations(w, h, d))
    
    # Sorting strategies
    def sort_by_volume_desc(idx_list):
        return sorted(idx_list, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: max(boxes[i]), reverse=True)
    
    def sort_by_footprint_desc(idx_list):
        # Max footprint area among orientations
        return sorted(idx_list, key=lambda i: max(w*h for w,h,d in box_orientations[i]), reverse=True)
    
    def sort_by_height_desc(idx_list):
        return sorted(idx_list, key=lambda i: max(d for w,h,d in box_orientations[i]), reverse=True)
    
    def sort_by_volume_asc(idx_list):
        return sorted(idx_list, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    
    strategies = [
        sort_by_volume_desc,
        sort_by_max_dim_desc,
        sort_by_footprint_desc,
        sort_by_height_desc,
        sort_by_volume_asc,
    ]
    
    all_indices = list(range(len(boxes)))
    best_placements = None
    best_score = 0.0
    
    start_time = time.time()
    
    for strategy in strategies:
        if time.time() - start_time > time_limit:
            break
        
        order = strategy(all_indices)
        placed = []  # List of (x, y, z, w, h, d)
        box_placed = [False] * len(boxes)
        box_pos = [None] * len(boxes)
        
        for idx in order:
            w, h, d = boxes[idx]
            orientations = box_orientations[idx]
            
            # Try all orientations, find best fit position
            best_pos = None
            best_score_pos = None
            
            # Generate candidate points
            candidates = _candidate_points(placed, W, H, D)
            # Add origin
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
            
            # For each orientation, try each candidate point
            found = False
            for ow, oh, od in orientations:
                for (cx, cy, cz) in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if _fits(placed, cx, cy, cz, ow, oh, od):
                        # Score: prefer lower z, then lower y, then lower x
                        pos_score = (cz, cy, cx)
                        if best_pos is None or pos_score < best_score_pos:
                            best_pos = (cx, cy, cz, ow, oh, od)
                            best_score_pos = pos_score
                        found = True
                        break
                if found:
                    break
            
            if best_pos is not None:
                placed.append(best_pos)
                box_placed[idx] = True
                box_pos[idx] = best_pos
        
        # Calculate score
        total_vol = sum(p[3]*p[4]*p[5] for p in placed)
        score = total_vol / container_vol
        
        if score > best_score:
            best_score = score
            # Build result
            result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
            for i in range(len(boxes)):
                if box_placed[i]:
                    x, y, z, w, h, d = box_pos[i]
                    result[i] = (0, x, y, z, w, h, d)
            best_placements = result
    
    # If no strategy worked, return all -1
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
    
    return best_placements

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Check if any box fits at all
    any_fits = False
    for w, h, d in boxes:
        for ow, oh, od in _get_all_orientations(w, h, d):
            if ow <= W and oh <= H and od <= D:
                any_fits = True
                break
        if any_fits:
            break
    
    if not any_fits:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Time budget: leave safety margin
    start = time.time()
    deadline = 110.0  # 110 seconds out of 120s limit
    
    # Run BLB solver
    result = _blb_solve(boxes, container, time_limit=deadline - (time.time() - start) - 5.0)
    
    elapsed = time.time() - start
    if elapsed > deadline:
        # Emergency: return simple shelf packing
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    return result
# EVOLVE-BLOCK-END
