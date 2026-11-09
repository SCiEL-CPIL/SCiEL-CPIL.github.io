"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import permutations

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    # Initial placement list
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n

    # Precompute unique orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        orients = []
        seen = set()
        for perm in permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        box_orientations.append(orients)

    def try_all_orientations(boxes_list):
        """For a single box, generate all unique orientations."""
        w, h, d = boxes_list
        orientations = []
        seen = set()
        for perm in permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                orientations.append(perm)
        return orientations

    def get_candidate_positions(placed, ow, oh, od, W, H, D):
        """Generate candidate positions for a box with dimensions (ow, oh, od)."""
        positions = set()
        positions.add((0, 0, 0))
        
        for px, py, pz, pw, ph, pd in placed:
            # Right of box
            if px + pw + ow <= W:
                positions.add((px + pw, py, pz))
                positions.add((px + pw, py + ph, pz))
                positions.add((px + pw, py, pz + pd))
                positions.add((px + pw, py + ph, pz + pd))
            # Above box (in y)
            if py + ph + oh <= H:
                positions.add((px, py + ph, pz))
                positions.add((px + pw, py + ph, pz))
                positions.add((px, py + ph, pz + pd))
                positions.add((px + pw, py + ph, pz + pd))
            # In front of box (in z)
            if pz + pd + od <= D:
                positions.add((px, py, pz + pd))
                positions.add((px + pw, py, pz + pd))
                positions.add((px, py + ph, pz + pd))
                positions.add((px + pw, py + ph, pz + pd))
        
        return list(positions)

    def greedy_fill(boxes_sorted, container):
        """Greedy first-fit decreasing with multiple orientation tries."""
        W, H, D = container
        placed = []  # list of (x, y, z, w, h, d)
        result = {}  # box_idx -> (x, y, z, w, h, d)
        
        for idx in boxes_sorted:
            w, h, d = boxes[idx]
            if time.time() - start_time > DEADLINE:
                break
            
            orientations = box_orientations[idx]
            best_orient = None
            best_pos = None
            best_score = float('inf')
            
            # Try each orientation
            for ow, oh, od in orientations:
                # Check if fits in container
                if ow > W or oh > H or od > D:
                    continue
                
                positions_to_try = get_candidate_positions(placed, ow, oh, od, W, H, D)
                
                # Try each position
                for (tx, ty, tz) in positions_to_try:
                    if tx + ow > W or ty + oh > H or tz + od > D:
                        continue
                    # Fast overlap check using placed list
                    ok = True
                    for px, py, pz, pw, ph, pd in placed:
                        if not (tx + ow <= px or px + pw <= tx or
                                ty + oh <= py or py + ph <= ty or
                                tz + od <= pz or pz + pd <= tz):
                            ok = False
                            break
                    if ok:
                        # Score: prefer lower positions (bottom-left-front)
                        score = tx * 1000 + ty * 100 + tz
                        if score < best_score:
                            best_score = score
                            best_orient = (ow, oh, od)
                            best_pos = (tx, ty, tz)
            
            if best_pos is not None:
                placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
                result[idx] = (best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2])
        
        return result, placed

    def compute_fill(result, container):
        """Compute total volume of placed boxes."""
        total = 0
        for idx, (x, y, z, w, h, d) in result.items():
            total += w * h * d
        return total / container_vol

    # Strategy 1: Sort by volume descending
    def sort_by_volume(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Strategy 2: Sort by max dimension descending
    def sort_by_max_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
    
    # Strategy 3: Sort by min dimension descending
    def sort_by_min_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)
    
    # Strategy 4: Sort by base area (w*d) descending
    def sort_by_base_area(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][2], reverse=True)
    
    # Strategy 5: Sort by volume/shortest_dim (prefer cubical)
    def sort_by_cubicality(boxes):
        return sorted(range(len(boxes)), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2]) / min(boxes[i]) if min(boxes[i]) > 0 else 0, reverse=True)
    
    # Strategy 6: Sort by longest edge / volume ratio (prefer elongated)
    def sort_by_elongation(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]) / (boxes[i][0]*boxes[i][1]*boxes[i][2]) ** 0.333 if boxes[i][0]*boxes[i][1]*boxes[i][2] > 0 else 0, reverse=True)

    # Random ordering strategy
    def sort_random(boxes, seed=None):
        idxs = list(range(len(boxes)))
        if seed is not None:
            random.seed(seed)
        random.shuffle(idxs)
        return idxs

    best_result = {}
    best_fill = 0.0

    strategies = [
        sort_by_volume,
        sort_by_max_dim,
        sort_by_min_dim,
        sort_by_base_area,
        sort_by_cubicality,
        sort_by_elongation,
    ]

    # Phase 1: Deterministic strategies
    for strat in strategies:
        if time.time() - start_time > DEADLINE:
            break
        
        boxes_sorted = strat(boxes)
        result, placed = greedy_fill(boxes_sorted, container)
        
        fill = compute_fill(result, container)
        if fill > best_fill:
            best_fill = fill
            best_result = dict(result)
    
    # Phase 2: Random restarts with remaining time budget
    if time.time() - start_time < DEADLINE - 5:
        remaining_time = DEADLINE - (time.time() - start_time)
        # Estimate time per restart: use elapsed so far / number of attempts
        elapsed = time.time() - start_time
        num_attempts = len(strategies)
        if num_attempts > 0:
            avg_time_per_restart = elapsed / num_attempts
        else:
            avg_time_per_restart = 1.0
        
        # Allow at least 1 restart, but don't exceed remaining budget
        max_restarts = max(1, int(remaining_time / max(avg_time_per_restart, 0.1)))
        max_restarts = min(max_restarts, 50)  # cap to avoid too many
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Mix deterministic order with some randomness
            # Shuffle a random subset of the volume-sorted list
            base_order = sort_by_volume(boxes)
            boxes_sorted = list(base_order)
            
            # Randomly swap some adjacent pairs to create variation
            for _ in range(min(5, n // 2)):
                i = random.randint(0, n - 2)
                boxes_sorted[i], boxes_sorted[i+1] = boxes_sorted[i+1], boxes_sorted[i]
            
            result, placed = greedy_fill(boxes_sorted, container)
            
            fill = compute_fill(result, container)
            if fill > best_fill:
                best_fill = fill
                best_result = dict(result)
    
    # Phase 3: Second pass - try to fit remaining boxes
    if best_result and time.time() - start_time < DEADLINE - 10:
        placed = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in best_result.values()]
        
        # Get unplaced boxes, sorted by volume descending
        unplaced = [i for i in range(n) if i not in best_result]
        unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        # Try to place unplaced boxes one by one
        for idx in unplaced:
            if time.time() - start_time > DEADLINE:
                break
            w, h, d = boxes[idx]
            orientations = box_orientations[idx]
            
            best_orient = None
            best_pos = None
            best_score = float('inf')
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                positions_to_try = get_candidate_positions(placed, ow, oh, od, W, H, D)
                
                for (tx, ty, tz) in positions_to_try:
                    if tx + ow > W or ty + oh > H or tz + od > D:
                        continue
                    ok = True
                    for px, py, pz, pw, ph, pd in placed:
                        if not (tx + ow <= px or px + pw <= tx or
                                ty + oh <= py or py + ph <= ty or
                                tz + od <= pz or pz + pd <= tz):
                            ok = False
                            break
                    if ok:
                        score = tx * 1000 + ty * 100 + tz
                        if score < best_score:
                            best_score = score
                            best_orient = (ow, oh, od)
                            best_pos = (tx, ty, tz)
            
            if best_pos is not None:
                best_result[idx] = (best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2])
                placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
    
    # Build final placements list
    for i in range(n):
        if i in best_result:
            x, y, z, w, h, d = best_result[i]
            placements[i] = (0, x, y, z, w, h, d)
        else:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
