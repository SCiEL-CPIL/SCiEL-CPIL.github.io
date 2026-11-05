import time
import random
import itertools
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    # Strategy:
    # 1. Generate a diverse set of candidate configurations (grids, sheared, staggered, random)
    # 2. Evaluate each to find the best initial configuration
    # 3. Use local search with incremental area counting to improve
    
    candidates = []
    
    # Standard grids
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3), (2, 15), (15, 2), (1, 30), (30, 1)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) == 30 and len(set(pts)) == 30:
            cx, cy = w // 2, h // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            candidates.append(pts)
            
    # Sheared grids (parallelograms)
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3)]:
        for k in [1, -1, 2, -2]:
            pts = []
            valid = True
            for x in range(w):
                for y in range(h):
                    px = x + k * y
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid: break
            if valid and len(pts) == 30 and len(set(pts)) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)

    # Staggered grids (brick wall)
    for w, h in [(6, 5), (5, 6)]:
        for offset_pattern in [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0]]:
            if len(offset_pattern) != h: continue
            pts = []
            valid = True
            for y in range(h):
                off = offset_pattern[y]
                for x in range(w):
                    px = x + off
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid: break
            if valid and len(pts) == 30 and len(set(pts)) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)
    
    # Random configurations
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        pts = []
        attempts = 0
        while len(pts) < 30 and attempts < 500:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
            attempts += 1
        if len(pts) == 30:
            candidates.append(pts)

    # Evaluate all candidates
    best_score = 0
    best_points = candidates[0] if candidates else [(0,0)]
    best_area = 1
    
    for pts in candidates:
        score, area = _calculate_max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            best_area = area
            
    if not candidates:
        return [(0,0)] * 30  # Fallback, though shouldn't happen

    # Local search with incremental updates
    current_points = list(best_points)
    current_score = best_score
    current_area = best_area
    
    # Helper: Calculate delta in count of area A when moving point i from old_pos to new_pos
    def calc_delta_area(points, idx, old_pos, new_pos, target_area):
        delta = 0
        x1, y1 = new_pos
        ox, oy = old_pos
        n = len(points)
        
        for j in range(n):
            if j == idx: continue
            x2, y2 = points[j]
            for k in range(j + 1, n):
                if k == idx: continue
                x3, y3 = points[k]
                
                # Area of (new_pos, Pj, Pk)
                det_new = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det_new == target_area:
                    delta += 1
                
                # Area of (old_pos, Pj, Pk)
                det_old = abs((x2 - ox) * (y3 - oy) - (y2 - oy) * (x3 - ox))
                if det_old == target_area:
                    delta -= 1
        return delta

    # Main local search loop
    steps = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 3:
        steps += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Try moves in a 5x5 neighborhood around old position
        best_move = None
        best_delta = 0
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0: continue
                new_x = old_x + dx
                new_y = old_y + dy
                
                # Bounds check
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                # Distinctness check
                new_pos = (new_x, new_y)
                if new_pos in current_points:
                    continue
                
                # Calculate delta for current_target_area
                delta = calc_delta_area(current_points, idx, (old_x, old_y), new_pos, current_area)
                
                if delta > best_delta:
                    best_delta = delta
                    best_move = new_pos
        
        if best_move:
            # Apply move
            current_points[idx] = best_move
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # If stuck for a while, re-evaluate full configuration
            if no_improve_count > max_no_improve:
                score, area = _calculate_max_multiplicity(current_points)
                if score > current_score:
                    current_score = score
                    current_area = area
                    no_improve_count = 0
                else:
                    # Reset to best known
                    current_points = list(best_points)
                    current_score = best_score
                    current_area = best_area
                    no_improve_count = 0
            
            # Occasionally try a bigger jump
            if steps % 100 == 0:
                idx = random.randint(0, 29)
                old_x, old_y = current_points[idx]
                for _ in range(20):
                    new_x = random.randint(-30, 30)
                    new_y = random.randint(-30, 30)
                    new_pos = (new_x, new_y)
                    if new_pos in current_points:
                        continue
                    delta = calc_delta_area(current_points, idx, (old_x, old_y), new_pos, current_area)
                    if delta > 0:
                        current_points[idx] = new_pos
                        no_improve_count = 0
                        break

    # Final full evaluation
    final_score, final_area = _calculate_max_multiplicity(current_points)
    if final_score > best_score:
        best_score = final_score
        best_points = current_points

    return best_points

def _calculate_max_multiplicity(points):
    """Calculate the maximum multiplicity of any positive doubled area."""
    n = len(points)
    if n < 3:
        return 0, 0
        
    areas = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
                    
    if not areas:
        return 0, 0
        
    max_mult = max(areas.values())
    max_area = [a for a, m in areas.items() if m == max_mult][0]
    return max_mult, max_area
# EVOLVE-BLOCK-END
