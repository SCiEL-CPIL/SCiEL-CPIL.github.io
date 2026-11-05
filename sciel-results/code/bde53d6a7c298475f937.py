import time
import random
import itertools
import numpy as np
from collections import Counter, defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    def calc_multiplicity(pts):
        """Calculate max multiplicity of any positive doubled area."""
        n = len(pts)
        areas = Counter()
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] += 1
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        max_area = [a for a, m in areas.items() if m == max_mult][0]
        return max_mult, max_area
    
    def generate_lattice_points(a, b, c, d, rows, cols):
        """Generate points from lattice basis vectors (a,b) and (c,d)."""
        pts = set()
        for i in range(rows):
            for j in range(cols):
                x = i * a + j * c
                y = i * b + j * d
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.add((x, y))
        if len(pts) >= 30:
            # Take first 30 distinct points
            return list(pts)[:30]
        return None
    
    def generate_sheared_grid(w, h, shear):
        """Generate sheared grid points."""
        pts = set()
        for i in range(w):
            for j in range(h):
                x = i + shear * j
                y = j
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.add((x, y))
        if len(pts) >= 30:
            return list(pts)[:30]
        return None
    
    candidates = []
    
    # Generate lattice-based candidates
    # Try various lattice bases
    lattice_bases = []
    for a in range(1, 8):
        for b in range(0, 8):
            for c in range(1, 8):
                for d in range(0, 8):
                    # Ensure basis vectors are linearly independent
                    det = a * d - b * c
                    if det != 0:
                        lattice_bases.append((a, b, c, d))
    
    # Try various grid dimensions
    for rows, cols in [(5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2), (1, 30), (30, 1), (4, 8), (8, 4), (3, 10)]:
        for (a, b, c, d) in lattice_bases[:50]:  # Limit to keep time reasonable
            pts = generate_lattice_points(a, b, c, d, rows, cols)
            if pts and len(set(pts)) == 30:
                candidates.append(pts)
    
    # Generate sheared grid candidates
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3), (4, 8), (8, 4)]:
        for shear in range(-5, 6):
            pts = generate_sheared_grid(w, h, shear)
            if pts:
                candidates.append(pts)
    
    # Random candidates
    for _ in range(100):
        if time.time() > deadline - 15:
            break
        pts = set()
        attempts = 0
        while len(pts) < 30 and attempts < 1000:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
            attempts += 1
        if len(pts) == 30:
            candidates.append(list(pts))
    
    # Evaluate all candidates
    best_score = 0
    best_points = candidates[0] if candidates else [(0, 0)] * 30
    best_area = 1
    
    for pts in candidates:
        score, area = calc_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            best_area = area
    
    if not candidates:
        return [(0, 0)] * 30
    
    # Local search: try moving individual points to improve multiplicity
    current_points = list(best_points)
    current_score = best_score
    current_area = best_area
    
    # Helper: calculate change in count of target_area when moving point idx
    def delta_for_move(pts, idx, new_pos, target_area):
        old_pos = pts[idx]
        delta = 0
        n = len(pts)
        ox, oy = old_pos
        nx, ny = new_pos
        
        for j in range(n):
            if j == idx:
                continue
            x2, y2 = pts[j]
            for k in range(j + 1, n):
                if k == idx:
                    continue
                x3, y3 = pts[k]
                
                # New area
                det_new = abs((x2 - nx) * (y3 - ny) - (y2 - ny) * (x3 - nx))
                if det_new == target_area:
                    delta += 1
                
                # Old area
                det_old = abs((x2 - ox) * (y3 - oy) - (y2 - oy) * (x3 - ox))
                if det_old == target_area:
                    delta -= 1
        return delta
    
    # Hill climbing with larger neighborhood
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5:
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        best_move = None
        best_delta = 0
        
        # Try larger neighborhood: radius 5
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                new_pos = (new_x, new_y)
                if new_pos in current_points:
                    continue
                
                delta = delta_for_move(current_points, idx, new_pos, current_area)
                if delta > best_delta:
                    best_delta = delta
                    best_move = new_pos
        
        if best_move:
            current_points[idx] = best_move
            no_improve = 0
        else:
            no_improve += 1
            
            if no_improve > max_no_improve:
                # Re-evaluate and potentially reset target area
                score, area = calc_multiplicity(current_points)
                if score > current_score:
                    current_score = score
                    current_area = area
                    no_improve = 0
                else:
                    # Reset to best known
                    current_points = list(best_points)
                    current_score = best_score
                    current_area = best_area
                    no_improve = 0
            
            # Occasionally try a big jump
            if no_improve % 50 == 0:
                idx = random.randint(0, 29)
                for _ in range(10):
                    new_x = random.randint(-30, 30)
                    new_y = random.randint(-30, 30)
                    new_pos = (new_x, new_y)
                    if new_pos in current_points:
                        continue
                    delta = delta_for_move(current_points, idx, new_pos, current_area)
                    if delta > 0:
                        current_points[idx] = new_pos
                        no_improve = 0
                        break
    
    # Final evaluation
    final_score, final_area = calc_multiplicity(current_points)
    if final_score > best_score:
        best_score = final_score
        best_points = current_points
    
    return best_points
