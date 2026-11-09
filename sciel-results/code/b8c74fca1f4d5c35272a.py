"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

# EVOLVE-BLOCK-START

def _doubled_area(p, q, r):
    """Return absolute doubled area of triangle p,q,r."""
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

def _score(points):
    """Return the maximum multiplicity of any single positive doubled area."""
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _random_point():
    return (random.randint(-30, 30), random.randint(-30, 30))

def _random_initial():
    pts = set()
    while len(pts) < 30:
        pts.add(_random_point())
    return list(pts)

def _grid_initial(ax, ay, bx, by, cx, cy, dx, dy):
    """Generate a 30-point configuration from a grid-like structure."""
    pts = set()
    # Try a rectangular grid
    for i in range(ax):
        for j in range(ay):
            if len(pts) >= 30:
                break
            x = cx + i * dx
            y = cy + j * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
        if len(pts) >= 30:
            break
    # Fill remaining with random points
    while len(pts) < 30:
        pts.add(_random_point())
    return list(pts)

def _local_search(points, deadline, best_points, best_score):
    """Improve points via local search, returning best found."""
    current = list(points)
    cur_score = _score(current)
    if cur_score > best_score:
        best_score = cur_score
        best_points = list(current)
    
    n = len(current)
    idx = list(range(n))
    
    while time.time() < deadline:
        # Randomly pick a point to move
        pi = random.choice(idx)
        old_pt = current[pi]
        
        # Try several candidate moves
        best_move = None
        best_move_score = cur_score
        
        # Local moves: small perturbations
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                    continue
                if new_pt in current:
                    continue
                
                # Evaluate
                candidate = current[:]
                candidate[pi] = new_pt
                s = _score(candidate)
                if s > best_move_score:
                    best_move_score = s
                    best_move = new_pt
        
        # Also try a random jump
        for _ in range(3):
            new_pt = _random_point()
            if new_pt in current:
                continue
            candidate = current[:]
            candidate[pi] = new_pt
            s = _score(candidate)
            if s > best_move_score:
                best_move_score = s
                best_move = new_pt
        
        if best_move is not None:
            current[pi] = best_move
            cur_score = best_move_score
            if cur_score > best_score:
                best_score = cur_score
                best_points = list(current)
        else:
            # No improvement found; try a random restart of one point
            if random.random() < 0.1:
                pi2 = random.choice(idx)
                new_pt = _random_point()
                if new_pt not in current:
                    current[pi2] = new_pt
                    cur_score = _score(current)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_points = list(current)
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_points = None
    best_score = 0
    
    # Initial diverse configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append([(x, y) for x in range(6) for y in range(5)])
    
    # 2. Different aspect ratio grids
    for (w, h) in [(5, 6), (10, 3), (15, 2)]:
        pts = []
        for x in range(w):
            for y in range(h):
                if len(pts) >= 30:
                    break
                pts.append((x - w//2, y - h//2))
            if len(pts) >= 30:
                break
        # Clamp to [-30,30]
        pts = [(max(-30, min(30, px)), max(-30, min(30, py))) for px, py in pts]
        if len(set(pts)) == 30:
            initial_configs.append(list(set(pts)))
    
    # 3. Random configurations
    for _ in range(5):
        initial_configs.append(_random_initial())
    
    # 4. Grid-like with offsets
    for offset_x in [-10, 0, 10]:
        for offset_y in [-10, 0, 10]:
            pts = [(x + offset_x, y + offset_y) for x in range(6) for y in range(5)]
            pts = [(max(-30, min(30, px)), max(-30, min(30, py))) for px, py in pts]
            if len(set(pts)) == 30:
                initial_configs.append(list(set(pts)))
    
    # Evaluate all initial configs and run local search on the best ones
    scored = []
    for cfg in initial_configs:
        if len(cfg) != 30:
            continue
        s = _score(cfg)
        scored.append((s, cfg))
    
    scored.sort(key=lambda x: -x[0])
    
    # Run local search on top candidates
    for s, cfg in scored[:10]:
        if time.time() > deadline:
            break
        if s > best_score:
            best_score = s
            best_points = list(cfg)
        
        best_points, best_score = _local_search(cfg, deadline, best_points, best_score)
    
    # If still time, do more random restarts
    while time.time() < deadline - 2.0:
        cfg = _random_initial()
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = list(cfg)
        
        if time.time() < deadline - 5.0:
            best_points, best_score = _local_search(cfg, deadline, best_points, best_score)
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points

# EVOLVE-BLOCK-END
