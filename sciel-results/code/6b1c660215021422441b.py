"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _doubled_areas(points):
    """Compute absolute doubled areas for all triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if d > 0:
                    areas.append(d)
    return areas

def _score(points):
    """Compute the maximum multiplicity of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    from collections import Counter
    c = Counter(areas)
    return c.most_common(1)[0][1]

def _generate_grid_config():
    """Generate a grid-like configuration."""
    # 6x5 grid
    return [(x, y) for x in range(6) for y in range(5)]

def _generate_hex_config():
    """Generate a hexagonal-ish configuration."""
    points = []
    # Center
    points.append((0, 0))
    # Rings
    for r in range(1, 5):
        for angle_idx in range(6*r):
            angle = angle_idx * np.pi / (3*r)
            x = int(round(r * np.cos(angle)))
            y = int(round(r * np.sin(angle)))
            if (x, y) not in points and -30 <= x <= 30 and -30 <= y <= 30:
                points.append((x, y))
    # Fill to 30
    while len(points) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in points:
            points.append((x, y))
    return points[:30]

def _generate_arithmetic_config():
    """Generate points in arithmetic progressions to create equal-area triangles."""
    points = []
    # Use a 5x6 grid but with some structure
    for i in range(5):
        for j in range(6):
            points.append((2*i, 3*j))
    return points

def _local_search(points, max_steps=5000, deadline=None):
    """Perform local search by moving individual points."""
    current_points = list(points)
    current_score = _score(current_points)
    
    if deadline is not None and time.time() > deadline:
        return current_points, current_score
    
    best_points = list(current_points)
    best_score = current_score
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = random.randint(0, len(current_points)-1)
        old_x, old_y = current_points[idx]
        
        # Try random new positions
        found_improvement = False
        for _ in range(20):
            if deadline is not None and time.time() > deadline:
                break
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = (new_x, new_y)
            
            # Check if point already exists
            if new_point in current_points:
                continue
            
            current_points[idx] = new_point
            new_score = _score(current_points)
            
            if new_score > current_score:
                current_score = new_score
                found_improvement = True
                if new_score > best_score:
                    best_score = new_score
                    best_points = list(current_points)
                break
            else:
                current_points[idx] = (old_x, old_y)
        
        if not found_improvement:
            # Try small perturbations
            for _ in range(10):
                if deadline is not None and time.time() > deadline:
                    break
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
                if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                    new_point = (new_x, new_y)
                    if new_point in current_points:
                        continue
                    current_points[idx] = new_point
                    new_score = _score(current_points)
                    if new_score > current_score:
                        current_score = new_score
                        found_improvement = True
                        if new_score > best_score:
                            best_score = new_score
                            best_points = list(current_points)
                        break
                    else:
                        current_points[idx] = (old_x, old_y)
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Generate several initial configurations
    candidates = []
    
    # Grid
    candidates.append(_generate_grid_config())
    
    # Arithmetic progression
    candidates.append(_generate_arithmetic_config())
    
    # Hexagonal
    candidates.append(_generate_hex_config())
    
    # Random
    for _ in range(5):
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        candidates.append(pts)
    
    best_points = None
    best_score = 0
    
    for config in candidates:
        if time.time() > deadline:
            break
        
        # Evaluate initial score
        initial_score = _score(config)
        if initial_score > best_score:
            best_score = initial_score
            best_points = list(config)
        
        # Local search
        searched_points, searched_score = _local_search(config, max_steps=2000, deadline=deadline)
        if searched_score > best_score:
            best_score = searched_score
            best_points = searched_points
    
    if best_points is None:
        best_points = _generate_grid_config()
    
    return best_points
# EVOLVE-BLOCK-END

