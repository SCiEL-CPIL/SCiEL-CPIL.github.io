"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

# EVOLVE-BLOCK-START

def _compute_scores(points):
    """Compute the score for a set of points.
    
    Score = largest multiplicity of a single positive doubled area.
    Doubled area of triangle (p1,p2,p3) = |det(p2-p1, p3-p1)|
    """
    n = len(points)
    if n < 3:
        return 0
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _compute_scores_fast(points):
    """Faster computation using numpy."""
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0]))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _greedy_construct():
    """Construct points greedily to maximize equal-area triangles."""
    # Strategy: use a grid-based approach where many triangles share the same area
    # Key insight: if we place points on a line, all triangles have zero area.
    # If we use a regular grid, we get many equal areas.
    
    # Try different grid configurations
    best_score = 0
    best_points = None
    
    # Standard grid: 6x5
    for w in range(3, 8):
        for h in range(3, 8):
            if w * h >= 30:
                # Create a w x h grid, take first 30 points
                points = []
                for x in range(w):
                    for y in range(h):
                        points.append((x, y))
                        if len(points) == 30:
                            break
                    if len(points) == 30:
                        break
                if len(points) == 30:
                    score = _compute_scores_fast(points)
                    if score > best_score:
                        best_score = score
                        best_points = points[:]
    
    return best_points, best_score

def _random_search(points, n_iters, deadline):
    """Random local search to improve the configuration."""
    best_score = _compute_scores_fast(points)
    best_points = points[:]
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Pick a random point and move it to a random nearby location
        idx = random.randint(0, 29)
        x, y = best_points[idx]
        new_x = x + random.randint(-3, 3)
        new_y = y + random.randint(-3, 3)
        
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
        
        new_points = best_points[:]
        new_points[idx] = (new_x, new_y)
        
        # Check distinctness
        if len(set(new_points)) != 30:
            continue
        
        score = _compute_scores_fast(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points[:]
    
    return best_points, best_score

def _structured_search(deadline):
    """Try structured patterns that are known to produce many equal-area triangles."""
    best_score = 0
    best_points = None
    
    # Pattern 1: Points on two parallel lines
    # If we have m points on one line and n on another, triangles with 2 on one line and 1 on the other
    # have area proportional to the distance between lines times the base length
    
    # Pattern 2: Points forming a regular polygon-like structure
    # Try concentric structures
    
    # Let's try: place points such that many triangles have the same base and height
    # For example, all points on a horizontal line at y=0, plus some at y=1
    
    # Configuration: 15 points at y=0, 15 points at y=1
    # Triangles with 2 points at y=0 and 1 at y=1: area = base/2 where base is distance between the 2 points
    # Triangles with 1 point at y=0 and 2 at y=1: area = base/2 where base is distance between the 2 points
    
    for split in range(10, 21):
        n1 = split
        n2 = 30 - split
        if n1 < 2 or n2 < 2:
            continue
        
        # Place n1 points on y=0, n2 points on y=1
        # To maximize equal areas, use evenly spaced points
        # Points on y=0: x = 0, 1, 2, ..., n1-1
        # Points on y=1: x = 0, 1, 2, ..., n2-1
        
        points = []
        for i in range(n1):
            points.append((i, 0))
        for i in range(n2):
            points.append((i, 1))
        
        if len(points) == 30 and len(set(points)) == 30:
            score = _compute_scores_fast(points)
            if score > best_score:
                best_score = score
                best_points = points[:]
    
    # Pattern 3: Points on a grid with specific structure
    # Try 5x6 grid
    for offset_x in range(-30, 31):
        for offset_y in range(-30, 31):
            points = []
            for x in range(5):
                for y in range(6):
                    points.append((x + offset_x, y + offset_y))
            if len(points) == 30 and all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in points):
                score = _compute_scores_fast(points)
                if score > best_score:
                    best_score = score
                    best_points = [(p[0]-offset_x, p[1]-offset_y) for p in points]
            if time.time() > deadline:
                return best_points, best_score
    
    return best_points, best_score

def _hill_climbing(points, deadline):
    """More systematic hill climbing."""
    best_score = _compute_scores_fast(points)
    best_points = points[:]
    
    while time.time() < deadline:
        improved = False
        for idx in range(30):
            if time.time() > deadline:
                break
            
            x, y = best_points[idx]
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x = x + dx
                    new_y = y + dy
                    if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                        continue
                    
                    new_points = best_points[:]
                    new_points[idx] = (new_x, new_y)
                    
                    if len(set(new_points)) != 30:
                        continue
                    
                    score = _compute_scores_fast(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points[:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try random restarts
            if random.random() < 0.3:
                # Perturb multiple points
                for _ in range(3):
                    idx = random.randint(0, 29)
                    x, y = best_points[idx]
                    new_x = x + random.randint(-5, 5)
                    new_y = y + random.randint(-5, 5)
                    if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                        new_points = best_points[:]
                        new_points[idx] = (new_x, new_y)
                        if len(set(new_points)) == 30:
                            score = _compute_scores_fast(new_points)
                            if score > best_score:
                                best_score = score
                                best_points = new_points[:]
    
    return best_points, best_score

def construct():
    """Main entry point. Returns exactly 30 distinct integer coordinate pairs in [-30,30]^2."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Start with a good initial configuration
    initial_points, initial_score = _greedy_construct()
    
    if initial_points is None:
        # Fallback: simple grid
        initial_points = [(x, y) for x in range(6) for y in range(5)]
        initial_score = _compute_scores_fast(initial_points)
    
    best_points = initial_points[:]
    best_score = initial_score
    
    # Try structured search
    if time.time() < deadline:
        struct_points, struct_score = _structured_search(deadline)
        if struct_points is not None and struct_score > best_score:
            best_points = struct_points
            best_score = struct_score
    
    # Random local search
    if time.time() < deadline:
        random.seed(42)
        search_points, search_score = _random_search(best_points, 2000, deadline)
        if search_score > best_score:
            best_points = search_points
            best_score = search_score
    
    # Hill climbing
    if time.time() < deadline:
        hc_points, hc_score = _hill_climbing(best_points, deadline)
        if hc_score > best_score:
            best_points = hc_points
            best_score = hc_score
    
    # Final check
    if time.time() > deadline - 5:
        # Return what we have
        pass
    
    return best_points

# EVOLVE-BLOCK-END
