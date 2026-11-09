"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import Counter

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                det = abs(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def _random_point():
    return (random.randint(-30, 30), random.randint(-30, 30))

def _distinct_random_points(n):
    pts = set()
    while len(pts) < n:
        pts.add(_random_point())
    return list(pts)

def _local_search(points, max_iter=5000, deadline=None):
    """Local search: try swapping one point with a random new point."""
    n = len(points)
    best = list(points)
    best_score = _score(best)
    current = list(best)
    current_score = best_score
    start = time.time()
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random index to replace
        idx = random.randint(0, n-1)
        old_pt = current[idx]
        new_pt = _random_point()
        if new_pt == old_pt:
            continue
        # Check distinctness
        if new_pt in current:
            continue
        
        candidate = current[:]
        candidate[idx] = new_pt
        s = _score(candidate)
        if s > current_score:
            current = candidate
            current_score = s
            if s > best_score:
                best = candidate
                best_score = s
        # Accept with some probability for exploration
        elif random.random() < 0.05:
            current = candidate
            current_score = s
    
    return best, best_score

def _hill_climb_with_target(points, target_area, max_iter=3000, deadline=None):
    """Hill climbing specifically targeting a given area value."""
    n = len(points)
    current = list(points)
    
    def target_score(pts):
        cnt = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    ax, ay = pts[i]
                    bx, by = pts[j]
                    cx, cy = pts[k]
                    det = abs(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
                    if det == target_area:
                        cnt += 1
        return cnt
    
    current_score = target_score(current)
    start = time.time()
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        idx = random.randint(0, n-1)
        old_pt = current[idx]
        new_pt = _random_point()
        if new_pt == old_pt or new_pt in current:
            continue
        
        candidate = current[:]
        candidate[idx] = new_pt
        s = target_score(candidate)
        if s > current_score:
            current = candidate
            current_score = s
    
    return current

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    n = 30
    
    best_points = None
    best_val = 0
    
    # Multiple random restarts with local search
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Generate initial random configuration
        pts = _distinct_random_points(n)
        
        # Local search
        found, val = _local_search(pts, max_iter=2000, deadline=deadline)
        if val > best_val:
            best_val = val
            best_points = found
        
        # If we have a good target area, do targeted hill climbing
        if best_val >= 5 and time.time() < deadline - 10:
            # Use the best area found so far as target
            areas = []
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        ax, ay = best_points[i]
                        bx, by = best_points[j]
                        cx, cy = best_points[k]
                        det = abs(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
                        if det > 0:
                            areas.append(det)
            if areas:
                target = Counter(areas).most_common(1)[0][0]
                improved = _hill_climb_with_target(best_points, target, max_iter=1000, deadline=deadline)
                iv = _score(improved)
                if iv > best_val:
                    best_val = iv
                    best_points = improved
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
