"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _place_circle(cx, cy, radius, placed, W, H):
    """
    Try to place a circle at (cx, cy) with given radius.
    Returns True if valid (no overlap with placed circles and inside rectangle).
    """
    # Check bounds
    if cx - radius < -1e-9 or cx + radius > W + 1e-9:
        return False
    if cy - radius < -1e-9 or cy + radius > H + 1e-9:
        return False
    # Check overlap with placed circles
    for (px, py, pr) in placed:
        dist = math.hypot(cx - px, cy - py)
        if dist < radius + pr - 1e-9:
            return False
    return True

def _max_radius_at(cx, cy, placed, W, H):
    """
    Given a center (cx, cy), compute the maximum radius such that the circle
    is inside the rectangle and doesn't overlap any placed circle.
    """
    # Max radius from walls
    r_wall = min(cx, W - cx, cy, H - cy)
    if r_wall <= 1e-12:
        return 0.0
    # Max radius from other circles
    r_max = r_wall
    for (px, py, pr) in placed:
        dist = math.hypot(cx - px, cy - py)
        # To not overlap: r + pr <= dist  =>  r <= dist - pr
        r_candidate = dist - pr
        if r_candidate < r_max:
            r_max = r_candidate
    if r_max <= 1e-12:
        return 0.0
    return r_max

def _greedy_pack(W, H, n_circles, seed=0):
    """
    Greedy sequential placement: place n_circles circles in rectangle W x H.
    Uses a grid of candidate centers to find the position that maximizes the
    achievable radius for each new circle.
    Returns list of (cx, cy, r).
    """
    placed = []
    # Candidate grid resolution
    # Use a reasonably fine grid for candidate centers
    nx = 40
    ny = 40
    xs = np.linspace(0, W, nx)
    ys = np.linspace(0, H, ny)
    
    for _ in range(n_circles):
        best_r = -1
        best_x = 0
        best_y = 0
        for x in xs:
            for y in ys:
                r = _max_radius_at(x, y, placed, W, H)
                if r > best_r:
                    best_r = r
                    best_x = x
                    best_y = y
        if best_r <= 1e-9:
            break
        placed.append((best_x, best_y, best_r))
    
    return placed

def _expand_radii(placed, W, H, iterations=50):
    """
    Coordinate descent: for each circle, try to expand its radius by moving
    it slightly or just increasing radius if possible.
    Here we just try to increase radius in place (center fixed) and if that fails,
    we try small center shifts.
    """
    # Simple version: just try to increase each radius slightly
    # This is a placeholder for a more sophisticated expansion.
    # For now, we just return as is, as the greedy placement is already quite good.
    # A better expansion would involve local search on each circle's position and radius.
    pass
    return placed

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = -1
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 2(W+H) = 4 => W+H = 2
    # Let W = w, H = 2 - w
    # w in (0, 2)
    # Symmetry: w and 2-w are equivalent (just swap x/y)
    # So try w in [0.5, 1.5]
    
    # We'll try a set of aspect ratios
    aspect_ratios = np.linspace(0.6, 1.4, 20)  # W from 0.6 to 1.4, H from 1.4 to 0.6
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        
        # Greedy pack
        placed = _greedy_pack(w, h, N)
        
        if len(placed) < N:
            continue
        
        # Calculate sum of radii
        score = sum(r for (_, _, r) in placed)
        
        if score > best_score:
            best_score = score
            best_result = np.array(placed)
    
    # If no valid result, fall back to a simple grid
    if best_result is None:
        # Fallback: 5x5 grid in a 1x1 square (perimeter 4)
        # But we need 21 circles, so use 5x5=25, take 21
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
