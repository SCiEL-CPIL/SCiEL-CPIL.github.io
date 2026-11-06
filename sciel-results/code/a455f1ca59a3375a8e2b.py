import time
import random
import numpy as np
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    # Precompute triple indices for n=30
    triple_indices = np.array([(i, j, k) for i in range(30) for j in range(i+1, 30) for k in range(j+1, 30)], dtype=np.int64)
    I, J, K = triple_indices[:, 0], triple_indices[:, 1], triple_indices[:, 2]
    N_TRIPLES = len(I)

    def evaluate_fast(points):
        """
        Vectorized evaluation of max multiplicity of positive doubled areas.
        points: list of 30 (x, y) tuples.
        """
        pts = np.array(points, dtype=np.int64)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Compute doubled area (determinant) for all triples
        det = (x[J] - x[I]) * (y[K] - y[I]) - (y[J] - y[I]) * (x[K] - x[I])
        
        # Take absolute value and filter positive
        det = np.abs(det)
        det = det[det > 0]
        
        if len(det) == 0:
            return 0
        
        vals, cnts = np.unique(det, return_counts=True)
        return int(cnts.max())

    def valid(points):
        if len(points) != 30:
            return False
        if len(set(points)) != 30:
            return False
        for px, py in points:
            if not (-30 <= px <= 30 and -30 <= py <= 30):
                return False
        return True

    # Number-theoretic construction: generalized arithmetic progression
    # Points as linear combinations of two basis vectors
    # p = a * u + b * v, where u, v are primitive lattice vectors
    # The determinant of three points (a1u+b1v, a2u+b2v, a3u+b3v) is:
    # (a2-a1)(b3-b1) - (a3-a1)(b2-b1)) * det(u,v)
    # So if we fix det(u,v) = d, all areas are multiples of d.
    # To maximize multiplicity of a single area, we want many triples with the same (a2-a1)(b3-b1) - (a3-a1)(b2-b1) value.

    best_points = None
    best_score = 0

    # Strategy 1: Two parallel lines (known good baseline)
    base_pts = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    if valid(base_pts):
        s = evaluate_fast(base_pts)
        if s > best_score:
            best_score = s
            best_points = base_pts[:]

    # Strategy 2: Grid configurations
    grid_configs = [(5, 6), (6, 5), (10, 3), (3, 10), (15, 2), (2, 15)]
    for w, h in grid_configs:
        if w * h != 30:
            continue
        pts = [(x, y) for x in range(w) for y in range(h)]
        if valid(pts):
            s = evaluate_fast(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if time.time() > deadline:
            return best_points

    # Strategy 3: Number-theoretic construction with varying basis vectors
    # Try different basis vectors u=(ux,uy), v=(vx,vy)
    # Points: p_i = a_i * u + b_i * v for (a_i, b_i) in a 2D grid
    # This creates a parallelogram lattice structure
    
    # For a w x h grid of (a,b) pairs, points are (a*ux+b*vx, a*uy+b*vy)
    # The area multiplier is |ux*vy - uy*vx|
    
    for w, h in grid_configs:
        if w * h != 30:
            continue
        # Try various basis vectors
        for ux in range(1, 8):
            for uy in range(-7, 8):
                for vx in range(-7, 8):
                    for vy in range(1, 8):
                        det_uv = abs(ux * vy - uy * vx)
                        if det_uv == 0:
                            continue
                        # Check that all points are in bounds
                        pts = []
                        valid_pts = True
                        for a in range(w):
                            for b in range(h):
                                px = a * ux + b * vx
                                py = a * uy + b * vy
                                if not (-30 <= px <= 30 and -30 <= py <= 30):
                                    valid_pts = False
                                    break
                                pts.append((px, py))
                            if not valid_pts:
                                break
                        if not valid_pts:
                            continue
                        if len(set(pts)) != 30:
                            continue
                        s = evaluate_fast(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts[:]
                        if time.time() > deadline:
                            return best_points

    # Strategy 4: Three parallel lines with optimized spacing
    for d1 in range(1, 6):
        for d2 in range(d1 + 1, 10):
            for a in range(5, 26):
                for b in range(5, 26):
                    c = 30 - a - b
                    if c < 1:
                        continue
                    pts = [(x, 0) for x in range(a)] + [(x, d1) for x in range(b)] + [(x, d2) for x in range(c)]
                    if valid(pts):
                        s = evaluate_fast(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts[:]
                    if time.time() > deadline:
                        return best_points

    # Strategy 5: Local search with larger neighborhood
    if best_points is None:
        best_points = base_pts[:]
        best_score = evaluate_fast(best_points)

    random.seed(42)
    
    # Pass 1: Small perturbations
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(30):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    s = evaluate_fast(new_points)
                    if s > best_score:
                        best_score = s
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Pass 2: Larger perturbations
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(30):
                if time.time() > deadline:
                    break
                old_x, old_y = best_points[idx]
                for dx in range(-6, 7):
                    for dy in range(-6, 7):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                            continue
                        new_points = list(best_points)
                        new_points[idx] = (new_x, new_y)
                        if len(set(new_points)) != 30:
                            continue
                        s = evaluate_fast(new_points)
                        if s > best_score:
                            best_score = s
                            best_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # Pass 3: Random restarts with local search
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            current_points = [(random.randint(-15, 15), random.randint(-15, 15)) for _ in range(30)]
            while len(set(current_points)) < 30:
                idx = random.randint(0, 29)
                current_points[idx] = (random.randint(-15, 15), random.randint(-15, 15))
            
            current_score = evaluate_fast(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                idx = random.randint(0, 29)
                old_x, old_y = current_points[idx]
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                            continue
                        new_points = list(current_points)
                        new_points[idx] = (new_x, new_y)
                        if len(set(new_points)) != 30:
                            continue
                        s = evaluate_fast(new_points)
                        if s > current_score:
                            current_score = s
                            current_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points[:]

    return best_points
