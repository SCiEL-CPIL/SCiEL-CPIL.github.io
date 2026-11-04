"""Optimized program for Heilbronn convex regions n=14 using simulated annealing."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import ConvexHull


def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # Fallback: compute hull area manually
        n = len(pts)
        if n < 3:
            return 0.0
        # Get hull vertices
        hull_idx = ConvexHull(pts).vertices
        hull_pts = pts[hull_idx]
        m = len(hull_pts)
        area = 0.0
        for i in range(m):
            j = (i + 1) % m
            area += hull_pts[i][0] * hull_pts[j][1] - hull_pts[j][0] * hull_pts[i][1]
        return abs(area) * 0.5


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _random_in_circle(rng, radius=1.0):
    """Generate random point in unit circle."""
    r = radius * math.sqrt(rng.random())
    theta = 2 * math.pi * rng.random()
    return [r * math.cos(theta), r * math.sin(theta)]


def _initial_points(rng):
    """Generate initial points: some on circle, some inside."""
    pts = []
    # Place 8 points roughly evenly on circle, 6 inside
    for i in range(8):
        theta = 2 * math.pi * i / 8 + rng.random() * 0.1
        pts.append([math.cos(theta), math.sin(theta)])
    for i in range(6):
        pts.append(_random_in_circle(rng, 0.7))
    return np.array(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    rng2 = np.random.RandomState(42)
    
    # Try multiple initializations with simulated annealing
    best_pts = None
    best_score = 0.0
    
    n_starts = 50
    for start_idx in range(n_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Different initializations
        if start_idx == 0:
            pts = _initial_points(rng2)
        elif start_idx == 1:
            # Regular polygon + center
            pts = []
            for i in range(12):
                theta = 2 * math.pi * i / 12
                pts.append([0.9 * math.cos(theta), 0.9 * math.sin(theta)])
            pts.append([0.0, 0.0])
            pts.append([0.3, 0.0])
            pts = np.array(pts)
        elif start_idx == 2:
            # Two concentric rings
            pts = []
            for i in range(7):
                theta = 2 * math.pi * i / 7
                pts.append([0.8 * math.cos(theta), 0.8 * math.sin(theta)])
            for i in range(7):
                theta = 2 * math.pi * i / 7 + math.pi / 7
                pts.append([0.4 * math.cos(theta), 0.4 * math.sin(theta)])
            pts = np.array(pts)
        else:
            pts = np.array([_random_in_circle(rng2, 1.0) for _ in range(N)])
        
        # Simulated annealing
        current_pts = pts.copy()
        current_score = _score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
        
        T = 0.05
        T_min = 0.0001
        cooling = 0.995
        max_steps = 20000
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Generate perturbation
            idx = rng2.randint(0, N)
            step_size = T * 0.5
            new_pts = current_pts.copy()
            new_pts[idx] += rng2.normal(0, step_size, 2)
            
            # Keep points in reasonable bounds
            new_pts[idx] = np.clip(new_pts[idx], -1.5, 1.5)
            
            new_score = _score(new_pts)
            
            # Accept if better or with probability exp((new-old)/T)
            if new_score >= current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                delta = new_score - current_score
                if delta > 0 or rng2.random() < math.exp(delta / T):
                    current_pts = new_pts
                    current_score = new_score
            
            T = max(T * cooling, T_min)
        
        # Local refinement with smaller steps
        if time.time() > deadline - 3.0:
            break
        
        T = 0.001
        for step in range(5000):
            if time.time() > deadline - 3.0:
                break
            idx = rng2.randint(0, N)
            step_size = T * 0.3
            new_pts = current_pts.copy()
            new_pts[idx] += rng2.normal(0, step_size, 2)
            new_pts[idx] = np.clip(new_pts[idx], -1.5, 1.5)
            
            new_score = _score(new_pts)
            if new_score >= current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                delta = new_score - current_score
                if delta > 0 or rng2.random() < math.exp(delta / T):
                    current_pts = new_pts
                    current_score = new_score
            T *= 0.999
    
    if best_pts is None:
        # Fallback: regular polygon
        pts = []
        for i in range(N):
            theta = 2 * math.pi * i / N
            pts.append([math.cos(theta), math.sin(theta)])
        best_pts = np.array(pts)
    
    return best_pts
# EVOLVE-BLOCK-END
