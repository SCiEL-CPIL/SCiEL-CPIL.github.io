"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import math

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area_vec(pts):
    """Vectorized minimum triangle area computation."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_tri = _min_triangle_area_vec(pts)
    return min_tri / hull_a

def _jittered_grid(n, jitter=0.1, seed=42):
    """Generate a jittered grid of n points in [0,1]^2."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n)))
    while side * side < n:
        side += 1
    
    points = []
    for i in range(side):
        for j in range(side):
            if len(points) >= n:
                break
            x = (i + 0.5) / side + rng.uniform(-jitter, jitter) / side
            y = (j + 0.5) / side + rng.uniform(-jitter, jitter) / side
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            points.append([x, y])
    
    return np.array(points[:n])

def _simulated_annealing(pts_init, deadline, max_iter=50000, T0=0.05, T_min=1e-6, cooling=0.9995):
    """Simulated annealing to maximize min triangle area / hull area."""
    pts = pts_init.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    current_score = best_score
    T = T0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = np.random.randint(N)
        delta = np.random.normal(0, T * 2, size=2)
        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts[idx] = np.clip(new_pts[idx], -0.5, 1.5)
        
        new_score = _score(new_pts)
        
        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_score

def _polish_nelder_mead(pts, deadline):
    """Polish using Nelder-Mead on the flattened point set."""
    flat = pts.flatten().copy()
    
    def neg_score(x):
        p = x.reshape(N, 2)
        return -_score(p)
    
    try:
        result = minimize(neg_score, flat, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        polished = result.x.reshape(N, 2)
        score = _score(polished)
        return polished, score
    except Exception:
        return pts, _score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 105
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Start with jittered grid and run simulated annealing
    try:
        pts_grid = _jittered_grid(N, jitter=0.15, seed=42)
        sa_pts, sa_score = _simulated_annealing(pts_grid, deadline, max_iter=30000, T0=0.03, cooling=0.9997)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_pts is not None:
            return best_pts
        return _jittered_grid(N, jitter=0.15, seed=42)
    
    # Phase 2: Try multiple SA restarts with different seeds and initial points
    rng = np.random.RandomState(123)
    for restart in range(3):
        if time.time() > deadline:
            break
        
        if restart == 0:
            points = []
            for i in range(N):
                x = (i + 0.5) / N + rng.uniform(-0.05, 0.05)
                y = ((i * 7) % N + 0.5) / N + rng.uniform(-0.05, 0.05)
                points.append([np.clip(x, 0.01, 0.99), np.clip(y, 0.01, 0.99)])
            pts_init = np.array(points)
        elif restart == 1:
            pts_init = rng.rand(N, 2) * 0.9 + 0.05
        else:
            pts_init = _jittered_grid(N, jitter=0.2, seed=42 + restart * 100)
        
        try:
            sa_pts, sa_score = _simulated_annealing(pts_init, deadline, max_iter=20000, T0=0.02, cooling=0.9998)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
        except Exception:
            pass
    
    if time.time() > deadline:
        if best_pts is not None:
            return best_pts
        return _jittered_grid(N, jitter=0.15, seed=42)
    
    # Phase 3: Final polishing with Nelder-Mead
    if best_pts is not None:
        try:
            polished, polish_score = _polish_nelder_mead(best_pts, deadline)
            if polish_score > best_score:
                best_score = polish_score
                best_pts = polished
        except Exception:
            pass
    
    if best_pts is not None:
        return best_pts
    
    return _jittered_grid(N, jitter=0.15, seed=42)

# EVOLVE-BLOCK-END
