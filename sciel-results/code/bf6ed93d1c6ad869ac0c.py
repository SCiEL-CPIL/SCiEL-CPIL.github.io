"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations

def _min_triangle_area_vec(pts):
    """Compute minimum triangle area over all triples using vectorized ops."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized cross product for all k > j
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            areas = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _convex_hull_area(pts):
    """Compute area of convex hull using Andrew's monotone chain."""
    pts = np.array(pts)
    pts = np.unique(pts, axis=0)
    n = len(pts)
    if n <= 2:
        return 0.0
    pts = np.sort(pts, axis=0)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area

def _score(pts):
    """Compute score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_score = 0.0
    
    def try_improve(pts, label=""):
        nonlocal best_pts, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            return s
        return s
    
    # Initial candidate: regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    pts14 = np.column_stack([np.cos(angles), np.sin(angles)])
    try_improve(pts14, "regular14")
    
    # Two rings: 7 inner + 7 outer
    angles7 = 2 * np.pi * np.arange(7) / 7
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    inner = 0.5 * np.column_stack([np.cos(angles7 + np.pi/7), np.sin(angles7 + np.pi/7)])
    pts2ring = np.vstack([outer, inner])
    try_improve(pts2ring, "two_ring")
    
    # Three rings: 5+5+4
    a5 = 2 * np.pi * np.arange(5) / 5
    a4 = 2 * np.pi * np.arange(4) / 4
    r1 = 0.35
    r2 = 0.7
    ring1 = r1 * np.column_stack([np.cos(a5), np.sin(a5)])
    ring2 = r2 * np.column_stack([np.cos(a5 + np.pi/5), np.sin(a5 + np.pi/5)])
    ring3 = np.column_stack([np.cos(a4), np.sin(a4)])
    pts3ring = np.vstack([ring1, ring2, ring3])
    try_improve(pts3ring, "three_ring")
    
    # Grid-like: 4x4 grid with corners removed, adjusted
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    grid = (grid - 1.5) / 1.5  # center and scale
    # Remove 2 corner points and add 2 in center
    grid = grid[2:]  # remove first 2
    center_pts = np.array([[0.3, 0.3], [-0.3, -0.3]])
    pts_grid = np.vstack([grid, center_pts])
    try_improve(pts_grid, "grid")
    
    # Hexagonal packing approximation
    hex_pts = []
    # Center
    hex_pts.append([0, 0])
    # Ring 1: 6 points
    for i in range(6):
        a = 2 * np.pi * i / 6
        hex_pts.append([np.cos(a), np.sin(a)])
    # Ring 2: need 7 more, use 7 points on ring 2
    for i in range(7):
        a = 2 * np.pi * i / 7 + np.pi/7
        hex_pts.append([1.8 * np.cos(a), 1.8 * np.sin(a)])
    pts_hex = np.array(hex_pts)
    try_improve(pts_hex, "hex")
    
    # Local optimization using Nelder-Mead on the best candidate
    if best_pts is not None:
        from scipy.optimize import minimize
        
        def neg_score_flat(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        x0 = best_pts.flatten()
        
        # Multiple restarts with different initial perturbations
        rng = np.random.RandomState(42)
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            if restart == 0:
                x_start = x0
            else:
                # Add noise
                noise_scale = 0.05 * (1 + restart * 0.02)
                x_start = x0 + rng.randn(N * 2) * noise_scale
            
            try:
                result = minimize(
                    neg_score_flat,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                s = -result.fun
                if s > best_score:
                    best_score = s
                    best_pts = result.x.reshape(N, 2).copy()
            except Exception:
                pass
        
        # Second round with finer tuning on the best found
        if best_pts is not None:
            x0 = best_pts.flatten()
            for restart in range(10):
                if time.time() > deadline:
                    break
                noise_scale = 0.01 * (1 + restart * 0.01)
                x_start = x0 + rng.randn(N * 2) * noise_scale
                try:
                    result = minimize(
                        neg_score_flat,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 300,
                            'xatol': 1e-7,
                            'fatol': 1e-9,
                            'adaptive': True
                        }
                    )
                    s = -result.fun
                    if s > best_score:
                        best_score = s
                        best_pts = result.x.reshape(N, 2).copy()
                except Exception:
                    pass
    
    # Final affine optimization: try to optimize the hull shape
    if best_pts is not None:
        # Apply various affine transformations and re-optimize
        from scipy.optimize import minimize_scalar
        
        for angle in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
            if time.time() > deadline:
                break
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]])
            pts_rot = best_pts @ rot.T
            try_improve(pts_rot, f"rot_{angle:.2f}")
        
        # Try shear
        for shear in [-0.5, -0.3, 0, 0.3, 0.5]:
            if time.time() > deadline:
                break
            sh = np.array([[1, shear], [0, 1]])
            pts_sh = best_pts @ sh.T
            try_improve(pts_sh, f"shear_{shear:.1f}")
    
    # Ensure we return a valid result
    if best_pts is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
# EVOLVE-BLOCK-END
