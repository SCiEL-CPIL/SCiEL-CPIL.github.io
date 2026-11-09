"""Heilbronn triangle problem: place 11 points to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools
import random

N = 11
H = np.sqrt(3) / 2.0
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])
TRIA_AREA = 0.5 * np.sqrt(3)  # area of the unit base equilateral triangle


def project_to_triangle(pts):
    """Project points to be inside the triangle (0,0), (1,0), (0.5, H)."""
    pts = pts.copy()
    # Clip to bounding box
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, H)
    
    # Check three edges:
    # Edge AB: y >= 0 (already handled by clip)
    # Edge AC: y <= sqrt(3)*x  =>  sqrt(3)*x - y >= 0
    # Edge BC: y <= -sqrt(3)*(x-1) + sqrt(3) = -sqrt(3)*x + 2*sqrt(3)
    #         => sqrt(3)*x + y <= 2*sqrt(3)
    
    # For points violating AC: y > sqrt(3)*x
    # Project onto line y = sqrt(3)*x
    # The line is sqrt(3)*x - y = 0. Normal vector (sqrt(3), -1), norm = 2.
    # Signed distance = (sqrt(3)*x - y)/2
    # For points where sqrt(3)*x - y < 0, we need to project.
    d_ac = np.sqrt(3) * pts[:, 0] - pts[:, 1]
    mask_ac = d_ac < -1e-12
    if np.any(mask_ac):
        # Projection: p - ((sqrt(3)*x - y)/2) * (sqrt(3)/2, -1/2) * 2
        # Actually: p_new = p - (d/|n|^2) * n where n = (sqrt(3), -1), |n|^2 = 4
        # d = (sqrt(3)*x - y)
        # p_new = p - d/4 * (sqrt(3), -1)
        d = d_ac[mask_ac]
        pts[mask_ac, 0] -= d * np.sqrt(3) / 4.0
        pts[mask_ac, 1] += d / 4.0
    
    # For points violating BC: sqrt(3)*x + y > 2*sqrt(3)
    # Line: sqrt(3)*x + y - 2*sqrt(3) = 0. Normal (sqrt(3), 1), norm = 2.
    # Signed distance = (sqrt(3)*x + y - 2*sqrt(3))/2
    d_bc = np.sqrt(3) * pts[:, 0] + pts[:, 1] - 2 * np.sqrt(3)
    mask_bc = d_bc > 1e-12
    if np.any(mask_bc):
        d = d_bc[mask_bc]
        pts[mask_bc, 0] -= d * np.sqrt(3) / 4.0
        pts[mask_bc, 1] -= d / 4.0
    
    return pts


def min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def objective(x):
    """Negative of the minimum triangle area (to minimize)."""
    pts = x.reshape(N, 2)
    pts = project_to_triangle(pts)
    return -min_triangle_area(pts)


def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    best_points = None
    best_val = 0.0
    
    # Start with a reasonable initial configuration
    # Use a grid-like or known good starting point
    # Let's start with points spread out in the triangle
    rng = random.Random(42)
    
    # Initial guess: place points somewhat evenly
    initial = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.25, H/4],
        [0.75, H/4],
        [0.5, H/2],
        [0.125, H/8],
        [0.875, H/8],
        [0.375, 3*H/8],
        [0.625, 3*H/8],
        [0.5, 3*H/4],
    ])
    initial = project_to_triangle(initial)
    
    # Evaluate initial
    val = -objective(initial.flatten())
    if val > best_val:
        best_val = val
        best_points = initial.copy()
    
    # Try multiple random restarts with local optimization
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Generate random initial points
        if restart == 0:
            x0 = initial.flatten()
        else:
            # Random points in triangle
            x0 = np.zeros(N * 2)
            for i in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x0[2*i] = u + 0.5 * v
                x0[2*i+1] = H * v
        
        # Local optimization using Nelder-Mead
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            pts = project_to_triangle(result.x.reshape(N, 2))
            val = -objective(result.x)
            
            if val > best_val:
                best_val = val
                best_points = pts.copy()
                
        except Exception:
            continue
    
    # Additional refinement with more iterations on best point
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            result = minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            pts = project_to_triangle(result.x.reshape(N, 2))
            val = -objective(result.x)
            if val > best_val:
                best_val = val
                best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return the initial
        best_points = initial
    
    return project_to_triangle(best_points)
