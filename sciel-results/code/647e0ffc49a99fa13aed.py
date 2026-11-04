"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
VERTS = np.array([V0, V1, V2])

def _project_to_triangle(p):
    """Project point p onto the equilateral triangle defined by VERTS."""
    # Convert to barycentric coordinates
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # Using the formula for barycentric coordinates
    d0 = np.dot(V0, V0) - 2 * np.dot(V0, V1) + np.dot(V1, V1)
    d1 = np.dot(V1, V1) - 2 * np.dot(V1, V2) + np.dot(V2, V2)
    d2 = np.dot(V2, V2) - 2 * np.dot(V2, V0) + np.dot(V0, V0)
    
    # Actually, let's use a simpler approach: clip barycentric coordinates
    # Barycentric coordinates relative to V0, V1, V2
    # Solve: p = (1-a-b)*V0 + a*V1 + b*V2
    # p - V0 = a*(V1-V0) + b*(V2-V0)
    # [V1-V0, V2-V0] [a, b]^T = p - V0
    
    M = np.column_stack([V1 - V0, V2 - V0])
    rhs = p - V0
    try:
        ab = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return V0.copy()
    
    a, b = ab
    c = 1.0 - a - b
    
    # Clip to simplex
    if a < 0:
        a = 0
        # renormalize
        s = b + c
        if s < 1e-12:
            return V0.copy()
        b /= s
        c /= s
    if b < 0:
        b = 0
        s = a + c
        if s < 1e-12:
            return V0.copy()
        a /= s
        c /= s
    if c < 0:
        c = 0
        s = a + b
        if s < 1e-12:
            return V0.copy()
        a /= s
        b /= s
    
    return (1 - a - b) * V0 + a * V1 + b * V2

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _objective(params, points_init):
    """Objective function for optimization: negative of minimum triangle area."""
    n = len(params) // 2
    pts = params.reshape(n, 2)
    # Project each point to stay in triangle
    for i in range(n):
        pts[i] = _project_to_triangle(pts[i])
    return -_min_triangle_area(pts)

def _make_initial_grid():
    """Create a grid-like initial configuration."""
    points = []
    # Place points in a structured pattern
    # Use a hexagonal-like arrangement
    for row in range(4):
        for col in range(3):
            if len(points) >= N:
                break
            # Barycentric coordinates
            a = col / 2.5
            b = row / 3.0
            c = 1 - a - b
            if c < 0:
                continue
            pt = a * V0 + b * V1 + c * V2
            points.append(pt)
    
    # Fill remaining points
    while len(points) < N:
        pt = _project_to_triangle(np.array([0.5, np.sqrt(3) / 6.0]))
        points.append(pt + np.array([0.01 * len(points), 0.01 * len(points)]))
    
    return np.array(points[:N])

def _make_random_initial(seed):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(2)
        if u + v < 1.0:
            pt = u * V0 + v * V1 + (1 - u - v) * V2
            points.append(pt)
    return np.array(points)

def _local_optimize(points_init, max_iters=1000):
    """Run local optimization from initial points."""
    n = len(points_init)
    x0 = points_init.flatten()
    
    # Use Nelder-Mead for derivative-free optimization
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': max_iters,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    pts = result.x.reshape(n, 2)
    for i in range(n):
        pts[i] = _project_to_triangle(pts[i])
    
    return pts, -result.fun

def construct():
    """Construct 11 points in the equilateral triangle to maximize minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = []
    
    # 1. Grid-like structured initialization
    initializations.append(_make_initial_grid())
    
    # 2. Random initializations with different seeds
    for seed in range(20):
        initializations.append(_make_random_initial(seed))
    
    # 3. Symmetric patterns
    # Center + ring
    center = np.array([0.5, np.sqrt(3) / 6.0])
    ring_points = []
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        r = 0.3
        pt = center + r * np.array([np.cos(rad), np.sin(rad)])
        pt = _project_to_triangle(pt)
        ring_points.append(pt)
    if len(ring_points) >= 10:
        initializations.append(np.array([center] + ring_points[:10]))
    
    # Evaluate each initialization and run local optimization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Quick evaluation of initial score
        init_score = _min_triangle_area(init_pts)
        
        # Run local optimization
        opt_pts, opt_score = _local_optimize(init_pts)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
        
        # If time is running low, just use the best initial so far
        if time.time() > deadline - 20.0:
            if init_score > best_score:
                best_score = init_score
                best_points = init_pts
    
    # Final fallback
    if best_points is None:
        best_points = _make_random_initial(42)
    
    return best_points
