"""Heilbronn triangles: maximize smallest triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3 / 2.0])
TRI_AREA = 0.5 * S3 / 2.0  # area of the equilateral triangle with side 1

DEADLINE = 120.0  # seconds; leave safety margin


def _barycentric(p):
    """Compute barycentric coordinates of p w.r.t. triangle V0,V1,V2."""
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # Using the standard formula:
    # a = ((V2[1]-V1[1])*(p[0]-V1[0]) - (V2[0]-V1[0])*(p[1]-V1[1])) / denom
    # b = ((V0[1]-V2[1])*(p[0]-V1[0]) - (V0[0]-V2[0])*(p[1]-V1[1])) / denom
    # c = 1 - a - b
    denom = (V2[1] - V1[1]) * (V0[0] - V1[0]) - (V2[0] - V1[0]) * (V0[1] - V1[1])
    a = ((V2[1] - V1[1]) * (p[0] - V1[0]) - (V2[0] - V1[0]) * (p[1] - V1[1])) / denom
    b = ((V0[1] - V2[1]) * (p[0] - V1[0]) - (V0[0] - V2[0]) * (p[1] - V1[1])) / denom
    c = 1.0 - a - b
    return np.array([a, b, c])


def _min_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                                 (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1]))
                if area < min_a:
                    min_a = area
    return min_a


def _objective_flat(x):
    """Objective: minimize negative of (min_area - penalty). We want to maximize min_area.
    Using a soft constraint via penalty for points outside triangle."""
    pts = x.reshape(N, 2)
    
    # Compute barycentric coordinates for all points
    # For each point, check if inside triangle
    # Penalty for being outside
    total_penalty = 0.0
    for i in range(N):
        bc = _barycentric(pts[i])
        # If any barycentric coord < 0, point is outside
        min_bc = np.min(bc)
        if min_bc < 0:
            total_penalty += min_bc * min_bc * 100.0  # quadratic penalty
    
    min_a = _min_area(pts)
    # We want to maximize min_a, so minimize -(min_a) + penalty
    return -(min_a) + total_penalty


def _gradient_flat(x):
    """Numerical gradient for the objective."""
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _objective_flat(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _objective_flat(x_plus)
        x_minus = x.copy()
        x_minus[i] -= eps
        f_minus = _objective_flat(x_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def _lattice_init():
    """Create a structured lattice-based initial configuration."""
    # Use a grid in barycentric coordinates, then jitter
    # 11 points: try to spread them out
    # One approach: 3x3 grid minus corners, plus center, plus one more
    # Or use a more systematic approach
    
    # Let's use a combination of vertices, edge midpoints, and interior points
    # Vertices: V0, V1, V2
    # Edge midpoints: (V0+V1)/2, (V1+V2)/2, (V2+V0)/2
    # Center: centroid
    # That's 7 points. Need 4 more.
    
    # Let's use a 4x4 grid in barycentric space, pick 11 points
    # Actually, let's just create a reasonable spread
    
    pts = []
    # Vertices with small inset
    eps = 0.05
    pts.append(V0 + eps * (V1 + V2 - 2*V0))  # inset from V0
    pts.append(V1 + eps * (V0 + V2 - 2*V1))  # inset from V1
    pts.append(V2 + eps * (V0 + V1 - 2*V2))  # inset from V2
    
    # Edge midpoints
    pts.append((V0 + V1) / 2)
    pts.append((V1 + V2) / 2)
    pts.append((V2 + V0) / 2)
    
    # Centroid
    pts.append((V0 + V1 + V2) / 3)
    
    # 4 more interior points - use a small grid
    # In barycentric coords, use points like (0.3, 0.3, 0.4), etc.
    extra_bary = [
        [0.25, 0.25, 0.50],
        [0.50, 0.25, 0.25],
        [0.25, 0.50, 0.25],
        [0.40, 0.40, 0.20],
    ]
    for bc in extra_bary:
        p = bc[0] * V0 + bc[1] * V1 + bc[2] * V2
        pts.append(p)
    
    return np.array(pts)


def _random_init(seed):
    """Random initialization inside the triangle."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            p = u * V1 + v * V2
            pts.append(p)
    return np.array(pts)


def _refine(pts, time_budget):
    """Refine points using Nelder-Mead optimization."""
    x0 = pts.flatten()
    
    # Use Nelder-Mead with numerical gradient
    options = {
        'maxiter': 5000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True,
    }
    
    result = minimize(
        _objective_flat,
        x0,
        method='Nelder-Mead',
        options=options
    )
    
    pts_refined = result.x.reshape(N, 2)
    
    # Check if points are inside triangle; if not, project back
    for i in range(N):
        bc = _barycentric(pts_refined[i])
        if np.min(bc) < -1e-6:
            # Project back to nearest point in triangle
            # Simple approach: clip barycentric coords
            bc_clipped = np.clip(bc, 0, None)
            bc_clipped /= bc_clipped.sum()
            pts_refined[i] = bc_clipped[0] * V0 + bc_clipped[1] * V1 + bc_clipped[2] * V2
    
    return pts_refined


def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = []
    
    # Lattice-based init
    inits.append(_lattice_init())
    
    # Several random inits with different seeds
    for seed in range(5):
        inits.append(_random_init(seed))
    
    best_pts = None
    best_score = -float('inf')
    
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:  # Leave 15s safety margin
            break
        
        # Refine this initialization
        refined = _refine(init, DEADLINE - elapsed)
        
        # Compute the actual score (normalized by triangle area)
        min_a = _min_area(refined)
        score = min_a / TRI_AREA
        
        if score > best_score:
            best_score = score
            best_pts = refined.copy()
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
    
    # If we didn't find any valid points (shouldn't happen), fallback
    if best_pts is None:
        best_pts = _lattice_init()
    
    # Final check: ensure all points are inside or on the triangle
    for i in range(N):
        bc = _barycentric(best_pts[i])
        if np.min(bc) < -1e-9:
            bc_clipped = np.clip(bc, 0, None)
            if bc_clipped.sum() > 0:
                bc_clipped /= bc_clipped.sum()
                best_pts[i] = bc_clipped[0] * V0 + bc_clipped[1] * V1 + bc_clipped[2] * V2
    
    return best_pts
