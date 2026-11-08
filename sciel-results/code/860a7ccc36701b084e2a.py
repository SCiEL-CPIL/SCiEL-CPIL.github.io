import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0
# Vertices of the standard equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute indices for all C(11, 3) = 165 triangles
combos = list(combinations(range(N), 3))
idx_i = np.array([c[0] for c in combos])
idx_j = np.array([c[1] for c in combos])
idx_k = np.array([c[2] for c in combos])

def _min_triangle_area(points):
    """
    Calculate the minimum area of all triangles formed by the points.
    points: shape (N, 2)
    """
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _project_to_triangle(p):
    """
    Project a single point p to the closest point on the triangle T.
    """
    # Barycentric coordinates
    # T is defined by V0, V1, V2
    # We check if p is inside.
    # Edge 1: V0-V1 (y=0)
    # Edge 2: V1-V2
    # Edge 3: V2-V0
    
    # Function to project point to line segment
    def _project_to_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        return a + t * ab

    # Check if inside using barycentric coordinates or half-plane tests
    # Normal vectors pointing inwards
    # Edge V0-V1: normal (0, 1). Dot(p - V0, n) >= 0
    # Edge V1-V2: normal. V2-V1 = (-0.5, H). Normal = (H, 0.5). Dot(p - V1, n) >= 0
    # Edge V2-V0: normal. V0-V2 = (-0.5, -H). Normal = (-H, 0.5). Dot(p - V2, n) >= 0
    
    n1 = np.array([0.0, 1.0])
    n2 = np.array([H, 0.5])
    n3 = np.array([-H, 0.5])
    
    d1 = np.dot(p - V0, n1)
    d2 = np.dot(p - V1, n2)
    d3 = np.dot(p - V2, n3)
    
    if d1 >= 0 and d2 >= 0 and d3 >= 0:
        return p
        
    # If outside, project to the edge with the most negative distance?
    # Or just project to all edges and pick the one that minimizes distance to original p?
    # Actually, for convex polygons, if a point is outside, it is outside at least one edge.
    # The closest point on the polygon is on one of the edges or vertices.
    # A simple robust method: project to each of the 3 edges and 3 vertices, return the closest.
    
    candidates = []
    candidates.append(_project_to_segment(p, V0, V1))
    candidates.append(_project_to_segment(p, V1, V2))
    candidates.append(_project_to_segment(p, V2, V0))
    candidates.append(V0)
    candidates.append(V1)
    candidates.append(V2)
    
    candidates = np.array(candidates)
    dists = np.sum((candidates - p)**2, axis=1)
    min_idx = np.argmin(dists)
    return candidates[min_idx]

def _project_points(points):
    """
    Project all points to the triangle.
    """
    projected = np.array([_project_to_triangle(p) for p in points])
    return projected

def _objective_flat(x):
    """
    Objective function for the optimizer.
    x: flat array of length 2*N
    Returns negative min area (to maximize min area, we minimize -min area)
    """
    points = x.reshape(N, 2)
    points = _project_points(points)
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    rng = random.Random(42)
    
    # 1. Initialize with a structured grid + noise
    # We want points to be well-distributed.
    # A simple grid in barycentric coordinates or just cartesian with rejection.
    
    # Let's try a few different initializations.
    # Init 1: Random
    # Init 2: Grid-like
    
    # We will run multiple restarts of Nelder-Mead.
    
    # To save time, we limit the number of restarts and the maxiter for NM.
    max_restarts = 20
    max_iter_per_restart = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial points
        if restart == 0:
            # Structured: Place points on a triangular grid
            # For N=11, we can't make a perfect grid. 
            # Let's try to place them somewhat regularly.
            # 1 point center, 6 around, 4 corners? No, corners are bad for min area?
            # Actually, Heilbronn configurations often avoid corners for N>4?
            # For N=11, the optimal configuration is known to be irregular.
            # Let's start with a jittered grid.
            # Create a 4x4 grid in the triangle?
            # Simpler: Random but constrained to be "spread out"
            points = np.zeros((N, 2))
            # Place first 3 at vertices? No, that makes small areas.
            # Let's use a random initialization but with a "repulsion" force initially?
            # Or just random.
            
            # Better: Use the known approximate structure if available, or just random.
            # Given the short time, random + NM is the way.
            # But to help NM, let's try to initialize with points that are not too close.
            
            # Simple random initialization
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        points[i] = [u + 0.5 * v, H * v]
                        break
        else:
            # Perturb the best found so far
            if best_points is not None:
                points = best_points + rng.gauss(0, 0.05, (N, 2))
            else:
                for i in range(N):
                    while True:
                        u, v = rng.random(), rng.random()
                        if u + v < 1.0:
                            points[i] = [u + 0.5 * v, H * v]
                            break
        
        x0 = points.flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(_objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter_per_restart, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < best_score:
                best_points = res.x.reshape(N, 2)
                best_score = -res.fun
        except Exception:
            pass

    # If no optimization ran or failed, return best random
    if best_points is None:
        # Fallback to random
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        best_points = np.array(points)
        
    # Final check: ensure points are inside
    best_points = _project_points(best_points)
    
    return best_points
