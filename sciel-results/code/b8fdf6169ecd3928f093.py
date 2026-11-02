"""Heilbronn triangle problem: place 11 points in unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3.0)
# Vertices of unit-area equilateral triangle
# Actually the triangle (0,0), (1,0), (0.5, S3/2) has area S3/4.
# The task says "unit-area triangle" but then uses this specific triangle.
# Let's check: area = 0.5 * base * height = 0.5 * 1 * S3/2 = S3/4 ≈ 0.433.
# The task says "divides by its area", so we need to normalize.
# But the evaluator measures "raw value". Let me re-read.
# "The evaluator measures the raw value (higher is better)."
# "The notebook uses an equilateral triangle and divides by its area."
# So the score is min_triangle_area / triangle_area = min_triangle_area / (S3/4).
# But wait, the seed program just returns points. The evaluator must compute the score.
# Let me assume the evaluator computes: min over all triples of area(triple) / area(triangle).
# To maximize this, we just need to maximize min triangle area since triangle area is fixed.

def point_in_triangle(p, a, b, c):
    """Check if point p is inside triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def barycentric_to_cartesian(bary, a, b, c):
    """Convert barycentric coordinates to Cartesian."""
    return bary[0] * a + bary[1] * b + bary[2] * c

def compute_min_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(x, a, b, c):
    """Objective function for optimization. x is 2N flat array of coordinates."""
    points = x.reshape(N, 2)
    # Check if all points are inside the triangle
    for p in points:
        if not point_in_triangle(p, a, b, c):
            return 1e10  # Large penalty
    return -compute_min_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, S3/2])
    
    # Use barycentric coordinates to ensure points stay inside
    # Barycentric coords (u, v) with u >= 0, v >= 0, u+v <= 1
    # Cartesian: p = u*a + v*b + (1-u-v)*c
    
    def bary_to_cart(bary_coords):
        """Convert barycentric coords (u,v) where p = u*a + v*b + (1-u-v)*c to Cartesian."""
        u = bary_coords[0::2]
        v = bary_coords[1::2]
        w = 1.0 - u - v
        # Check validity
        if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
            return None
        # Clip small negative values
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        w = np.clip(w, 0, 1)
        # Normalize if needed
        s = u + v + w
        if s < 1e-12:
            return None
        u /= s
        v /= s
        w /= s
        points = u[:, np.newaxis] * a + v[:, np.newaxis] * b + w[:, np.newaxis] * c
        return points
    
    def objective_bary(x):
        """Objective in barycentric coordinates."""
        points = bary_to_cart(x)
        if points is None:
            return 1e10
        return -compute_min_area(points)
    
    # Differential evolution for global optimization
    # Bounds: u in [0,1], v in [0,1], with constraint u+v <= 1
    # We'll use bounds [0,1] for both and rely on the constraint handling
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple random restarts with differential evolution
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Random initial population
        u_init = rng.random(N)
        v_init = rng.random(N)
        # Ensure u+v <= 1
        for i in range(N):
            if u_init[i] + v_init[i] > 1:
                scale = 1.0 / (u_init[i] + v_init[i])
                u_init[i] *= scale
                v_init[i] *= scale
        
        x0 = np.zeros(2*N)
        x0[0::2] = u_init
        x0[1::2] = v_init
        
        # Use differential evolution
        bounds = [(0, 1)] * (2*N)
        
        try:
            result = differential_evolution(
                objective_bary, bounds, seed=restart, maxiter=100, 
                popsize=15, tol=1e-4, update_interval=10,
                init='latinhypercube'
            )
            if result.success or result.fun < best_score:
                points = bary_to_cart(result.x)
                if points is not None:
                    score = compute_min_area(points)
                    if score > best_score:
                        best_score = score
                        best_result = points.copy()
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead on the best result
    if best_result is not None and time.time() < deadline - 5:
        try:
            def obj_local(x):
                points = bary_to_cart(x)
                if points is None:
                    return 1e10
                return -compute_min_area(points)
            
            x_best = np.zeros(2*N)
            for i in range(N):
                # Convert Cartesian back to barycentric
                p = best_result[i]
                # Solve for u, v, w where p = u*a + v*b + w*c, u+v+w=1
                # a=(0,0), b=(1,0), c=(0.5, S3/2)
                # x = v*1 + w*0.5
                # y = w*S3/2
                # So w = y / (S3/2) = 2y/S3
                # v = x - w*0.5 = x - y/S3
                # u = 1 - v - w
                w = p[1] / (S3/2)
                v = p[0] - w*0.5
                u = 1 - v - w
                x_best[2*i] = u
                x_best[2*i+1] = v
            
            result = minimize(obj_local, x_best, method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            points = bary_to_cart(result.x)
            if points is not None:
                score = compute_min_area(points)
                if score > best_score:
                    best_score = score
                    best_result = points.copy()
        except Exception:
            pass
    
    # If no valid result, fall back to a simple grid
    if best_result is None:
        # Place points in a grid pattern
        best_result = np.zeros((N, 2))
        # 3x3 grid plus 2 extra points
        positions = []
        for i in range(3):
            for j in range(3):
                u = i/2 * 0.8
                v = j/2 * 0.8
                if u + v <= 1:
                    positions.append([u, v])
        # Add 2 more points
        positions.append([0.4, 0.4])
        positions.append([0.5, 0.3])
        
        for i in range(N):
            u, v = positions[i]
            w = 1 - u - v
            best_result[i] = u * a + v * b + w * c
    
    return best_result
