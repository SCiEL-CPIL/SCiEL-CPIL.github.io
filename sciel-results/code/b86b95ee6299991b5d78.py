"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Equilateral triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    # Area of the equilateral triangle with side 1
    tri_area = np.sqrt(3) / 4.0
    
    def is_inside_triangle(p):
        """Check if point p is inside or on the boundary of the triangle."""
        # Barycentric coordinates
        v0 = B - A
        v1 = C - A
        v2 = p - A
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d02 = np.dot(v0, v2)
        d11 = np.dot(v1, v1)
        d12 = np.dot(v1, v2)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return False
        v = (d11 * d02 - d01 * d12) / denom
        w = (d00 * d12 - d01 * d02) / denom
        u = 1 - v - w
        return (u >= -1e-10) and (v >= -1e-10) and (w >= -1e-10)
    
    def triangle_area(p1, p2, p3):
        """Compute area of triangle formed by three points."""
        return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    def min_triangle_area(points):
        """Compute the minimum triangle area among all triples."""
        n = len(points)
        min_area = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    area = triangle_area(points[i], points[j], points[k])
                    if area < min_area:
                        min_area = area
        return min_area
    
    def objective(x):
        """Objective function: negative of the minimum triangle area (to minimize)."""
        points = x.reshape(N, 2)
        # Check if all points are inside the triangle
        for p in points:
            if not is_inside_triangle(p):
                return 1e10  # Large penalty
        min_area = min_triangle_area(points)
        return -min_area
    
    def constraint_func(x):
        """Constraints: points must be inside the triangle."""
        points = x.reshape(N, 2)
        # Use barycentric coordinate constraints
        # For each point, u, v, w >= 0 where p = u*A + v*B + w*C and u+v+w=1
        # p = v*B + w*C (since A is origin)
        # v = p[0] - 0.5*p[1]/(sqrt(3)/2) = p[0] - p[1]/sqrt(3)
        # w = p[1]/(sqrt(3)/2) = 2*p[1]/sqrt(3)
        # u = 1 - v - w
        constraints = []
        for i in range(N):
            px, py = points[i]
            v = px - py / np.sqrt(3)
            w = 2 * py / np.sqrt(3)
            u = 1 - v - w
            # u >= 0, v >= 0, w >= 0
            constraints.append(u)
            constraints.append(v)
            constraints.append(w)
        return constraints
    
    def make_constraints():
        """Create scipy constraint objects."""
        cons = []
        for i in range(N):
            # u = 1 - v - w >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[2*i] - x[2*i+1]/np.sqrt(3)) - 2*x[2*i+1]/np.sqrt(3)})
            # v = p[0] - p[1]/sqrt(3) >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*i+1]/np.sqrt(3)})
            # w = 2*p[1]/sqrt(3) >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 2*x[2*i+1]/np.sqrt(3)})
        return cons
    
    constraints = make_constraints()
    
    # Initial guess: place points at vertices and centroid-like positions
    # Use a grid-like initial placement
    rng = random.Random(42)
    
    # Start with a reasonable initial configuration
    # Place points at vertices, midpoints of edges, and center
    initial_points = []
    # Vertices
    initial_points.append(A.copy())
    initial_points.append(B.copy())
    initial_points.append(C.copy())
    # Midpoints of edges
    initial_points.append((A + B) / 2)
    initial_points.append((B + C) / 2)
    initial_points.append((C + A) / 2)
    # Center
    center = (A + B + C) / 3
    initial_points.append(center)
    
    # Add remaining points with some randomness
    for _ in range(N - len(initial_points)):
        u, v = rng.random(), rng.random()
        while u + v >= 1.0:
            u, v = rng.random(), rng.random()
        p = u * B + v * C  # A is origin
        initial_points.append(p)
    
    x0 = np.array(initial_points).flatten()
    
    best_points = np.array(initial_points)
    best_value = min_triangle_area(best_points) / tri_area
    
    # Try optimization from multiple starting points
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial solution
            x_start = x0.copy()
            for _ in range(N):
                idx = rng.randint(0, N-1)
                x_start[2*idx] += rng.gauss(0, 0.1)
                x_start[2*idx+1] += rng.gauss(0, 0.1)
                # Clamp to reasonable range
                x_start[2*idx] = np.clip(x_start[2*idx], 0, 1)
                x_start[2*idx+1] = np.clip(x_start[2*idx+1], 0, np.sqrt(3)/2)
        
        try:
            result = minimize(
                objective,
                x_start,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < objective(x_start):
                points = result.x.reshape(N, 2)
                # Verify points are inside
                valid = True
                for p in points:
                    if not is_inside_triangle(p):
                        valid = False
                        break
                if valid:
                    val = min_triangle_area(points) / tri_area
                    if val > best_value:
                        best_value = val
                        best_points = points.copy()
        except Exception:
            continue
    
    # Return the best points found
    return best_points
# EVOLVE-BLOCK-END
