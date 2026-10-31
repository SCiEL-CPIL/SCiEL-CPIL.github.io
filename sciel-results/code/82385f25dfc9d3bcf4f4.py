"""Rewritten program for Heilbronn triangles task."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def point_in_triangle(p, v0, v1, v2):
    """Check if point p is inside triangle v0-v1-v2."""
    v01 = v1 - v0
    v02 = v2 - v0
    v0p = p - v0
    
    d00 = np.dot(v01, v01)
    d01 = np.dot(v01, v02)
    d02 = np.dot(v01, v0p)
    d11 = np.dot(v02, v02)
    d12 = np.dot(v02, v0p)
    
    den = d00 * d11 - d01 * d01
    if den == 0:
        return False
        
    u = (d11 * d02 - d01 * d12) / den
    v = (d00 * d12 - d01 * d02) / den
    
    return u >= -1e-9 and v >= -1e-9 and (u + v) <= 1 + 1e-9

def get_areas(points):
    """Calculate areas of all triangles formed by the points."""
    areas = []
    for i, j, k in itertools.combinations(range(N), 3):
        a = points[i]
        b = points[j]
        c = points[k]
        # 2D cross product magnitude
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        area = 0.5 * abs(cross)
        areas.append(area)
    return np.array(areas)

def objective_de(x):
    """Objective for Differential Evolution: Maximize min area.
    Returns negative penalty if out of bounds."""
    points = x.reshape(N, 2)
    for p in points:
        if not point_in_triangle(p, V0, V1, V2):
            return -1.0
    areas = get_areas(points)
    return np.min(areas)

def objective_local(x, beta):
    """Objective for local refinement: Minimize negative soft-min area."""
    points = x.reshape(N, 2)
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p, V0, V1, V2):
            penalty += 10.0
            
    if penalty > 0:
        return 100.0 + penalty # Large positive value to discourage
        
    areas = get_areas(points)
    min_a = np.min(areas)
    # Shift to prevent numerical issues in exp
    shifted = areas - min_a
    soft_min = min_a - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    # Bounds for Cartesian coordinates
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, np.sqrt(3)/2))
        
    best_x = None
    best_val = -1.0
    
    # 1. Global Search: Differential Evolution
    # Use 60% of the time budget for DE
    de_deadline = start_time + deadline * 0.6
    
    def de_callback(xk, convergence):
        if time.time() > de_deadline:
            return True
        return False

    try:
        res_de = differential_evolution(
            objective_de, 
            bounds, 
            maxiter=1000, 
            popsize=15, 
            tol=1e-5, 
            mutation=(0.5, 1.0), 
            recombination=0.7, 
            seed=42, 
            callback=de_callback, 
            polish=False
        )
        if res_de.fun > best_val:
            best_val = res_de.fun
            best_x = res_de.x
    except Exception:
        pass

    # Fallback if DE failed or found nothing valid
    if best_x is None or best_val <= 0:
        # Try random restarts
        for _ in range(20):
            if time.time() > start_time + deadline * 0.8:
                break
            pts = []
            for _ in range(N):
                u, v = np.random.rand(2)
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2) * v
                pts.append([x, y])
            pts = np.array(pts)
            val = objective_de(pts.flatten())
            if val > best_val:
                best_val = val
                best_x = pts.flatten()

    # If still no valid solution, return a known valid configuration
    if best_x is None:
        # Return vertices and centroid, plus random valid points
        pts = [V0, V1, V2, (V0+V1+V2)/3]
        while len(pts) < N:
            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            x = u + 0.5 * v
            y = (np.sqrt(3) / 2) * v
            pts.append([x, y])
        return np.array(pts)

    # 2. Local Refinement
    # Use remaining time (40%)
    local_deadline = start_time + deadline * 0.95
    
    current_x = best_x
    current_val = best_val
    
    # Anneal beta for soft-min
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > local_deadline:
            break
            
        def local_obj(x):
            return objective_local(x, beta)
            
        try:
            res_local = minimize(
                local_obj, 
                current_x, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            # Evaluate the raw min area of the new point
            new_points = res_local.x.reshape(N, 2)
            # Check validity
            valid = True
            for p in new_points:
                if not point_in_triangle(p, V0, V1, V2):
                    valid = False
                    break
            
            if valid:
                new_val = np.min(get_areas(new_points))
                if new_val > current_val:
                    current_val = new_val
                    current_x = res_local.x
        except Exception:
            pass

    final_points = current_x.reshape(N, 2)
    
    # Final safety check: ensure all points are inside
    for i in range(N):
        if not point_in_triangle(final_points[i], V0, V1, V2):
            final_points[i] = (V0 + V1 + V2) / 3.0
            
    return final_points
