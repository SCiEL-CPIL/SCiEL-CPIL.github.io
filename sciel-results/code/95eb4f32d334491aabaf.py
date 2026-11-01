import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 50

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _repulsion_step(points, step_size=0.01, iterations=100):
    points = points.copy()
    for _ in range(iterations):
        forces = np.zeros_like(points)
        for i in range(len(points)):
            for j in range(len(points)):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                # Repulsive force 1/r^2
                force = diff / (dist_sq * np.sqrt(dist_sq)) # 1/r^3 direction, magnitude 1/r^2? 
                # Standard electrostatic: F = k * (ri - rj) / ||ri - rj||^3
                forces[i] += force
        points = points + step_size * forces
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
    return points

def _get_circumcenters_and_radii(points):
    """
    Returns list of (circumcenter, radius, [indices]) for each facet of the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    
    facets = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        A = points[simplex[0]]
        B = points[simplex[1]]
        C = points[simplex[2]]
        
        # Calculate circumcenter of spherical triangle ABC
        # V = A(a-c) + B(a-b) + C(b-c) where a=B.C, b=C.A, c=A.B
        a = np.dot(B, C)
        b = np.dot(C, A)
        c = np.dot(A, B)
        
        V = A * (a - c) + B * (a - b) + C * (b - c)
        
        norm_V = np.linalg.norm(V)
        if norm_V < 1e-10:
            # Degenerate or antipodal?
            continue
            
        U = V / norm_V
        
        # Radius is angular distance from U to A
        dot_val = np.dot(U, A)
        # Clamp to [-1, 1] for arccos
        dot_val = np.clip(dot_val, -1.0, 1.0)
        radius = np.arccos(dot_val)
        
        facets.append((U, radius, simplex))
        
    return facets

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initialization
    points = _fibonacci_spiral(N)
    
    # 2. Repulsion phase to establish global structure
    # Use a decaying step size
    current_step = 0.05
    for i in range(200):
        if time.time() > deadline - 30.0:
            break
        points = _repulsion_step(points, step_size=current_step, iterations=1)
        current_step *= 0.95
        
    # 3. Local Optimization Phase
    best_radius = float('inf')
    best_points = points.copy()
    
    # Get initial facets
    facets = _get_circumcenters_and_radii(points)
    if facets:
        max_rad = max(f[1] for f in facets)
        best_radius = max_rad
    else:
        # Fallback if hull fails
        return points

    step_size = 0.05
    
    while time.time() < deadline:
        facets = _get_circumcenters_and_radii(points)
        if not facets:
            break
            
        # Find worst facet
        max_rad = 0
        worst_idx = -1
        for i, (U, r, simplex) in enumerate(facets):
            if r > max_rad:
                max_rad = r
                worst_idx = i
                
        if worst_idx == -1:
            break
            
        U, r, simplex = facets[worst_idx]
        
        # Check if we improved
        if r < best_radius:
            best_radius = r
            best_points = points.copy()
            
        # Move vertices of the worst facet towards the circumcenter
        # This shrinks the facet
        A_idx, B_idx, C_idx = simplex
        
        # Tangent projection of (U - A)
        # We want to move A towards U on the sphere
        # Direction is U - A, projected to tangent plane at A
        # Tangent vector T = (U - A) - ((U-A).A)A
        # Since A is unit, (U-A).A = U.A - 1.
        
        # Simpler: just move towards U and normalize
        # But we should ensure we don't cross other points or create invalid hulls immediately
        # A small step is safer
        
        new_A = points[A_idx] + step_size * (U - points[A_idx])
        new_B = points[B_idx] + step_size * (U - points[B_idx])
        new_C = points[C_idx] + step_size * (U - points[C_idx])
        
        points[A_idx] = _normalize(new_A)
        points[B_idx] = _normalize(new_B)
        points[C_idx] = _normalize(new_C)
        
        # Adaptive step size: if radius increases significantly, reduce step
        # We don't compute new radius immediately to save time, but we can check next iteration
        # If we detect stagnation, reduce step
        # Simple heuristic: if we've done many steps without improvement, reduce step
        # (Not tracking improvement count strictly here for simplicity, but step decay helps)
        step_size *= 0.98
        
        # Occasionally re-run repulsion to fix local minima?
        # Maybe every 100 local steps, do a small repulsion pass
        # This helps escape local minima where points are stuck
        if int((time.time() - start_time) * 10) % 10 == 0: # Every 0.1s? No, let's do it based on loop count
            pass

    return best_points
