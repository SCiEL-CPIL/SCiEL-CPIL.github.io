import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
DEADLINE = 140.0  # Leave 10s margin for 150s limit


def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])


def _covering_radius_exact(points):
    """Compute exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return _covering_radius_grid(points)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        v1 = points[simplex[0]]
        v2 = points[simplex[1]]
        v3 = points[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-15:
            continue
        
        n_unit = n / n_norm
        d = np.dot(n_unit, v1)
        d_abs = abs(d)
        if d_abs > 1.0:
            d_abs = 1.0
        
        angle = np.arccos(d_abs)
        if angle > max_radius:
            max_radius = angle
    
    return np.degrees(max_radius)


def _covering_radius_grid(points, n_theta=360, n_phi=720):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    max_dist = 0.0
    chunk_size = 10000
    for start in range(0, len(grid_points), chunk_size):
        end = min(start + chunk_size, len(grid_points))
        chunk = grid_points[start:end]
        dots = np.clip(chunk @ points.T, -1.0, 1.0)
        max_dots = dots.max(axis=1)
        angles = np.arccos(max_dots)
        if angles.max() > max_dist:
            max_dist = angles.max()
    
    return np.degrees(max_dist)


def _refine_facets(points, n_iters=200, step_size=0.02):
    """Refine points by moving vertices of worst facets toward facet normals."""
    points = points.copy()
    
    for it in range(n_iters):
        if time.time() > DEADLINE:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Compute circumradius for each facet
        radii = np.zeros(len(hull.simplices))
        normals = []
        for idx, simplex in enumerate(hull.simplices):
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            e1 = v2 - v1
            e2 = v3 - v1
            n = np.cross(e1, e2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                normals.append(np.zeros(3))
                radii[idx] = 0.0
                continue
            n_unit = n / n_norm
            d = abs(np.dot(n_unit, v1))
            d = min(max(d, 0.0), 1.0)
            radii[idx] = np.arccos(d)
            normals.append(n_unit)
        
        normals = np.array(normals)
        
        # Find the worst facets
        max_radius = np.max(radii)
        if max_radius < 1e-10:
            break
        
        # Move vertices of facets with radius > 0.9 * max_radius
        threshold = 0.9 * max_radius
        worst_mask = radii > threshold
        
        if not np.any(worst_mask):
            break
        
        # Accumulate displacement for each point
        displacement = np.zeros_like(points)
        count = np.zeros(N)
        
        for idx in np.where(worst_mask)[0]:
            simplex = hull.simplices[idx]
            n_unit = normals[idx]
            for vi in simplex:
                # Move vertex toward the normal direction
                # Project n_unit onto the tangent plane at points[vi]
                tangent = n_unit - np.dot(n_unit, points[vi]) * points[vi]
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-12:
                    displacement[vi] += tangent / tangent_norm
                    count[vi] += 1
        
        # Apply displacement
        for i in range(N):
            if count[i] > 0:
                avg_disp = displacement[i] / count[i]
                new_pt = points[i] + step_size * avg_disp
                new_norm = np.linalg.norm(new_pt)
                if new_norm > 1e-12:
                    points[i] = new_pt / new_norm
                else:
                    points[i] = points[i] + step_size * n_unit
                    points[i] = points[i] / np.linalg.norm(points[i])
        
        # Ensure no coincident points
        for i in range(N):
            for j in range(i + 1, N):
                dot = np.dot(points[i], points[j])
                if dot > 0.9999999:
                    diff = points[i] - points[j]
                    dn = np.linalg.norm(diff)
                    if dn < 1e-12:
                        diff = np.array([1.0, 0.0, 0.0])
                        dn = 1.0
                    diff /= dn
                    points[i] = points[i] + 0.01 * diff
                    points[j] = points[j] - 0.01 * diff
                    points[i] /= np.linalg.norm(points[i])
                    points[j] /= np.linalg.norm(points[j])
    
    return points


def construct():
    start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    best_points = _fibonacci_sphere(N, offset=0.5)
    best_radius = _covering_radius_exact(best_points)
    
    # Multiple random starts with facet-based refinement
    n_starts = 8
    for s in range(n_starts):
        if time.time() > DEADLINE - 15:
            break
        
        if s == 0:
            points = best_points.copy()
        else:
            noise = np.random.normal(0, 0.3, size=(N, 3))
            points = best_points + noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            points = points / norms
        
        # Refine using facet-based method
        points = _refine_facets(points, n_iters=150, step_size=0.02)
        
        # Compute exact radius
        radius = _covering_radius_exact(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Final refinement with smaller step
    if time.time() < DEADLINE - 10:
        best_points = _refine_facets(best_points, n_iters=200, step_size=0.005)
        final_radius = _covering_radius_exact(best_points)
        if final_radius < best_radius:
            best_radius = final_radius
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_points = best_points / norms
    
    return best_points
