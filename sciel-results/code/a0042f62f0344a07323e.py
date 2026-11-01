"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math
import random

def _covering_radius(points):
    """Compute the covering radius (in radians) as the max circumradius of convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                continue
            C = C_plane / norm_C
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_r:
                max_r = R
        return max_r
    except Exception:
        return np.pi

def _local_polish(points, iterations=100):
    """Local optimization using circumcenter moves."""
    points = points.copy()
    for _ in range(iterations):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        facet_radii = []
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                facet_radii.append(0.0)
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                facet_radii.append(0.0)
                continue
            C = C_plane / norm_C
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            facet_radii.append(R)
        
        facet_radii = np.array(facet_radii)
        max_idx = np.argmax(facet_radii)
        max_r = facet_radii[max_idx]
        
        if max_r < 0.197:
            break
        
        simplex = hull.simplices[max_idx]
        idx1, idx2, idx3 = simplex
        
        p1, p2, p3 = points[idx1], points[idx2], points[idx3]
        v = p2 - p1
        w = p3 - p1
        vv = np.dot(v, v)
        ww = np.dot(w, w)
        vw = np.dot(v, w)
        det = vv * ww - vw * vw
        if abs(det) < 1e-12:
            break
        t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
        t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
        C_plane = p1 + t1 * v + t2 * w
        norm_C = np.linalg.norm(C_plane)
        if norm_C < 1e-12:
            break
        C = C_plane / norm_C
        
        step = 0.005
        for idx in [idx1, idx2, idx3]:
            p = points[idx]
            new_p = p + step * (C - p)
            new_p = new_p / np.linalg.norm(new_p)
            points[idx] = new_p
    
    return points

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Simulated Annealing
    T_start = 0.1
    T_end = 0.001
    n_iterations = 5000
    step_size = 0.03
    
    current_energy = _covering_radius(points)
    best_points = points.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    for i_iter in range(n_iterations):
        T = T_start * (T_end / T_start) ** (i_iter / n_iterations)
        
        idx = rng.integers(0, N)
        
        move_step = step_size * (T / T_start + 0.2)
        
        noise = rng.normal(size=3)
        noise /= np.linalg.norm(noise)
        
        new_points = points.copy()
        new_points[idx] = new_points[idx] + move_step * noise
        new_points[idx] /= np.linalg.norm(new_points[idx])
        
        new_energy = _covering_radius(new_points)
        
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        
    # Polish with local search
    final_points = _local_polish(best_points, iterations=200)
    final_energy = _covering_radius(final_points)
    
    if final_energy < best_energy:
        best_points = final_points
        best_energy = final_energy
        
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
# EVOLVE-BLOCK-END
