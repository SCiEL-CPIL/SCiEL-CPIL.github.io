import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        # Vectorized circumcenter calculation
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms[norms < 1e-12] = 1e-12
        c = cross / norms
        
        # Ensure correct orientation (dot with p1 > 0)
        dots = np.sum(c * p1, axis=1)
        flip = dots < 0
        c[flip] *= -1
        
        # Calculate distances
        dots_c = np.sum(c * p1, axis=1)
        dots_c = np.clip(dots_c, -1.0, 1.0)
        dists = np.arccos(dots_c)
        
        return np.degrees(np.max(dists))
    except Exception:
        return np.pi / 2

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin before 120s kill
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Compute initial hull to get topology
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        # Fallback if hull fails (shouldn't happen with Fib spiral)
        return best_points
    
    num_facets = len(facets)
    
    # Pre-compute indices for vectorized accumulation if possible, 
    # but simple loop over facets is fast enough for N=130 (~260 facets)
    
    iteration = 0
    while time.time() < deadline:
        try:
            p1 = points[facets[:, 0]]
            p2 = points[facets[:, 1]]
            p3 = points[facets[:, 2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norms = np.linalg.norm(cross, axis=1, keepdims=True)
            
            # Avoid division by zero
            valid_norms = norms > 1e-12
            norms_safe = np.where(valid_norms, norms, 1.0)
            c = cross / norms_safe
            
            # Zero out invalid circumcenters
            c[~valid_norms] = 0.0
            
            # Fix orientation
            dots = np.sum(c * p1, axis=1)
            flip = dots < 0
            c[flip] *= -1
            
            # Accumulate circumcenters for each vertex
            sums = np.zeros((N, 3))
            # Using np.add.at is safe for repeated indices
            # facets shape (F, 3)
            # We want to add c[i] to sums[facets[i,0]], sums[facets[i,1]], sums[facets[i,2]]
            
            # Vectorized approach using repeat
            # Create an array of vertex indices and corresponding circumcenter vectors
            # indices: [f0_0, f0_1, f0_2, f1_0, ...]
            # c_repeated: [c0, c0, c0, c1, ...]
            
            # This is faster than a python loop for moderate N
            # But for N=130, a loop is also fine. Let's use the loop for clarity and safety against memory spikes, 
            # or the vectorized one if we want max speed.
            # Let's stick to the vectorized one to be safe on speed.
            
            # Actually, np.add.at is O(N) effectively but has overhead.
            # Let's use the explicit loop as it's very fast for 260 iterations.
            for idx in range(num_facets):
                tri = facets[idx]
                c_vec = c[idx]
                sums[tri[0]] += c_vec
                sums[tri[1]] += c_vec
                sums[tri[2]] += c_vec
            
            # Normalize sums to get new positions
            norms_sums = np.linalg.norm(sums, axis=1, keepdims=True)
            # If a vertex is isolated (shouldn't happen in a valid hull), norm is 0.
            # Keep old position if norm is too small.
            mask = norms_sums > 1e-12
            new_points = points.copy()
            new_points[mask] = sums[mask] / norms_sums[mask]
            
            # Optional: Damping? Lloyd's usually converges well.
            points = new_points
            
            # Periodic evaluation
            # Every 50 iterations or if time allows
            if iteration % 50 == 0 and (time.time() - start_time) < deadline - 10.0:
                 current_radius = _compute_covering_radius(points)
                 if current_radius < best_radius:
                     best_radius = current_radius
                     best_points = points.copy()
            
            iteration += 1
             
        except Exception:
            # If something goes wrong, break and return best
            break
            
    # Final polish: compute the true radius of the best points found
    # The best_points might have been evaluated with a slightly different hull topology 
    # if we updated it, but we didn't update topology. 
    # However, _compute_covering_radius computes the *true* hull for the given points.
    # So best_radius is accurate for best_points.
    
    return best_points
