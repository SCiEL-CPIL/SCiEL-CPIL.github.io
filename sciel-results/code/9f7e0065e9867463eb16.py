"""
Strategy: Stochastic Local Search with Simulated Annealing.
Instead of deterministic gradient descent on the max facet radius, we use
Simulated Annealing. This allows the algorithm to accept moves that
temporarily increase the covering radius, enabling it to escape local minima
where the deterministic gradient method gets stuck.

The objective is to minimize the covering radius.
Moves: Randomly perturb a small number of points (or all points slightly).
Acceptance: Metropolis criterion.
"""

import numpy as np
import time
import random
from scipy.spatial import ConvexHull

N = 130

def _compute_circumcenters(points, facets):
    """
    Compute circumcenters and circumradii for all facets.
    """
    n_facets = facets.shape[0]
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p1 - p2
    v2 = p1 - p3
    c_dir = np.cross(v1, v2)
    
    norms = np.linalg.norm(c_dir, axis=1)
    valid = norms > 1e-10
    
    if np.any(valid):
        centers[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        # Orient centers
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(centers * centroids, axis=1)
        centers[dots < 0] *= -1
        
        # Compute circumradius
        cos_theta = np.sum(centers[valid] * p1[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii[valid] = np.arccos(cos_theta)
    
    return centers, radii

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        # Check for coincident points or invalid configurations
        # If points are too close, hull might fail or be degenerate
        hull = ConvexHull(points)
        facets = hull.simplices
        centers, radii = _compute_circumcenters(points, facets)
        
        if np.any(radii == 0) and len(radii[radii>0]) == 0:
             return 180.0
             
        return np.max(radii) * 180.0 / np.pi
    except Exception:
        return 180.0

def _random_perturbation(points, magnitude):
    """
    Generate a new candidate configuration by randomly perturbing the current points.
    We perturb all points by a small random vector to maintain global connectivity,
    but scale the perturbation by 'magnitude'.
    """
    # Generate random unit vectors
    random_vecs = np.random.randn(N, 3)
    random_vecs /= np.linalg.norm(random_vecs, axis=1, keepdims=True)
    
    # Add perturbation
    new_points = points + magnitude * random_vecs
    
    # Normalize to sphere
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    new_points = new_points / norms
    
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_points = points.copy()
    current_radius = _covering_radius(current_points)
    
    best_points = current_points.copy()
    best_radius = current_radius
    
    # Simulated Annealing parameters
    # Temperature starts high enough to accept large jumps, cools down
    T_start = 5.0 # degrees
    T_end = 0.01
    # We don't know the exact number of steps, so we use time-based cooling
    # or a fixed number of steps if time allows.
    
    # Let's estimate steps per second to plan the schedule
    # Or just use a fixed schedule that fits in ~140s.
    # _covering_radius takes ~1-2ms? Let's assume ~5ms per eval including hull.
    # 140s / 0.005s = 28000 steps.
    
    max_steps = 30000
    # If we run out of time, we stop.
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Cooling schedule: Exponential decay
        # T = T_start * (T_end/T_start)^(step/max_steps)
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Perturbation magnitude: Start with larger jumps, shrink later
        # Scale with T? Or independent?
        # Let's scale magnitude with T to explore more when hot.
        # Base magnitude ~ 0.1 rad (5 deg).
        magnitude = 0.05 + 0.1 * (T / T_start)
        
        new_points = _random_perturbation(current_points, magnitude)
        new_radius = _covering_radius(new_points)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_points = new_points
            current_radius = new_radius
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            # If rejected, keep current. 
            # Optionally, if we are stuck for a long time, we could restart?
            # No, SA handles this via temperature.
            pass
            
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
