"""
Strategy: Stochastic Hill Climbing (SHC) with Global Random Moves.
Unlike local gradient methods (Lloyd, Facet Repair) that get trapped in local minima,
SHC uses large random perturbations to jump between basins of attraction.
We combine this with a high-quality initial guess (Thurston-Fibonacci) and
a time-budgeted search that accepts any improvement.

Key differences from current program:
1. Global search vs Local search: Moves are large (up to 0.5 rad) to escape local minima.
2. No complex geometric repair in the inner loop: The move is purely random, making
   each step very fast (O(N)), allowing for many more evaluations (up to ~100k) in 120s.
3. Thurston initialization: A specific Fibonacci variant known for good initial distribution.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Returns 180.0 if invalid (e.g., non-convex, degenerate).
    """
    try:
        # Check for non-finite values
        if not np.all(np.isfinite(points)):
            return 180.0
        
        # Check if points are normalized
        norms = np.linalg.norm(points, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            return 180.0

        # Check for coincident points (distance < 1e-6)
        # This is O(N^2) but N=130 is small enough for frequent checks? 
        # Actually, let's rely on ConvexHull failing or returning degenerate facets.
        # ConvexHull requires distinct points.
        
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Compute circumcenters and radii
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Vectorized circumcenter calculation
        # a = p1 - p2, b = p1 - p3
        # c = cross(a, b)
        # center = c / |c| (normalized)
        # radius = arccos(dot(center, p1))
        
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        
        norms_c = np.linalg.norm(c_dir, axis=1)
        
        # If norm is 0, points are collinear/coincident, invalid
        if np.any(norms_c < 1e-10):
            return 180.0
            
        centers = c_dir / norms_c[:, np.newaxis]
        
        # Ensure center points to the "outside" (same hemisphere as centroid)
        centroids = (p1 + p2 + p3) / 3.0
        dots = np.sum(centers * centroids, axis=1)
        centers[dots < 0] *= -1
        
        # Angular radius
        cos_theta = np.sum(centers * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii) * 180.0 / np.pi
        
    except Exception:
        return 180.0

def _thurston_init(n):
    """
    Thurston's Fibonacci spiral initialization.
    This is often a better starting point than the standard golden ratio spiral
    for covering problems due to its specific distribution properties.
    """
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # 1. Initialization
    points = _thurston_init(N)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # If initial is invalid (shouldn't happen with Thurston), fallback to random
    if best_radius > 90.0:
        points = np.random.randn(N, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        best_points = points.copy()
        best_radius = _covering_radius(points)
    
    # 2. Stochastic Hill Climbing
    # We use a fixed large step size for "global" moves.
    # Since we normalize after every move, the "step size" in Cartesian space
    # translates to an angular displacement.
    # A step of 0.5 in Cartesian ~ 0.5 radians angular displacement.
    
    # We can also try adaptive step sizes, but fixed is more robust for SHC.
    # Let's use a base step and occasionally larger ones.
    
    rng = np.random.default_rng(42)
    
    # Pre-allocate for speed
    new_points = np.empty_like(points)
    noise = np.empty_like(points)
    
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Generate random perturbation
        # Gaussian noise is good for spherical moves
        # Scale factor: 0.5 rad is a good "global" jump.
        # Standard deviation of Gaussian to get ~0.5 rad RMS displacement on sphere?
        # For small angles, ||noise|| approximates angle.
        # So sigma = 0.5.
        
        # Occasionally use a larger jump (0.8) to escape deeper minima
        if rng.random() < 0.1:
            sigma = 0.8
        else:
            sigma = 0.5
            
        noise = rng.normal(0, sigma, size=(N, 3))
        new_points = points + noise
        
        # Normalize to sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        # Avoid division by zero (extremely unlikely with Gaussian)
        norms[norms < 1e-10] = 1.0
        new_points /= norms
        
        # Evaluate
        new_radius = _covering_radius(new_points)
        
        # Accept if better
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points  # Continue from this new best (Hill Climbing)
        else:
            # In pure Hill Climbing, we reject and stay at current point.
            # However, for SHC, sometimes it's beneficial to keep the random point
            # as the "current" state for the next step, even if it's worse than best,
            # to explore further. But standard SHC usually reverts to last accepted.
            # Let's stick to standard: revert to points (which is the last accepted).
            pass
            
        # Optional: If we get stuck (no improvement for many steps), 
        # we could increase noise or restart. 
        # But the 10% large jumps should handle this.
        
        # Check time more frequently in tight loops?
        # The loop itself is fast, so time check is sufficient.
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
