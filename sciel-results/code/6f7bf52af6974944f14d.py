"""Rewritten program: Stochastic Lloyd's Algorithm and Coordinate Repulsion."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n):
    """Generate n random points on the unit sphere."""
    # Using normal distribution and normalizing
    v = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid zero norms (extremely unlikely)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _stochastic_lloyd(pts, n_samples=5000, n_iters=50):
    """
    Stochastic Lloyd's Algorithm for sphere.
    Approximates Voronoi centroids by sampling.
    """
    pts = pts.copy()
    n = len(pts)
    
    for _ in range(n_iters):
        # Generate random samples on the sphere
        samples = _random_sphere(n_samples)
        
        # Assign each sample to the nearest point in pts
        # Compute distances: (n_samples, n)
        # d_{i,j} = ||samples_i - pts_j||^2
        # = ||s_i||^2 + ||p_j||^2 - 2 s_i . p_j
        # Since all are on unit sphere: 2 - 2 s_i . p_j
        # Minimizing distance is equivalent to maximizing dot product
        dots = samples @ pts.T  # (n_samples, n)
        nearest = np.argmax(dots, axis=1)  # (n_samples,)
        
        # Accumulate sums for each point
        # sum_pts[i] = sum of samples assigned to i
        sum_pts = np.zeros((n, 3))
        counts = np.zeros(n)
        
        # Efficient accumulation using np.add.at or bincount logic
        # Since N=100 is small, we can loop or use advanced indexing
        # samples[nearest] gives the samples grouped by their assigned point
        # We want to sum them by index
        np.add.at(sum_pts, nearest, samples)
        np.add.at(counts, nearest, 1)
        
        # Update points to centroids
        for i in range(n):
            if counts[i] > 0:
                new_pt = sum_pts[i] / counts[i]
                norm = np.linalg.norm(new_pt)
                if norm > 1e-12:
                    pts[i] = new_pt / norm
            else:
                # If a point has no samples, it's likely isolated or in a tiny cell.
                # Keep it or move it slightly randomly to help it find a cell?
                # For now, keep it.
                pass
                
        # Renormalize all
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.where(norms < 1e-12, 1.0, norms)
        
    return pts


def _coordinate_repulsion(pts, n_iters=500, alpha=0.01):
    """
    Coordinate-wise repulsion (Gauss-Seidel style).
    Distinct from simultaneous gradient descent.
    """
    pts = pts.copy()
    n = len(pts)
    
    for _ in range(n_iters):
        # Random order for better mixing
        order = np.random.permutation(n)
        for i in order:
            # Compute repulsion force on point i
            # F_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
            # diff = pts[i] - pts[j]
            # dist_sq = ||diff||^2
            
            diffs = pts[i] - pts  # (n, 3)
            dist_sq = np.sum(diffs * diffs, axis=1)  # (n,)
            
            # Avoid division by zero for j=i
            dist_sq[i] = 1.0 
            inv_dist_sq = 1.0 / dist_sq
            inv_dist_sq[i] = 0.0
            
            force = np.sum(diffs * inv_dist_sq[:, np.newaxis], axis=0)
            
            # Project force to tangent space at pts[i]
            # tangent_force = force - (force . pts[i]) * pts[i]
            dot = np.dot(force, pts[i])
            tangent_force = force - dot * pts[i]
            
            # Move
            pts[i] = pts[i] + alpha * tangent_force
            
            # Renormalize
            norm = np.linalg.norm(pts[i])
            if norm > 1e-12:
                pts[i] = pts[i] / norm
    
    return pts


def construct():
    # Start from Fibonacci spiral
    pts = _fibonacci_sphere(N).astype(np.float64)
    
    # Phase 1: Global structure via Stochastic Lloyd's
    # This helps escape bad local minima and establishes good Voronoi cells
    pts = _stochastic_lloyd(pts, n_samples=10000, n_iters=30)
    
    # Phase 2: Local refinement via Coordinate Repulsion
    # This is a local search method, distinct from gradient descent
    pts = _coordinate_repulsion(pts, n_iters=1000, alpha=0.05)
    
    # Final cleanup
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Verify
    assert np.all(np.isfinite(pts))
    assert pts.shape == (N, 3)
    
    return pts
# EVOLVE-BLOCK-END
