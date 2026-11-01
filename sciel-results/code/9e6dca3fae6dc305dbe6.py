"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Initialize with a good starting configuration
    # Use a combination of Fibonacci spiral and some perturbation
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    def covering_radius(pts):
        """Compute the covering radius in radians."""
        # The covering radius is the maximum over all points on the sphere 
        # of the distance to the nearest point in pts.
        # This is equivalent to the largest circumradius of the Delaunay triangulation facets.
        # For efficiency, we can use the fact that the covering radius is the maximum
        # of the circumradii of the triangles in the convex hull.
        
        # A simpler approach: sample many points on the sphere and find the max distance
        # to the nearest point in pts.
        
        # Use a grid of test points
        # For better accuracy, use the convex hull approach or dense sampling
        
        # Dense sampling approach
        n_samples = 50000
        # Generate random points on sphere
        rng = np.random.RandomState(42)
        test_pts = rng.randn(n_samples, 3)
        test_pts /= np.linalg.norm(test_pts, axis=1, keepdims=True)
        
        # Also add regular grid points
        # Fibonacci grid
        j = np.arange(20000) + 0.5
        z_grid = 1.0 - 2.0 * j / 20000
        phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * j
        r_grid = np.sqrt(np.maximum(0, 1.0 - z_grid * z_grid))
        grid_pts = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
        
        all_test = np.vstack([test_pts, grid_pts])
        
        # Compute distances to nearest point
        # Use cKDTree for efficiency
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        dists, _ = tree.query(all_test)
        # dists are Euclidean distances, convert to angular
        # angular distance = 2 * arcsin(d / 2) for unit sphere
        angular_dists = 2.0 * np.arcsin(np.minimum(1.0, dists / 2.0))
        
        return np.max(angular_dists)
    
    def repulsion_energy(pts):
        """Compute a repulsion-based energy to spread points."""
        # Pairwise angular distances
        dot = pts @ pts.T
        np.fill_diagonal(dot, 2.0)  # avoid self
        angular_dists = np.arccos(np.clip(dot, -1.0, 1.0))
        # Energy: sum of inverse powers of distances (repulsion)
        # Use a soft version to avoid singularities
        energy = np.sum(1.0 / (angular_dists + 1e-6) ** 2)
        return energy
    
    def gradient_descent(pts, max_iters=5000, lr_init=0.01):
        """Gradient descent to minimize covering radius via repulsion + targeted reduction."""
        best_pts = pts.copy()
        best_cr = covering_radius(best_pts)
        lr = lr_init
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute gradients for repulsion
            dot = pts @ pts.T
            np.fill_diagonal(dot, 2.0)
            angular_dists = np.arccos(np.clip(dot, -1.0, 1.0))
            
            # Repulsion gradient: move points apart
            # d/dx_i (1/d^2) = -2/d^3 * d/dx_i(d)
            # Simplified: use pairwise forces
            grad = np.zeros_like(pts)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = angular_dists[i, j]
                    if d < 1e-6:
                        continue
                    # Force direction: away from j
                    # Tangential component of (pts[i] - pts[j])
                    diff = pts[i] - pts[j]
                    # Project onto tangent plane at pts[i]
                    tangent = diff - np.dot(diff, pts[i]) * pts[i]
                    # Repulsion: push away
                    grad[i] += tangent / (d ** 3 + 1e-6)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad /= grad_norm
            
            # Update
            new_pts = pts - lr * grad
            # Re-normalize to sphere
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            # Check improvement
            new_cr = covering_radius(new_pts)
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = new_pts.copy()
                # Decrease learning rate slowly
                lr *= 0.999
            else:
                # Increase learning rate slightly or restart
                lr = max(0.001, lr * 0.95)
            
            pts = new_pts
        
        return best_pts, best_cr
    
    # Run gradient descent
    points, best_cr = gradient_descent(points)
    
    # Final normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
