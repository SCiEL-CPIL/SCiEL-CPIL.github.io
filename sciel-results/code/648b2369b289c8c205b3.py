"""Rewritten program for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """Construct centres of 300 equal circles inside the unit circle.
    
    Uses a multi-stage approach:
    1. Initialize with a spiral/hexagonal hybrid pattern
    2. Optimize using gradient-free local search (Nelder-Mead on batches)
    3. Refine with simulated annealing-like perturbations
    """
    np.random.seed(42)
    random.seed(42)
    
    # Step 1: Create initial configuration using a well-distributed pattern
    # Use a phyllotaxis-inspired spiral with some hexagonal structure
    def make_initial_config(n_points):
        """Create initial points using a combination of rings and spiral."""
        points = np.zeros((n_points, 2))
        
        # Place points in concentric rings with alternating offsets
        # This gives a more uniform distribution than pure hexagonal lattice
        n_rings = int(np.sqrt(n_points)) + 2
        
        idx = 0
        for ring in range(n_rings):
            if ring == 0:
                # Center point
                points[idx] = [0.0, 0.0]
                idx += 1
            else:
                # Number of points in this ring
                # Use approximately 6*ring points for hexagonal-like density
                n_in_ring = min(6 * ring, n_points - idx)
                if n_in_ring <= 0:
                    break
                
                # Radius of this ring
                r_ring = ring * 0.15  # Adjust spacing
                # Scale to fit within unit circle
                max_r = 1.0 - 0.05
                r_ring = min(r_ring, max_r)
                
                for i in range(n_in_ring):
                    if idx >= n_points:
                        break
                    angle = 2 * np.pi * i / n_in_ring + (ring % 2) * np.pi / n_in_ring
                    points[idx] = [r_ring * np.cos(angle), r_ring * np.sin(angle)]
                    idx += 1
        
        # Fill remaining points with random positions inside the circle
        while idx < n_points:
            r = np.sqrt(np.random.random()) * 0.95
            theta = np.random.random() * 2 * np.pi
            points[idx] = [r * np.cos(theta), r * np.sin(theta)]
            idx += 1
        
        return points
    
    def compute_radius(points):
        """Compute the maximum valid radius for given points."""
        if len(points) < 2:
            return 0.0
        
        # Minimum distance between any two points
        # Use chunked computation for memory efficiency
        min_dist = np.inf
        n = len(points)
        chunk_size = 100
        
        for i in range(0, n, chunk_size):
            for j in range(i + 1, n, chunk_size):
                pi = points[i:i+chunk_size]
                pj = points[j:j+chunk_size]
                if len(pi) == 0 or len(pj) == 0:
                    continue
                # Compute pairwise distances
                diff = pi[:, np.newaxis, :] - pj[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                min_dist = min(min_dist, np.min(dists))
        
        # Minimum distance to boundary
        r_to_boundary = np.min(1.0 - np.hypot(points[:, 0], points[:, 1]))
        
        return min(min_dist / 2.0, r_to_boundary)
    
    def objective(points):
        """Objective function to minimize (negative radius)."""
        # Check validity
        if not np.all(np.isfinite(points)):
            return 1e10
        if np.any(np.hypot(points[:, 0], points[:, 1]) > 1.0):
            # Penalty for points outside
            violations = np.sum(np.maximum(0, np.hypot(points[:, 0], points[:, 1]) - 1.0))
            return 1e10 + violations
        return -compute_radius(points)
    
    def local_search(points, n_iterations=500, step_size=0.01):
        """Simple local search with random perturbations."""
        best_points = points.copy()
        best_radius = compute_radius(best_points)
        current_radius = best_radius
        
        for it in range(n_iterations):
            # Create a perturbed version
            new_points = best_points.copy()
            # Perturb a random subset of points
            n_perturb = max(1, len(new_points) // 10)
            indices = np.random.choice(len(new_points), size=n_perturb, replace=False)
            noise = np.random.randn(n_perturb, 2) * step_size
            new_points[indices] += noise
            
            # Project back inside the circle if needed
            r = np.hypot(new_points[:, 0], new_points[:, 1])
            mask = r > 0.98
            if np.any(mask):
                new_points[mask] *= 0.98 / r[mask]
            
            new_radius = compute_radius(new_points)
            
            if new_radius > current_radius:
                current_radius = new_radius
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
            else:
                # Accept with some probability (simulated annealing-like)
                if np.random.random() < 0.1:
                    current_radius = new_radius
                    best_points = new_points.copy()
        
        return best_points, best_radius
    
    # Initialize
    points = make_initial_config(N)
    initial_radius = compute_radius(points)
    
    # Run local search with decreasing step sizes
    best_points = points.copy()
    best_radius = initial_radius
    
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001]
    for step in step_sizes:
        points, radius = local_search(points, n_iterations=300, step_size=step)
        if radius > best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Final refinement: try to push points that are close to the boundary inward
    # and separate close pairs
    for _ in range(100):
        r = np.hypot(best_points[:, 0], best_points[:, 1])
        # Push points near boundary slightly inward
        mask = r > 0.95
        if np.any(mask):
            best_points[mask] *= 0.99
        
        # Check for close pairs and separate them
        n = len(best_points)
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < best_radius:
                    # Push them apart
                    mid = (best_points[i] + best_points[j]) / 2
                    direction = (best_points[i] - best_points[j]) / (d + 1e-10)
                    push = (best_radius - d) / 2 + 0.001
                    best_points[i] += direction * push
                    best_points[j] -= direction * push
                    # Keep inside circle
                    for k in [i, j]:
                        r_k = np.hypot(best_points[k, 0], best_points[k, 1])
                        if r_k > 0.99:
                            best_points[k] *= 0.99 / r_k
                    improved = True
        
        if not improved:
            break
    
    final_radius = compute_radius(best_points)
    if final_radius > best_radius:
        best_points = best_points.copy()
        best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
