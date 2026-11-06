"""Rewritten program: Fibonacci disk initialization + repulsion-based relaxation."""
import numpy as np
import time

N = 300


def construct():
    """
    Construct N=300 circle centers in the unit disk to maximize the minimum
    distance between centers and from the boundary.
    """
    start_time = time.time()
    deadline = 150.0  # wall-clock limit with safety margin

    # --- Initialization: Fibonacci disk distribution ---
    def fibonacci_disk(n):
        """Generate n points uniformly distributed in the unit disk using Fibonacci lattice."""
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        indices = np.arange(n)
        # Use sqrt for uniform area distribution
        r = np.sqrt((indices + 0.5) / n)
        theta = golden_angle * indices
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

    points = fibonacci_disk(N).astype(np.float64)

    # --- Compute current radius ---
    def compute_radius(pts):
        """Compute the achievable radius for a given set of points."""
        # Minimum distance between pairs / 2
        from scipy.spatial.distance import pdist
        min_pair = pdist(pts)
        min_pair = np.min(min_pair) / 2.0
        # Minimum distance to boundary
        dist_to_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dist_to_boundary)
        return min(min_pair, min_boundary)

    current_radius = compute_radius(points)

    # --- Repulsion-based relaxation ---
    # We use a gradient ascent approach on the minimum radius.
    # Since min is non-smooth, we use a soft approximation or subgradient.
    # Here we use a simple iterative approach: move points apart from their
    # nearest neighbors and away from the boundary.

    def soft_min_penalty(pts, temperature=0.001):
        """
        Compute a soft minimum of the constraints using log-sum-exp.
        Returns (value, gradient).
        """
        from scipy.spatial.distance import pdist
        dists = pdist(pts)
        # Soft min of distances: -log-sum-exp(-dists/T) / log(n_pairs)
        # Actually, we want to maximize min distance.
        # Let's use: f = -soft_min(-dists) = soft_max(dists)
        # soft_max(dists) = T * log(sum(exp(dists/T)))
        # But this is dominated by the max, not min.
        
        # Better: use a penalty for overlaps.
        # Penalty = sum of max(0, r_overlap - d_ij) for all pairs
        # and sum of max(0, r_boundary - (1 - |c_i|))
        
        # Actually, let's just do direct gradient ascent on the radius.
        # The radius is determined by the tightest constraint.
        # We can identify the critical pairs and push them apart.
        
        # For simplicity, use a repulsion force model:
        # Force on each point is sum of repulsion from all other points
        # weighted by inverse square distance, plus boundary repulsion.
        
        # Let's compute forces.
        n = len(pts)
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random small perturbation
                    diff = np.random.randn(2) * 1e-10
                    dist = np.linalg.norm(diff)
                # Repulsion strength: stronger when closer
                # Use 1/dist^2 or similar
                force_mag = 1.0 / (dist * dist + 1e-6)
                force = force_mag * diff / dist
                forces[i] += force
                forces[j] -= force
        
        # Boundary repulsion: push points inward if too close to edge
        for i in range(n):
            norm = np.linalg.norm(pts[i])
            if norm > 0.95:  # Only apply if close to boundary
                # Force pointing inward
                force_mag = (norm - 0.95) ** 2 * 10.0
                force = -force_mag * pts[i] / norm
                forces[i] += force
        
        return forces

    # --- Iterative relaxation ---
    lr = 0.01  # Learning rate
    best_points = points.copy()
    best_radius = current_radius

    max_iters = 10000
    for it in range(max_iters):
        if time.time() - start_time > deadline - 10.0:
            break
        
        # Compute forces
        forces = soft_min_penalty(points)
        
        # Update positions
        points += lr * forces
        
        # Clamp to unit disk (with small margin)
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            points[mask] *= 0.999 / norms[mask]
        
        # Compute new radius
        new_radius = compute_radius(points)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_points = points.copy()
        
        # Adaptive learning rate: decrease if no improvement
        if it % 100 == 0:
            lr *= 0.999
    
    # --- Final polish: local refinement ---
    # Use a more careful approach to polish the solution
    # Identify the critical constraints and do a local solve
    
    # For now, return the best found
    if best_radius > current_radius:
        return best_points
    else:
        return points


if __name__ == "__main__":
    result = construct()
    print(f"Shape: {result.shape}")
    from scipy.spatial.distance import pdist
    min_pair = np.min(pdist(result)) / 2.0
    min_boundary = np.min(1.0 - np.linalg.norm(result, axis=1))
    print(f"Radius: {min(min_pair, min_boundary)}")
