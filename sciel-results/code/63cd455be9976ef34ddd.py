"""Rewritten program: differentiable rigid-body mechanics approach for circle packing."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed and select n points closest to center."""
    # Start with a reasonable spacing
    # For n circles in unit disk, area per circle ~ pi/n, so radius ~ sqrt(1/n)
    # Hexagonal spacing ~ 2*r
    r_est = np.sqrt(1.0 / (n * 1.1))
    spacing = 2.0 * r_est * 1.05  # slight overestimate to get enough points
    
    # Generate hexagonal lattice points
    points = []
    k_max = int(1.0 / spacing) + 3
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    # Keep points that fit (with some margin)
    margin = spacing * 0.1
    mask = radii <= 1.0 - margin
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < n:
        # If not enough, reduce spacing
        for _ in range(20):
            spacing *= 0.95
            points = []
            k_max = int(1.0 / spacing) + 3
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    points.append([x, y])
            points = np.array(points)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii <= 1.0 - margin
            points = points[mask]
            radii = radii[mask]
            if len(points) >= n:
                break
    
    # Sort by distance from center and take first n
    idx = np.argsort(radii)
    return points[idx[:n]]


def _compute_score(centers):
    """Compute the packing radius for given centers."""
    n = len(centers)
    # Minimum distance between any pair / 2
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Minimum distance to boundary
    min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _spring_energy(centers, target_dist):
    """
    Compute energy for a system where circles are connected by stiff springs
    to maintain a fixed target diameter (distance = 2*r).
    We want to maximize the minimum clearance, so we use a penalty for violations.
    """
    n = len(centers)
    energy = 0.0
    
    # Pairwise spring energy: penalize if distance < target_dist
    # Use a smooth approximation
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < target_dist:
                # Stiff spring: quadratic penalty
                violation = target_dist - d
                energy += 1000.0 * violation ** 2
    
    # Boundary penalty: penalize if circle extends outside unit circle
    for i in range(n):
        r_i = np.linalg.norm(centers[i])
        if r_i > 1.0 - target_dist / 2.0:
            violation = r_i - (1.0 - target_dist / 2.0)
            energy += 1000.0 * violation ** 2
    
    return energy


def _optimize_packing(centers, target_dist, max_iter=1000):
    """
    Optimize the packing using gradient descent to maximize the minimum clearance.
    We reformulate: for a fixed target_dist, find positions that satisfy constraints.
    Then we can binary search on target_dist.
    """
    n = len(centers)
    
    def objective(x):
        """Minimize the maximum violation (or a smooth approximation)."""
        c = x.reshape(-1, 2)
        
        # Compute all pairwise distances
        min_pair = np.inf
        min_boundary = np.inf
        
        # To make it differentiable, we use a soft-min approach
        # But for simplicity, let's use a penalty-based approach
        
        total_penalty = 0.0
        
        # Pairwise constraints
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((c[i] - c[j]) ** 2) + 1e-10)
                if d < target_dist:
                    total_penalty += (target_dist - d) ** 2
        
        # Boundary constraints
        for i in range(n):
            r_i = np.sqrt(np.sum(c[i] ** 2) + 1e-10)
            if r_i > 1.0 - target_dist / 2.0:
                total_penalty += (r_i - (1.0 - target_dist / 2.0)) ** 2
        
        return total_penalty
    
    def gradient(x):
        """Compute gradient of the objective."""
        c = x.reshape(-1, 2).copy()
        grad = np.zeros_like(x)
        
        # Pairwise gradients
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                d = np.sqrt(np.sum(diff ** 2) + 1e-10)
                if d < target_dist:
                    # Gradient of (target_dist - d)^2 with respect to c[i] and c[j]
                    # d/dd (target - d)^2 = -2(target - d)
                    # d(d)/d(c[i]) = -diff/d
                    factor = -2.0 * (target_dist - d) * (-diff / d)
                    grad[2*i:2*i+2] += factor
                    grad[2*j:2*j+2] -= factor
        
        # Boundary gradients
        for i in range(n):
            c_i = c[i]
            r_i = np.sqrt(np.sum(c_i ** 2) + 1e-10)
            if r_i > 1.0 - target_dist / 2.0:
                # Gradient of (r_i - (1 - target/2))^2 with respect to c_i
                # = 2*(r_i - (1-target/2)) * c_i / r_i
                factor = 2.0 * (r_i - (1.0 - target_dist / 2.0)) * c_i / r_i
                grad[2*i:2*i+2] += factor
        
        return grad
    
    x0 = centers.flatten()
    
    # Use L-BFGS-B for optimization
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    return result.x.reshape(-1, 2)


def _binary_search_radius(centers, low, high, tol=1e-8, max_iter=30):
    """
    Binary search on the radius to find the maximum feasible packing radius.
    """
    best_centers = centers.copy()
    
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        
        # Try to find a valid configuration for radius mid
        optimized = _optimize_packing(centers, 2.0 * mid, max_iter=200)
        
        # Check if the optimized configuration is valid
        score = _compute_score(optimized)
        
        if score >= mid - tol:
            # Feasible, try larger radius
            low = mid
            best_centers = optimized.copy()
        else:
            # Not feasible, try smaller radius
            high = mid
        
        if high - low < tol:
            break
    
    return best_centers, low


def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    Uses a differentiable rigid-body mechanics approach with spring constraints
    and binary search on the radius.
    """
    # Generate initial hexagonal seed
    centers = _hexagonal_seed(N)
    
    # Estimate initial radius
    initial_score = _compute_score(centers)
    
    # Binary search on the radius
    low = initial_score * 0.9
    high = initial_score * 1.1
    
    # First, try to improve with the current estimate
    optimized, final_radius = _binary_search_radius(centers, low, high, tol=1e-7, max_iter=20)
    
    # Verify the result
    final_score = _compute_score(optimized)
    
    # If the score is worse than the initial, fall back to the initial
    if final_score < initial_score:
        optimized = centers
    
    # Ensure all points are within the unit circle and non-overlapping
    # Final cleanup: clip any points that might be slightly outside
    for i in range(len(optimized)):
        r = np.linalg.norm(optimized[i])
        if r > 1.0:
            # Scale back to the boundary
            scale = (1.0 - 1e-10) / r
            optimized[i] *= scale
    
    return optimized
# EVOLVE-BLOCK-END
