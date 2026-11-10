"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    x = centers[:, 0]
    y = centers[:, 1]
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate a hexagonal lattice packing inside the unit circle."""
    best_points = None
    best_radius = 0.0
    
    for spacing in np.linspace(0.18, 0.26, 40):
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            # Select the N points closest to the center
            points = points[np.argsort(radius, kind='stable')[:N]]
            r = _compute_radius(points)
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    # Fallback: evenly spaced on a circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

def _soft_min(values, temperature=0.01):
    """Soft minimum using log-sum-exp."""
    min_val = np.min(values)
    return min_val - temperature * np.log(np.sum(np.exp((values - min_val) / temperature)))

def _smooth_objective(centers, temperature=0.01):
    """
    Smooth surrogate for the packing objective.
    We want to maximize r = min(min_pairwise/2, min_boundary).
    Equivalent to maximizing the minimum of all constraints.
    """
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Boundary constraints: 1 - |c_i|
    dists_boundary = 1.0 - np.hypot(x, y)
    
    # Pairwise constraints: |c_i - c_j| / 2
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dists = np.hypot(dx, dy) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Combine all constraints into a single vector
    # We want to maximize the minimum of all these values
    all_constraints = np.concatenate([dists_boundary, dists[np.triu_indices(N, k=1)]])
    
    # Use soft-min to get a smooth approximation of the minimum
    return _soft_min(all_constraints, temperature)

def _gradient_ascent(centers, max_iters=2000, lr=0.001, temp_start=0.05, temp_end=0.001):
    """
    Gradient ascent on the smooth surrogate objective.
    """
    current_centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    for it in range(max_iters):
        # Anneal temperature
        temp = temp_start * (temp_end / temp_start) ** (it / max_iters)
        
        # Compute gradient numerically (central differences)
        grad = np.zeros_like(current_centers)
        eps = 1e-4
        obj_current = _smooth_objective(current_centers, temp)
        
        for i in range(N):
            for d in range(2):
                perturbed = current_centers.copy()
                perturbed[i, d] += eps
                obj_perturbed = _smooth_objective(perturbed, temp)
                grad[i, d] = (obj_perturbed - obj_current) / eps
        
        # Update
        new_centers = current_centers + lr * grad
        
        # Check if improvement in hard objective
        new_radius = _compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
        
        # Accept or reject based on simulated annealing
        obj_new = _smooth_objective(new_centers, temp)
        delta_obj = obj_new - obj_current
        
        if delta_obj > 0 or np.random.random() < np.exp(delta_obj / temp):
            current_centers = new_centers
            obj_current = obj_new
        
        # Decay learning rate
        lr *= 0.999
    
    return best_centers, best_radius

def _local_optimization(centers, max_iters=5000):
    """
    Local optimization using a combination of gradient ascent and random restarts.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Phase 1: Gradient ascent on smooth surrogate
    centers1, radius1 = _gradient_ascent(centers, max_iters=1000, lr=0.002, temp_start=0.1, temp_end=0.01)
    if radius1 > best_radius:
        best_radius = radius1
        best_centers = centers1
    
    # Phase 2: Fine-tune with smaller steps
    centers2, radius2 = _gradient_ascent(best_centers, max_iters=1000, lr=0.0005, temp_start=0.02, temp_end=0.001)
    if radius2 > best_radius:
        best_radius = radius2
        best_centers = centers2
    
    # Phase 3: Random perturbations and re-optimize
    for restart in range(5):
        perturbed = best_centers + np.random.normal(0, 0.005, (N, 2))
        centers3, radius3 = _gradient_ascent(perturbed, max_iters=500, lr=0.001, temp_start=0.05, temp_end=0.005)
        if radius3 > best_radius:
            best_radius = radius3
            best_centers = centers3
    
    return best_centers, best_radius

def construct():
    """Construct the packing of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Run local optimization
    improved_centers, improved_radius = _local_optimization(best_centers)
    if improved_radius > best_radius:
        best_radius = improved_radius
        best_centers = improved_centers
    
    # If time permits, try more restarts
    while time.time() < deadline - 20:
        # Perturb the best solution
        perturbed = best_centers + np.random.normal(0, 0.01, (N, 2))
        # Ensure perturbed points are still inside the circle
        for i in range(N):
            r = np.hypot(perturbed[i, 0], perturbed[i, 1])
            if r > 1.0:
                perturbed[i] = perturbed[i] / r * 0.99
        
        centers_opt, radius_opt = _local_optimization(perturbed, max_iters=2000)
        if radius_opt > best_radius:
            best_radius = radius_opt
            best_centers = centers_opt
    
    # Final validation
    final_radius = _compute_radius(best_centers)
    if final_radius > 0:
        return best_centers
    
    # Fallback if something went wrong
    return _hexagonal_packing()
# EVOLVE-BLOCK-END
