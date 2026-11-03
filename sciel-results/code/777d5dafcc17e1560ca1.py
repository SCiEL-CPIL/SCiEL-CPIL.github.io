import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
DEADLINE = 135.0  # Leave 15s margin before 150s kill

def to_sphere(params):
    """Convert raw params to unit vectors."""
    v = np.asarray(params).reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return v / norms

def compute_covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere."""
    points = np.asarray(points)
    points = to_sphere(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_circum = 0.0
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
        
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        if np.linalg.norm(circumcenter) < 1e-10:
            continue
        
        pole = circumcenter / np.linalg.norm(circumcenter)
        
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        max_circum = max(max_circum, circum_angular)
    
    return max_circum

def energy_function(points, beta=20.0):
    """Compute a smooth energy that encourages even distribution.
    
    Uses a repulsive potential based on pairwise distances. Minimizing this
    energy tends to maximize the minimum distance, which correlates with
    minimizing the covering radius.
    """
    points = np.asarray(points)
    points = to_sphere(points)
    n = len(points)
    
    # Compute all pairwise dot products
    dots = points @ points.T
    # Clip for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angular distances
    angles = np.arccos(dots)
    np.fill_diagonal(angles, np.pi)  # Self-distance is pi
    
    # Use a smooth max-like function on the negative distances
    # We want to maximize the minimum distance, so minimize -min(dist)
    # Using log-sum-exp on negative distances
    neg_angles = -angles
    neg_angles = np.where(np.eye(n, dtype=bool), -1e10, neg_angles)
    
    # For each point, compute the soft max of negative distances to others
    # This gives us a smooth approximation of -min_dist for each point
    m = np.max(neg_angles, axis=1)
    logsumexp = m + np.log(np.sum(np.exp((neg_angles - m[:, None]) * beta), axis=1)) / beta
    
    # The energy is the sum of these soft-min distances (we want to maximize them)
    # So we minimize the negative
    return -np.sum(logsumexp)

def energy_gradient(points, beta=20.0):
    """Compute gradient of the energy function via finite differences."""
    points = np.asarray(points)
    points = to_sphere(points)
    n = len(points)
    grad = np.zeros_like(points)
    eps = 1e-6
    
    for i in range(n):
        for j in range(3):
            perturbed_plus = points.copy()
            perturbed_plus[i, j] += eps
            perturbed_plus = to_sphere(perturbed_plus)
            f_plus = energy_function(perturbed_plus, beta)
            
            perturbed_minus = points.copy()
            perturbed_minus[i, j] -= eps
            perturbed_minus = to_sphere(perturbed_minus)
            f_minus = energy_function(perturbed_minus, beta)
            
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def simulated_annealing(initial_points, n_iterations=5000, t_start=0.1, t_end=0.001, rng=None):
    """Optimize points using simulated annealing with energy-based objective."""
    if rng is None:
        rng = np.random.RandomState()
    
    points = np.asarray(initial_points).copy()
    points = to_sphere(points)
    
    best_points = points.copy()
    best_cr = compute_covering_radius(points)
    best_energy = energy_function(points)
    
    current_energy = best_energy
    
    for it in range(n_iterations):
        if time.time() > DEADLINE - 10:
            break
        
        # Annealing schedule
        progress = it / n_iterations
        temperature = t_start * (t_end / t_start) ** progress
        
        # Generate a random perturbation
        perturbation = rng.randn(N, 3) * temperature
        new_points = to_sphere(points + perturbation)
        
        new_energy = energy_function(new_points)
        delta_e = new_energy - current_energy
        
        # Accept with probability exp(-delta_e / T)
        if delta_e < 0 or rng.rand() < np.exp(-delta_e / max(temperature, 1e-10)):
            points = new_points
            current_energy = new_energy
            
            # Periodically check actual covering radius
            if it % 100 == 0:
                cr = compute_covering_radius(points)
                if cr < best_cr:
                    best_cr = cr
                    best_points = points.copy()
        
    # Final check
    cr = compute_covering_radius(points)
    if cr < best_cr:
        best_cr = cr
        best_points = points.copy()
    
    return best_points, best_cr

def gradient_descent_with_backtracking(points, max_iters=200, lr_init=0.01, beta=20.0):
    """Gradient descent with backtracking line search on the energy function."""
    points = np.asarray(points).copy()
    points = to_sphere(points)
    
    best_points = points.copy()
    best_cr = compute_covering_radius(points)
    
    lr = lr_init
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
        
        # Compute gradient
        grad = energy_gradient(points, beta)
        
        # Backtracking line search
        alpha = lr
        current_energy = energy_function(points)
        
        for _ in range(20):
            new_points = to_sphere(points - alpha * grad)
            new_energy = energy_function(new_points)
            
            if new_energy < current_energy:
                break
            alpha *= 0.5
        
        points = new_points
        
        # Periodically check actual covering radius
        if it % 20 == 0:
            cr = compute_covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
        
        # Adapt learning rate
        if new_energy < current_energy - 1e-6:
            lr = min(lr * 1.1, 0.1)
        else:
            lr = max(lr * 0.8, 1e-4)
    
    # Final check
    cr = compute_covering_radius(points)
    if cr < best_cr:
        best_cr = cr
        best_points = points.copy()
    
    return best_points, best_cr

def fibonacci_spiral(n_points):
    """Fibonacci spiral on the sphere."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def random_init(n_points, rng=None):
    """Generate random points on the unit sphere."""
    if rng is None:
        rng = np.random.RandomState()
    v = rng.randn(n_points, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def icosahedron_based_init(n_points, rng=None):
    """Generate points based on icosahedron vertices with subdivisions."""
    if rng is None:
        rng = np.random.RandomState()
    
    # Start with icosahedron vertices
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # If we need more points, add random points
    points = [ico]
    while len(points) < n_points:
        points.append(random_init(1, rng)[0])
    
    return np.array(points[:n_points])

def local_optimize(initial_points, max_iter=100, rng=None):
    """Combine gradient descent and simulated annealing for local optimization."""
    if rng is None:
        rng = np.random.RandomState()
    
    # First, gradient descent
    points, cr = gradient_descent_with_backtracking(initial_points, max_iters=max_iter)
    
    # Then, simulated annealing for fine-tuning
    if time.time() < DEADLINE - 15:
        points_sa, cr_sa = simulated_annealing(points, n_iterations=3000, t_start=0.05, t_end=0.001, rng=rng)
        if cr_sa < cr:
            cr = cr_sa
            points = points_sa
    
    return points, cr

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    best_points = fibonacci_spiral(N)
    best_cr = compute_covering_radius(best_points)
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 20:
            break
        
        if restart == 0:
            init = fibonacci_spiral(N)
        elif restart == 1:
            init = icosahedron_based_init(N, rng)
        elif restart < 4:
            # Perturb current best
            noise = 0.05 * rng.randn(N, 3)
            init = to_sphere(best_points + noise)
        else:
            # Random initialization
            init = random_init(N, rng)
        
        # Optimize
        optimized, cr = local_optimize(init, max_iter=50, rng=rng)
        if cr < best_cr:
            best_cr = cr
            best_points = optimized.copy()
    
    # Final fine-tuning
    if time.time() < DEADLINE - 30:
        optimized, cr = local_optimize(best_points, max_iter=100, rng=rng)
        if cr < best_cr:
            best_cr = cr
            best_points = optimized
    
    # Ensure all points are on the unit sphere
    best_points = to_sphere(best_points)
    
    # Validate: check no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                best_points[j] = best_points[j] * 0.999 + 0.001 * rng.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    # Check not all in a closed hemisphere
    try:
        hull = ConvexHull(best_points)
    except Exception:
        pass
    
    return best_points
