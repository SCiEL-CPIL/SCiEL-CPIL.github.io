"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _triangle_areas_fast(points):
    """Compute areas of all triangles formed by triples of points using vectorized operations."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas.append(0.5 * abs(cross))
    return np.array(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    hull = ConvexHull(points)
    return hull.volume


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = np.min(_triangle_areas_fast(points))
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_tri / hull_area


def _generate_points(params):
    """
    Generate 13 points from parameters using angular Fourier perturbation.
    params: [theta0, a1, phi1, a2, phi2]
    - theta0: global rotation
    - a1, phi1: amplitude and phase for 1st harmonic angular perturbation
    - a2, phi2: amplitude and phase for 2nd harmonic angular perturbation
    
    Points are placed on a unit circle with perturbed angles:
    angle(k) = theta0 + k*2*pi/13 + a1*cos(k*2*pi/13 + phi1) + a2*cos(2*k*2*pi/13 + phi2)
    """
    n = 13
    theta0 = params[0]
    a1, phi1 = params[1], params[2]
    a2, phi2 = params[3], params[4]
    
    base_angles = np.arange(n) * 2 * np.pi / n
    # Angular perturbation with 2 harmonics
    angle_perturbation = a1 * np.cos(1 * base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    angles = theta0 + base_angles + angle_perturbation
    
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _is_convex(points):
    """Check if all points are on the convex hull (i.e., form a convex polygon in order)."""
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == len(points)
    except Exception:
        return False


def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    if not _is_convex(points):
        return 1.0  # Penalize non-convex configurations
    return -_score(points)


def _cma_es(params, sigma, rng, bounds, max_iter=500, pop_size=None):
    """
    Simple CMA-ES implementation for small-scale optimization.
    """
    n_params = len(params)
    if pop_size is None:
        pop_size = max(4, int(4 + 3 * np.log(n_params)))
    
    mean = params.copy()
    sigma_val = sigma
    C = np.eye(n_params)
    C_inv = np.eye(n_params)
    C_sqrt = np.eye(n_params)
    
    c1 = (2 + 2.0 / (n_params + 1.3 ** 2)) / ((n_params + 2) ** 2 + 2)
    cmu = min(1 - c1, 2 * (2 - (n_params + 1) / (n_params + 4)) / ((n_params + 2) ** 2 + 5))
    cs = (4 + n_params / 4) / (n_params + 4 + 4 / n_params)
    c_sigma = 1.0
    d_sigma = 1.0 + 2 * max(0, np.sqrt((n_params - 1) / (n_params + 3)) - 1) + cs
    chi_n = np.sqrt(n_params) * (1 - 1.0 / (4 * n_params) + 1.0 / (21 * n_params ** 2))
    
    p_sigma = np.zeros(n_params)
    p_c = np.zeros(n_params)
    
    best_x = mean.copy()
    best_f = _objective(mean)
    
    for it in range(max_iter):
        # Generate population
        population = []
        for i in range(pop_size):
            z = rng.standard_normal(n_params)
            x = mean + sigma_val * C_sqrt @ z
            # Wrap angle parameters to [0, 2*pi)
            x[0] = x[0] % (2 * np.pi)
            x[2] = x[2] % (2 * np.pi)
            x[4] = x[4] % (2 * np.pi)
            # Clip amplitude parameters
            x[1] = np.clip(x[1], -0.5, 0.5)
            x[3] = np.clip(x[3], -0.3, 0.3)
            population.append(x)
        
        # Evaluate
        fitness = np.array([_objective(x) for x in population])
        
        # Sort by fitness
        order = np.argsort(fitness)
        population_sorted = [population[i] for i in order]
        fitness_sorted = fitness[order]
        
        # Update best
        if fitness_sorted[0] < best_f:
            best_f = fitness_sorted[0]
            best_x = population_sorted[0].copy()
        
        # Weighted mean
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        new_mean = np.sum([weights[i] * population_sorted[i] for i in range(mu)], axis=0)
        
        # Update step size
        p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) / sigma_val * (C_inv @ (new_mean - mean))
        sigma_val *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
        sigma_val = np.clip(sigma_val, 1e-8, 1.0)
        
        # Update covariance matrix
        mu_eff = 1 / np.sum(weights ** 2)
        p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mu_eff) * (new_mean - mean)
        
        delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1.4 + 2 / (n_params + 1)) * chi_n else 0
        C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - delta_h) * cs * (2 - cs) * C) + cmu * np.sum([weights[i] * np.outer(population_sorted[i] - mean, population_sorted[i] - mean) for i in range(mu)], axis=0)
        
        # Recompute C_sqrt and C_inv periodically
        if it % 10 == 0:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        
        mean = new_mean
        mean[0] = mean[0] % (2 * np.pi)
        mean[2] = mean[2] % (2 * np.pi)
        mean[4] = mean[4] % (2 * np.pi)
        mean[1] = np.clip(mean[1], -0.5, 0.5)
        mean[3] = np.clip(mean[3], -0.3, 0.3)
        
        # Convergence check
        if sigma_val < 1e-7:
            break
    
    return best_x, best_f


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES to optimize 5 parameters: global rotation and 2 angular harmonics.
    """
    best_score = 0.0
    best_points = None
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    n_restarts = 6
    for trial in range(n_restarts):
        if trial == 0:
            # Start with regular polygon (zero perturbations)
            init_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            init_params = np.array([
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.1, 0.1),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.05, 0.05),
                rng.uniform(0, 2 * np.pi),
            ])
        
        try:
            # Run CMA-ES
            optimal_params, opt_val = _cma_es(
                init_params,
                sigma=0.3,
                rng=rng,
                bounds=None,
                max_iter=400
            )
            
            points = _generate_points(optimal_params)
            
            # Verify convexity
            if not _is_convex(points):
                continue
            
            score = _score(points)
            
            if score > best_score:
                best_score = score
                best_points = points
        except Exception:
            continue
    
    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
