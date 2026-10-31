"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _triangle_areas_fast(points):
    """Compute areas of all triangles formed by triples of points using vectorized operations."""
    n = len(points)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas[idx] = 0.5 * abs(cross)
                idx += 1
    return areas


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
    Generate 13 points with angular perturbations using first 2 harmonics.
    params: [theta0, a1, phi1, a2, phi2]
    - theta0: base angle offset
    - a_m, phi_m: amplitude and phase for m-th harmonic angular perturbation
    
    Angles are: theta0 + k*2*pi/13 + sum_m a_m * cos(m * k*2*pi/13 + phi_m)
    Points are on unit circle: (cos(angle), sin(angle))
    """
    n = 13
    theta0 = params[0]
    a1, phi1 = params[1], params[2]
    a2, phi2 = params[3], params[4]
    
    base_angles = np.arange(n) * 2 * np.pi / n
    
    # Angular perturbation with first 2 harmonics
    angle_perturb = a1 * np.cos(1 * base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    
    angles = theta0 + base_angles + angle_perturb
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _is_convex(points):
    """Check if all 13 points are on the convex hull (i.e., form a convex polygon)."""
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == 13
    except Exception:
        return False


def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    if not _is_convex(points):
        return 0.01  # Penalty for non-convex configurations
    return -_score(points)


def _cma_es(params, sigma, rng, bounds, pop_size=40, max_iter=300, seed=None):
    """
    Simple CMA-ES implementation for maximizing score (minimizing negative score).
    """
    n_params = len(params)
    mu = pop_size // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= np.sum(w)
    sigma_eff = sigma
    mean = np.array(params, dtype=float)
    
    # Initialize covariance matrix as identity
    C = np.eye(n_params)
    # Learning rates
    c_sigma = (mu + 2) / (n_params + 4) + 2 * max(0, sqrt(n_params / mu) - 1)
    c_c = (4 + n_params / mu) / (n_params + 4 + 2 * mu / n_params)
    
    # Track best
    best_score = _score(_generate_points(mean))
    best_params = mean.copy()
    
    ps = np.zeros(n_params)
    
    for it in range(max_iter):
        # Sample population
        population = []
        for i in range(pop_size):
            # Sample from N(mean, sigma^2 * C)
            z = rng.standard_normal(n_params)
            # Transform using Cholesky of C
            L = np.linalg.cholesky(C)
            x = mean + sigma_eff * L @ z
            # Clip to bounds
            for j in range(n_params):
                x[j] = np.clip(x[j], bounds[j][0], bounds[j][ 1])
            population.append(x)
        
        # Evaluate population
        scores = [_objective(x) for x in population]
        
        # Sort by objective (lower is better)
        idx = np.argsort(scores)
        x_w = np.array(population)[idx[:mu]]
        scores_w = np.array(scores)[idx[:mu]]
        
        # Update mean
        new_mean = np.sum(w[:, None] * x_w, axis=0)
        
        # Update step size
        sigma_eff *= np.exp(c_sigma / c_sigma * (np.linalg.norm(new_mean - mean) / (sigma_eff * np.sqrt(2 * mu)) - 1))
        sigma_eff = np.clip(sigma_eff, 1e-8, 10.0)
        
        # Update covariance
        diff = (new_mean - mean) / sigma_eff
        ps = (1 - c_c) * ps + np.sqrt(c_c * (2 - c_c) * mu * w) * diff
        C = (1 - c_c) * C + c_c * (np.outer(ps, ps) + (1 - (1 - c_c) ** 2) * C)
        
        mean = new_mean
        
        # Check best
        current_score = _score(_generate_points(mean))
        if current_score > best_score:
            best_score = current_score
            best_params = mean.copy()
        
        # Early stopping if sigma is very small
        if sigma_eff < 1e-10:
            break
    
    return best_params, best_score


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES to optimize angular perturbations with first 2 harmonics.
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [theta0, a1, phi1, a2, phi2]
    bounds = [
        (0, 2 * np.pi),      # theta0
        (-0.3, 0.3),         # a1
        (0, 2 * np.pi),      # phi1
        (-0.2, 0.2),         # a2
        (0, 2 * np.pi),      # phi2
    ]
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    for trial in range(10):
        if trial == 0:
            # Start with regular polygon
            init_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            sigma0 = 0.1
        else:
            init_params = np.array([
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.2, 0.2),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.15, 0.15),
                rng.uniform(0, 2 * np.pi),
            ])
            sigma0 = 0.05 + 0.05 * rng.random()
        
        try:
            opt_params, opt_score = _cma_es(
                init_params, sigma0, rng, bounds,
                pop_size=30, max_iter=200
            )
            
            points = _generate_points(opt_params)
            if _is_convex(points):
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
