"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _triangle_areas_fast(points):
    """Compute areas of all triangles formed by triples of points using vectorized operations."""
    n = len(points)
    # Use the cross product formula: area = 0.5 * |det([p2-p1, p3-p1])|
    # For all triples (i, j, k) with i < j < k
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
    Generate 13 points as a regular 13-gon with angular perturbation.
    params: [theta0, a1, phi1, a2, phi2]
    - theta0: base angle offset
    - a1, phi1: amplitude and phase for 1st harmonic angular perturbation
    - a2, phi2: amplitude and phase for 2nd harmonic angular perturbation
    
    angle(k) = theta0 + k*2*pi/13 + a1*cos(1*(k*2*pi/13) + phi1) + a2*cos(2*(k*2*pi/13) + phi2)
    Points are on unit circle: (cos(angle), sin(angle))
    """
    n = 13
    theta0 = params[0]
    a1, phi1 = params[1], params[2]
    a2, phi2 = params[3], params[4]
    
    base_angles = np.arange(n) * 2 * np.pi / n
    
    # Angular perturbation with 2 harmonics
    angle_pert = a1 * np.cos(1 * base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    
    angles = theta0 + base_angles + angle_pert
    
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points





def _cma_es(objective, x0, sigma0, bounds, max_iter=200, seed=42):
    """
    CMA-ES optimizer.
    
    Parameters:
    - objective: function to minimize
    - x0: initial mean
    - sigma0: initial step size
    - bounds: list of (low, high) for each parameter
    - max_iter: maximum iterations
    - seed: random seed
    
    Returns: best_x, best_f
    """
    rng = np.random.RandomState(seed)
    n = len(x0)
    
    # Initialize
    mean = np.array(x0, dtype=float)
    sigma = sigma0
    C = np.eye(n)  # Covariance matrix
    C_inv = np.eye(n)
    
    # Step size adaptation parameters
    lambda_ = max(4, int(4 + 3 * np.log(n)))
    mu = lambda_ // 2  # Number of weights
    # Weights: mu_i proportional to log(i+1) - log(i)
    weights = np.array([np.log(i + 1) - np.log(i) for i in range(1, mu + 1)])
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights ** 2)
    
    # Learning rates
    cc = (4 + mu_eff / n) / ((n + 4) ** 2 + mu_eff / 2)
    cs = (mu_eff + 2) / (n + mu_eff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    sqrt_mu_eff = np.sqrt(mu_eff)
    if sqrt_mu_eff > 2:
        cmu = min(1 - c1, 2 * (sqrt_mu_eff - 2) / ((n + 2) ** 2 + sqrt_mu_eff))
    else:
        cmu = 1 - c1
    
    # Path updates
    ps = np.zeros(n)
    pc = np.zeros(n)
    
    # Damping for step size
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    
    best_x = mean.copy()
    best_f = objective(mean)
    
    for it in range(max_iter):
        # Generate lambda samples
        samples = np.array([mean + sigma * rng.multivariate_normal(np.zeros(n), C) for _ in range(lambda_)])
        
        # Clip to bounds
        for i in range(n):
            samples[:, i] = np.clip(samples[:, i], bounds[i][0], bounds[i][1])
        
        # Evaluate
        fitness = np.array([objective(x) for x in samples])
        
        # Sort by fitness
        idx = np.argsort(fitness)
        samples = samples[idx]
        fitness = fitness[idx]
        
        # Update best
        if fitness[0] < best_f:
            best_f = fitness[0]
            best_x = samples[0].copy()
        
        # Compute weighted mean
        new_mean = np.sum(weights[:, None] * samples[:mu], axis=0)
        
        # Update path ps
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv.dot(sigma * (new_mean - mean))
        
        # Update path pc
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) < (1 + 2 / (n + 1)) * chi_n
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * C_inv.dot(new_mean - mean)
        
        # Update covariance
        delta_hsig = (1 - hsig) * cc * (2 - cc)
        rank_one = np.outer(new_mean - mean, new_mean - mean)
        rank_mu = np.sum(weights[:, None] * np.outer(samples[:mu] - mean, samples[:mu] - mean), axis=0)
        C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * (rank_one + delta_hsig * np.outer(pc, pc)) + cmu * rank_mu
        
        # Ensure positive definiteness
        C = (C + C.T) / 2
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Update sigma
        sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
        
        # Update mean
        mean = new_mean
        
        # Re-clip mean to bounds
        for i in range(n):
            mean[i] = np.clip(mean[i], bounds[i][0], bounds[i][1])
        
        # Recompute C_inv
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            C_inv = np.eye(n)
    
    return best_x, best_f


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES with angular perturbation model (2 harmonics).
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [theta0, a1, phi1, a2, phi2]
    bounds = [
        (0, 2 * np.pi),      # theta0
        (-0.3, 0.3),         # a1 (angular perturbation amplitude)
        (0, 2 * np.pi),      # phi1
        (-0.2, 0.2),         # a2 (angular perturbation amplitude)
        (0, 2 * np.pi),      # phi2
    ]
    
    def objective(params):
        points = _generate_points(params)
        try:
            hull = ConvexHull(points)
            if len(hull.vertices) < 13:
                return 0.01  # Penalty for non-convex
        except Exception:
            return 0.01
        return -_score(points)
    
    # Try multiple restarts with different seeds
    for seed in range(5):
        rng = np.random.RandomState(seed)
        
        # Initial parameters
        if seed == 0:
            init_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            init_params = np.array([
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.2, 0.2),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.15, 0.15),
                rng.uniform(0, 2 * np.pi),
            ])
        
        try:
            best_params, best_f = _cma_es(
                objective,
                init_params,
                sigma0=0.1,
                bounds=bounds,
                max_iter=100,
                seed=seed
            )
            
            points = _generate_points(best_params)
            
            # Verify convexity
            try:
                hull = ConvexHull(points)
                if len(hull.vertices) < 13:
                    continue
            except Exception:
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
