"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
    Generate 13 points as a regular 13-gon with angular perturbations
    using first 2 Fourier harmonics.
    params: [a1, phi1, a2, phi2]
    - a1, phi1: amplitude and phase for 1st harmonic angular perturbation
    - a2, phi2: amplitude and phase for 2nd harmonic angular perturbation
    
    angle(k) = k*2*pi/13 + a1*cos(1*k*2*pi/13 + phi1) + a2*cos(2*k*2*pi/13 + phi2)
    All points are on the unit circle (r=1).
    """
    n = 13
    a1, phi1 = params[0], params[1]
    a2, phi2 = params[2], params[3]
    
    base_angles = np.arange(n) * 2 * np.pi / n
    
    # Angular perturbations (low-frequency Fourier series)
    angle_perturbation = a1 * np.cos(1 * base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    angles = base_angles + angle_perturbation
    
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    # Check convexity: all points should be on the convex hull
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < 13:
            return -0.01  # Strong penalty for non-convex configurations
    except Exception:
        return -0.01
    
    return -_score(points)


def _cma_es(params, sigma0, max_iter, pop_size, rng, bounds):
    """
    Simple CMA-ES implementation for low-dimensional optimization.
    params: initial center
    sigma0: initial step size
    max_iter: maximum iterations
    pop_size: population size (lambda)
    rng: numpy RandomState
    bounds: list of (low, high) for each parameter
    """
    n = len(params)
    center = np.array(params, dtype=float)
    sigma = sigma0
    best_x = center.copy()
    best_f = _objective(best_x)
    
    # Covariance matrix (start as identity)
    C = np.eye(n)
    # Eigenvalue decomposition for sampling
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-12)
    
    # Adaptation parameters
    alpha_mu = 2.0
    c_sigma = (n + 2) / (n + 2 + 5)
    c_c = (4 + n / n) / (n + 4 + 2 * n / n)
    c_1 = 2 / ((n + 1.3) ** 2 + n)
    c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (n + 2) - 1))
    mu_eff = sum([1.0] * (pop_size // 2))  # simplified
    chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
    
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    
    for it in range(max_iter):
        # Generate population
        pops = []
        fitnesses = []
        
        for i in range(pop_size):
            # Sample from multivariate normal using Cholesky-like decomposition
            z = rng.standard_normal(n)
            # Transform using eigenvectors and eigenvalues
            z_transformed = eigvecs @ (eigvals ** 0.5 * z)
            x = center + sigma * z_transformed
            
            # Clip to bounds
            for d in range(n):
                x[d] = np.clip(x[d], bounds[d][0], bounds[d][1])
            
            f = _objective(x)
            pops.append(x)
            fitnesses.append(f)
        
        # Sort by fitness (ascending for minimization)
        sorted_idx = np.argsort(fitnesses)
        sorted_pops = [pops[i] for i in sorted_idx]
        sorted_fits = [fitnesses[i] for i in sorted_idx]
        
        # Update best
        if sorted_fits[0] < best_f:
            best_f = sorted_fits[0]
            best_x = sorted_pops[0].copy()
        
        # Weighted mean of top mu individuals
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        y_w = np.zeros(n)
        for i in range(mu):
            y_w += weights[i] * (sorted_pops[i] - center) / sigma
        
        # Update sigma
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
        sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
        
        # Update covariance
        y_w = np.zeros(n)
        for i in range(mu):
            y_w += weights[i] * (sorted_pops[i] - center) / sigma
        
        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        
        # Rank-one and rank-mu updates
        C = (1 + c_1 * (mu_eff - 2 + c_mu) - c_1) * C
        C += c_1 * (np.outer(p_c, p_c) - C)
        for i in range(mu):
            C += c_mu * weights[i] * np.outer(y_w, y_w)
        
        # Re-orthogonalize and get new eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        
        # Update center
        center = np.zeros(n)
        for i in range(mu):
            center += weights[i] * sorted_pops[i]
        
        # Clip center to bounds
        for d in range(n):
            center[d] = np.clip(center[d], bounds[d][0], bounds[d][1])
        
        # Early stopping if sigma is very small
        if sigma < 1e-10:
            break
    
    return best_x, best_f


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES to optimize 4 parameters: angular perturbations via 2 Fourier harmonics.
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [a1, phi1, a2, phi2]
    # a1, a2: angular perturbation amplitudes (radians)
    # phi1, phi2: phases
    bounds = [
        (-0.3, 0.3),      # a1
        (0, 2 * np.pi),   # phi1
        (-0.2, 0.2),      # a2
        (0, 2 * np.pi),   # phi2
    ]
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with CMA-ES
    for trial in range(12):
        if trial == 0:
            # Start with regular polygon (zero perturbations)
            init_params = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            init_params = np.array([
                rng.uniform(-0.2, 0.2),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.15, 0.15),
                rng.uniform(0, 2 * np.pi),
            ])
        
        try:
            # Run CMA-ES
            final_params, final_f = _cma_es(
                init_params,
                sigma0=0.15,
                max_iter=200,
                pop_size=16,
                rng=rng,
                bounds=bounds
            )
            
            # Clip to bounds
            for i in range(len(final_params)):
                final_params[i] = np.clip(final_params[i], bounds[i][0], bounds[i][1])
            
            points = _generate_points(final_params)
            
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
