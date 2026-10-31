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
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    if len(points) < 3:
        return 0.0
    min_tri = np.min(_triangle_areas_fast(points))
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    return min_tri / hull_area


def _generate_points(params):
    """
    Generate 13 points from parameters using angular perturbation.
    params: [a1, phi1, a2, phi2]
    Angle k = 2*pi*k/13 + a1*cos(1*base_k + phi1) + a2*cos(2*base_k + phi2)
    Points are on the unit circle.
    """
    n = 13
    a1, phi1, a2, phi2 = params
    
    base_angles = np.arange(n) * 2 * np.pi / n
    # Perturbation
    angles = base_angles + a1 * np.cos(1 * base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    
    # Ensure angles are strictly increasing to maintain convex order and non-degenerate hull
    # If perturbation is too large, order might swap. 
    # We assume small a1, a2.
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _cma_es(objective, x0, bounds, sigma=0.1, max_iter=200, rng=None):
    """
    Simplified CMA-ES implementation.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    n = len(x0)
    # Population size
    lam = 4 + int(3 * np.log(n))
    
    # Weights
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mueff = 1.0 / np.sum(weights**2)
    
    # Strategy parameters
    c_sigma = (mueff + 2) / (n + mueff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c_1 = 2 / ((n + 1.3)**2 + mueff)
    c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
    
    # Initial state
    m = np.array(x0, dtype=float)
    sigma = sigma
    C = np.eye(n)
    B = np.eye(n)
    D = np.ones(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    
    best_x = m.copy()
    best_f = objective(m)
    
    for it in range(max_iter):
        # Sample population
        pop = np.zeros((lam, n))
        for i in range(lam):
            z = rng.standard_normal(n)
            # Transform z using B and D
            # B is eigenvectors, D is sqrt eigenvalues
            # x = m + sigma * B * (D * z)
            # Actually C = B * diag(D^2) * B^T
            # sqrt(C) = B * diag(D)
            # So x = m + sigma * B @ (D * z)
            pop[i] = m + sigma * (B @ (D * z))
        
        # Evaluate
        fits = np.array([objective(x) for x in pop])
        
        # Sort
        order = np.argsort(fits)
        pop = pop[order]
        fits = fits[order]
        
        # Update best
        if fits[0] < best_f:
            best_f = fits[0]
            best_x = pop[0].copy()
            
        # Update mean
        m_new = m + sigma * np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
        
        # Update pc
        pc = (1 - c_c) * pc + c_c * (m_new - m) / sigma
        
        # Update ps
        ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * B @ (D @ ((m_new - m) / sigma))
        
        # Update sigma
        sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(n) - 1))
        
        # Update C
        # Rank-one update
        v = (m_new - m) / sigma
        C += (c_1 + c_mu) * np.outer(v, v)
        
        # Rank-mu update
        # C += c_mu * sum(weights * outer(v_i, v_i))
        # v_i = (x_i - m) / sigma
        diff = (pop[:mu] - m) / sigma
        C += c_mu * (weights[:, np.newaxis] * diff).T @ diff
        
        # Ensure C is symmetric and positive definite (approx)
        C = (C + C.T) / 2.0
        
        # Decompose C
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip negative eigenvalues
            eigvals = np.maximum(eigvals, 1e-10)
            B = eigvecs
            D = np.sqrt(eigvals)
        except np.linalg.LinAlgError:
            # Fallback to identity if decomposition fails
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            
        m = m_new
        
        # Check convergence
        if sigma < 1e-6:
            break
            
    return best_x, best_f


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES to optimize angular perturbations.
    """
    best_score = 0.0
    best_points = None
    
    # Parameters: [a1, phi1, a2, phi2]
    # a1, a2 are small amplitudes to keep points on circle and ordered
    # phi1, phi2 are phases
    bounds = [
        (-0.2, 0.2),  # a1
        (0, 2 * np.pi),  # phi1
        (-0.2, 0.2),  # a2
        (0, 2 * np.pi),  # phi2
    ]
    
    rng = np.random.RandomState(42)
    
    # Objective function for CMA-ES (minimization of negative score)
    def obj(params):
        # Clip to bounds
        p = np.array(params)
        for i in range(len(p)):
            p[i] = np.clip(p[i], bounds[i][0], bounds[i][1])
        
        points = _generate_points(p)
        
        # Check convexity/ordering
        # If points are on unit circle with sorted angles, hull is the polygon.
        # We should verify that the angles are strictly increasing modulo 2pi
        # and that no three points are collinear (unlikely on circle).
        # The main risk is angle wrapping.
        # Let's just compute score. If hull area is 0 or invalid, return penalty.
        
        try:
            hull = ConvexHull(points)
            if len(hull.vertices) < 13:
                return 100.0 # Penalty
        except Exception:
            return 100.0
            
        s = _score(points)
        if s <= 0:
            return 100.0
        return -s
    
    # Initial guess: regular polygon (a1=0, a2=0)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Run CMA-ES
    try:
        best_params, best_neg_score = _cma_es(
            obj, 
            x0, 
            bounds, 
            sigma=0.1, 
            max_iter=300, 
            rng=rng
        )
        
        # Clip final params
        for i in range(len(best_params)):
            best_params[i] = np.clip(best_params[i], bounds[i][0], bounds[i][1])
            
        points = _generate_points(best_params)
        
        # Final validation
        try:
            hull = ConvexHull(points)
            if len(hull.vertices) < 13:
                raise Exception("Not convex")
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points
        except Exception:
            pass
            
    except Exception:
        pass
        
    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
