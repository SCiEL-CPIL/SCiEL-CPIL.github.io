"""Packing 50 equal circles in a unit circle using L-BFGS-B on a soft-min surrogate."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 115.0  # safety margin before 120s kill


def _score(points):
    """Compute the achievable radius for a set of points."""
    # Distance from each center to origin
    norms = np.linalg.norm(points, axis=1)
    # Min distance from center to boundary
    min_boundary = 1.0 - np.max(norms)
    
    # Pairwise distances
    n = len(points)
    if n < 2:
        return min_boundary
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)


def _soft_min_surrogate(points, beta):
    """
    Objective to minimize: -soft_min of constraints.
    We want to maximize r = min(min_pair_dist/2, min_boundary_dist).
    Equivalent to minimizing -r.
    
    Using log-sum-exp style soft-min:
    soft_min(a, b) ≈ -log(exp(-beta*a) + exp(-beta*b)) / beta  for large beta
    
    But we have many pairwise constraints. Let's use:
    r = min( min_{i<j} d_ij/2, min_i (1 - |c_i|) )
    
    We'll use a different approach: define a function that penalizes violations.
    Actually, let's use the approach from the inspiration: L-BFGS-B on soft-min.
    
    Let's define:
    f(points) = -softmin( [d_ij/2 for all i<j], [1-|c_i| for all i] )
    
    For numerical stability, we use:
    softmin(x) = -log(sum(exp(-beta*x_i))) / beta
    
    But with ~1250 constraints, this could be expensive. Let's use a simpler approach:
    Use a penalty-based approach with L-BFGS-B.
    
    Actually, let's use a cleaner formulation. We maximize r subject to:
    |c_i - c_j| >= 2r for all i < j
    |c_i| + r <= 1 for all i
    
    This is a QCQP. We can use the epigraph formulation with a smooth surrogate.
    
    Let's use: minimize -r + penalty * sum of constraint violations
    Or better: use the soft-min directly.
    """
    n = len(points)
    norms = np.linalg.norm(points, axis=1)
    boundary_vals = 1.0 - norms  # shape (n,)
    
    # Pairwise distances / 2
    if n >= 2:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
        # Get upper triangle
        iu = np.triu_indices(n, k=1)
        pair_vals = dists[iu]  # shape (n*(n-1)/2,)
    else:
        pair_vals = np.array([1.0])
    
    all_vals = np.concatenate([pair_vals, boundary_vals])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # For numerical stability, subtract min
    min_val = np.min(all_vals)
    if min_val < -1000:
        return min_val
    exp_vals = np.exp(-beta * (all_vals - min_val))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = min_val - log_sum / beta
    
    return -soft_min_val  # We minimize this, so we maximize soft_min


def _soft_min_grad(points, beta):
    """Gradient of the soft-min surrogate objective."""
    n = len(points)
    norms = np.linalg.norm(points, axis=1)
    boundary_vals = 1.0 - norms
    
    if n >= 2:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
        iu = np.triu_indices(n, k=1)
        pair_vals = dists[iu]
    else:
        pair_vals = np.array([1.0])
    
    all_vals = np.concatenate([pair_vals, boundary_vals])
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    sum_exp = np.sum(exp_vals)
    
    # d(soft_min)/d(x_k) = -exp(-beta*x_k) / (beta * sum(exp(-beta*x_j)))
    # So d(-soft_min)/d(x_k) = exp(-beta*x_k) / (beta * sum(exp(-beta*x_j)))
    
    weights = exp_vals / (beta * sum_exp)
    
    grad = np.zeros_like(points)
    
    # Pairwise part
    if n >= 2:
        pair_weights = weights[:len(pair_vals)]
        # For each pair (i,j), d(pair_val)/d(points[i]) = -(points[j]-points[i]) / (2 * |points[j]-points[i]|) * 2
        # Wait, pair_val = |c_i - c_j| / 2
        # d(pair_val)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
        # So d(-soft_min)/d(c_i) += pair_weight * (c_i - c_j) / (2 * |c_i - c_j|)
        
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                grad[i] += pair_weights[idx] * d / (2.0 * dist)
                grad[j] -= pair_weights[idx] * d / (2.0 * dist)
    
    # Boundary part
    boundary_weights = weights[len(pair_vals):]
    for i in range(n):
        if norms[i] > 1e-12:
            grad[i] += boundary_weights[i] * (-points[i] / norms[i])
    
    return grad


def _hex_init(n):
    """Generate a hexagonal lattice initialization."""
    points = []
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x**2 + y**2 <= (1.0 - spacing / 2.0)**2:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        return np.random.randn(n, 2) * 0.5
    if len(points) < n:
        # Pad with random points
        extra = np.random.randn(n - len(points), 2) * 0.3
        points = np.vstack([points, extra])
    # Sort by radius and take first n
    radii = np.linalg.norm(points, axis=1)
    idx = np.argsort(radii)[:n]
    return points[idx]


def _random_init(n, scale=0.5):
    """Random initialization."""
    return np.random.randn(n, 2) * scale


def _ring_init(n):
    """Initialize points on concentric rings."""
    points = []
    # Try to place points on rings
    remaining = n
    ring_radius = 0.1
    while remaining > 0 and ring_radius < 0.95:
        # How many points can fit on this ring?
        circumference = 2 * np.pi * ring_radius
        # Approximate number of points that fit with spacing ~ 2*r
        # This is rough
        if ring_radius < 0.05:
            count = 1
        else:
            count = max(1, int(circumference / (2.0 * ring_radius * 0.8)))
        
        count = min(count, remaining)
        if count == 1 and ring_radius < 0.1:
            points.append([0.0, 0.0])
            remaining -= 1
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for a in angles:
                points.append([ring_radius * np.cos(a), ring_radius * np.sin(a)])
            remaining -= count
        ring_radius += 0.15
    
    points = np.array(points)
    if len(points) < n:
        extra = np.random.randn(n - len(points), 2) * 0.4
        points = np.vstack([points, extra])
    elif len(points) > n:
        points = points[:n]
    
    return points


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -1.0
    
    # Multiple initializations
    inits = []
    
    # Hex lattice
    inits.append(_hex_init(N))
    
    # Ring init
    inits.append(_ring_init(N))
    
    # Random inits
    for scale in [0.3, 0.5, 0.7]:
        inits.append(_random_init(N, scale))
    
    # Also try a more structured random
    for _ in range(3):
        # Random but with some structure
        pts = np.random.randn(N, 2) * 0.6
        # Sort by radius
        radii = np.linalg.norm(pts, axis=1)
        idx = np.argsort(radii)
        inits.append(pts[idx])
    
    beta_values = [50, 100, 200, 500]
    
    for init in inits:
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        current_points = init.copy()
        current_score = _score(current_points)
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        
        # Try different beta values
        for beta in beta_values:
            if time.time() - start_time > _DEADLINE - 10:
                break
            
            def objective(x):
                pts = x.reshape(-1, 2)
                return _soft_min_surrogate(pts, beta)
            
            def gradient(x):
                pts = x.reshape(-1, 2)
                g = _soft_min_grad(pts, beta)
                return g.flatten()
            
            x0 = current_points.flatten()
            
            try:
                result = minimize(
                    objective,
                    x0,
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                )
                
                if result.success or result.fun < objective(x0):
                    new_points = result.x.reshape(-1, 2)
                    new_score = _score(new_points)
                    if new_score > current_score:
                        current_points = new_points
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
            except Exception:
                pass
        
        # Final polish with high beta
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        for beta in [1000, 2000]:
            if time.time() - start_time > _DEADLINE - 3:
                break
            
            def objective(x):
                pts = x.reshape(-1, 2)
                return _soft_min_surrogate(pts, beta)
            
            def gradient(x):
                pts = x.reshape(-1, 2)
                g = _soft_min_grad(pts, beta)
                return g.flatten()
            
            x0 = current_points.flatten()
            
            try:
                result = minimize(
                    objective,
                    x0,
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
                )
                
                if result.success or result.fun < objective(x0):
                    new_points = result.x.reshape(-1, 2)
                    new_score = _score(new_points)
                    if new_score > current_score:
                        current_points = new_points
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback to hex lattice
        best_points = _hex_init(N)
    
    return best_points
# EVOLVE-BLOCK-END
