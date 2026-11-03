"""Packing 50 equal circles in a unit circle via DE over ring parameters + multi-stage smooth polish."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

def _score(points):
    """Compute the objective: minimum of half-min-pair-distance and min-margin."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    norms = np.sqrt(np.sum(points**2, axis=1))
    min_margin = np.min(1.0 - norms)
    return min(min_pair_dist / 2.0, min_margin)

def _softmin_logsumexp(dists, beta):
    """Soft minimum via log-sum-exp."""
    neg_beta = -beta * dists
    max_val = np.max(neg_beta)
    if not np.isfinite(max_val):
        return -np.inf
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta - max_val)))
    return -log_sum_exp / beta

def _smooth_objective_and_grad(params, beta):
    """
    params: flat array of length 2*N (x0,y0,x1,y1,...)
    Returns (objective, gradient) for smooth approximation of -min_radius.
    Vectorized for speed.
    """
    pts = params.reshape(N, 2)
    
    # Pairwise distances (vectorized)
    iu = np.triu_indices(N, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1) + 1e-30)
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - radii
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    scaled = all_dists / 2.0
    
    # Soft min via log-sum-exp (stable)
    neg_beta_scaled = -beta * scaled
    max_val = np.max(neg_beta_scaled)
    exp_vals = np.exp(neg_beta_scaled - max_val)
    log_sum_exp = max_val + np.log(np.sum(exp_vals))
    soft_min_val = -log_sum_exp / beta
    objective = -soft_min_val
    
    # Weights for gradient
    weights = exp_vals / np.sum(exp_vals)
    if np.sum(weights) < 1e-300:
        return objective, np.zeros_like(params)
    
    grad_scaled = -beta * weights
    n_pairs = len(pair_dists)
    
    grad_pts = np.zeros_like(pts)
    
    # Pair contributions (vectorized)
    w_pairs = grad_scaled[:n_pairs] / 2.0
    d_safe = np.maximum(pair_dists, 1e-14)
    d_dij = diff / d_safe[:, np.newaxis]
    # Accumulate: for each pair (i,j), grad_pts[i] += w * d_dij, grad_pts[j] -= w * d_dij
    for k in range(n_pairs):
        w = w_pairs[k]
        if abs(w) > 1e-20:
            i, j = iu[0][k], iu[1][k]
            grad_pts[i] += w * d_dij[k]
            grad_pts[j] -= w * d_dij[k]
    
    # Boundary contributions (vectorized)
    w_boundary = grad_scaled[n_pairs:] / 2.0
    r_safe = np.maximum(radii, 1e-14)
    d_dr = pts / r_safe[:, np.newaxis]
    grad_pts += (w_boundary[:, np.newaxis] * d_dr)
    
    return objective, grad_pts.flatten()

def _ring_initialization(ring_radii, ring_counts, angular_offsets=None):
    """
    Generate points from ring specification.
    ring_radii: array of radii for each ring
    ring_counts: array of counts per ring
    angular_offsets: array of angular offsets per ring (optional)
    """
    points = []
    n_rings = len(ring_radii)
    if angular_offsets is None:
        angular_offsets = np.zeros(n_rings)
    
    for i in range(n_rings):
        r = ring_radii[i]
        count = ring_counts[i]
        offset = angular_offsets[i]
        if count == 0:
            continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + offset
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    
    return np.array(points[:N])

def _ring_params_to_points(params):
    """
    Convert DE parameters to points.
    params structure: [ring_radii (5), angular_offsets (5)] = 10 params
    Fixed ring counts: [1, 6, 12, 18, 13] = 50
    """
    ring_counts = np.array([1, 6, 12, 18, 13])
    ring_radii = params[:5]
    angular_offsets = params[5:10]
    
    # Ensure center is at origin
    ring_radii[0] = 0.0
    
    return _ring_initialization(ring_radii, ring_counts, angular_offsets)

def _optimize_ring_with_de(deadline):
    """Use differential evolution to find good ring parameters."""
    # Bounds: radii in [0, 0.95], offsets in [0, 2*pi]
    bounds = [(0.0, 0.95)] * 5 + [(0.0, 2*np.pi)] * 5
    
    def neg_score(params):
        pts = _ring_params_to_points(params)
        return -_score(pts)
    
    try:
        result = differential_evolution(
            neg_score, bounds,
            seed=42, maxiter=300, popsize=20,
            tol=1e-12, mutation=(0.4, 1.2), recombination=0.85,
            polish=True, init='latinhypercube'
        )
        best_params = result.x
        best_pts = _ring_params_to_points(best_params)
        return best_pts, -result.fun
    except Exception:
        # Fallback: fixed ring initialization
        ring_radii = np.array([0.0, 0.18, 0.36, 0.54, 0.72])
        ring_counts = np.array([1, 6, 12, 18, 13])
        return _ring_initialization(ring_radii, ring_counts), _score(_ring_initialization(ring_radii, ring_counts))

def _multi_stage_smooth_polish(points, deadline):
    """
    Multi-stage beta annealing: start with low beta (smooth), increase to high beta (sharp).
    Uses combined objective+gradient to avoid redundant computation.
    """
    current_pts = points.copy()
    current_score = _score(current_pts)
    
    beta_stages = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    
    for beta in beta_stages:
        if time.time() > deadline:
            break
        
        try:
            def obj(x):
                return _smooth_objective_and_grad(x, beta)[0]
            
            def grad(x):
                return _smooth_objective_and_grad(x, beta)[1]
            
            result = minimize(
                obj,
                grad,
                current_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            
            if time.time() > deadline:
                break
            
            new_pts = result.x.reshape(N, 2)
            
            if not np.all(np.isfinite(new_pts)):
                continue
            
            # Clamp to unit circle
            norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] *= (1.0 - 1e-12) / norms[mask]
            
            new_score = _score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        
        except Exception:
            continue
    
    return current_pts, current_score

def _binary_search_polish(points, deadline):
    """
    Multi-scale perturbation polish with re-optimization.
    """
    current_pts = points.copy()
    current_score = _score(current_pts)
    
    rng = np.random.default_rng(123)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Try multiple perturbation scales
    scales = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for scale in scales:
        if time.time() > deadline:
            break
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Random perturbation at this scale
            perturbed = current_pts + rng.normal(0, scale, size=(N, 2))
            norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                perturbed[mask] *= (1.0 - 1e-12) / norms[mask]
            
            polished, score = _multi_stage_smooth_polish(perturbed, deadline)
            if score > best_score:
                best_score = score
                best_pts = polished.copy()
                current_pts = polished.copy()
    
    return best_pts, best_score

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: DE over ring parameters
    if time.time() < deadline - 50.0:
        ring_pts, ring_score = _optimize_ring_with_de(deadline - 50.0)
        if ring_score > best_score:
            best_score = ring_score
            best_points = ring_pts.copy()
    
    # Phase 2: Multi-stage smooth polish from ring solution
    if best_points is not None and time.time() < deadline - 25.0:
        polished, score = _multi_stage_smooth_polish(best_points, deadline - 25.0)
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    # Phase 3: Try hexagonal initialization as alternative
    if time.time() < deadline - 25.0:
        hex_pts = _hexagonal_initial()
        hex_score = _score(hex_pts)
        if hex_score > best_score:
            best_score = hex_score
            best_points = hex_pts.copy()
        
        if time.time() < deadline - 15.0:
            polished, score = _multi_stage_smooth_polish(best_points, deadline - 15.0)
            if score > best_score:
                best_score = score
                best_points = polished.copy()
    
    # Phase 4: Multi-scale perturbation polish
    if best_points is not None and time.time() < deadline - 8.0:
        final_pts, final_score = _binary_search_polish(best_points, deadline - 8.0)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
    
    # Phase 5: Final high-beta polish
    if best_points is not None and time.time() < deadline - 3.0:
        polished, score = _multi_stage_smooth_polish(best_points, deadline - 3.0)
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    if best_points is None:
        best_points = _hexagonal_initial()
    
    best_points = np.array(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_points
