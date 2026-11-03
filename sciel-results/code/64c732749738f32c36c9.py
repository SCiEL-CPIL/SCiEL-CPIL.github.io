"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _score(centres):
    """Compute the packing score (largest common radius)."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Min distance to boundary
    dist_boundary = 0.5 - np.abs(centres)
    min_boundary = np.min(dist_boundary)
    if min_boundary < -1e-12:
        return 0.0
    # Pairwise distances
    d = pdist(centres)
    min_pair = np.min(d)
    r = min(min_pair / 2.0, min_boundary)
    return r

def _soft_min(d, beta):
    """Smooth approximation of min(d) using log-sum-exp."""
    d_min = np.min(d)
    # For numerical stability
    terms = np.exp(-beta * (d - d_min))
    s = np.sum(terms)
    return d_min - np.log(s) / beta

def _objective_and_grad(x, beta):
    """
    Objective: minimize -min_distance + boundary penalty.
    Uses soft-min for smoothness.
    Returns (value, gradient).
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    d = pdist(c)
    i, j = np.triu_indices(N, 1)
    diff = c[i] - c[j]
    dist = np.sqrt(np.sum(diff**2, axis=1) + 1e-12)
    
    # Soft min of pairwise distances
    min_pair_soft = _soft_min(dist, beta)
    
    # Boundary distances
    face_dists = 0.5 - np.abs(c)
    min_face_soft = _soft_min(face_dists, beta)
    
    # The radius is min(min_pair/2, min_face)
    # We want to maximize this, so minimize -min(min_pair/2, min_face)
    # Use soft min again for the outer min
    r_soft = _soft_min(np.array([min_pair_soft / 2.0, min_face_soft]), beta * 0.5)
    
    value = -r_soft
    
    # Gradient computation
    # Gradient of soft_min: -sum(exp(-beta*(d-dmin)) / sum(exp(...)) * grad_d)
    # This is complex to derive analytically for the nested soft-min.
    # Instead, we use a simpler approach: finite differences or just use L-BFGS-B with numerical gradient.
    # But L-BFGS-B is slow. Let's try to compute gradient for the pairwise part only, 
    # or use a different strategy.
    
    # Actually, let's just use the value and let L-BFGS-B compute the gradient numerically.
    # Or better, let's use a different optimization approach.
    
    return value

def _objective(x, beta):
    """Objective function for minimization."""
    c = x.reshape(N, 3)
    
    # Pairwise distances
    d = pdist(c)
    
    # Soft min of pairwise distances
    min_pair_soft = _soft_min(d, beta)
    
    # Boundary distances
    face_dists = 0.5 - np.abs(c)
    min_face_soft = _soft_min(face_dists, beta)
    
    # The radius is min(min_pair/2, min_face)
    r_soft = _soft_min(np.array([min_pair_soft / 2.0, min_face_soft]), beta * 0.5)
    
    return -r_soft

def _grad(x, beta):
    """Compute gradient of the objective function."""
    c = x.reshape(N, 3)
    
    # Pairwise distances
    i, j = np.triu_indices(N, 1)
    diff = c[i] - c[j]
    dist = np.sqrt(np.sum(diff**2, axis=1) + 1e-12)
    
    # Soft min weights
    d_min = np.min(dist)
    weights = np.exp(-beta * (dist - d_min))
    weights /= np.sum(weights)
    
    # Gradient of soft_min of pairwise distances
    grad_pair = np.zeros((N, 3))
    for k in range(len(i)):
        w = weights[k]
        # d(dist_k)/d(c_i) = -diff_k / dist_k
        # d(dist_k)/d(c_j) = diff_k / dist_k
        grad_pair[i[k]] -= w * diff[k] / dist[k]
        grad_pair[j[k]] += w * diff[k] / dist[k]
    
    # Boundary distances
    face_dists = 0.5 - np.abs(c)
    f_min = np.min(face_dists)
    weights_f = np.exp(-beta * (face_dists - f_min))
    weights_f /= np.sum(weights_f)
    
    # Gradient of soft_min of face distances
    grad_face = np.zeros((N, 3))
    # d(face_dists)/d(c) = -sign(c)
    for n in range(N):
        for k in range(3):
            grad_face[n, k] = -weights_f[n, k] * np.sign(c[n, k])
    
    # Now we need the gradient of the outer soft_min
    # r_soft = soft_min([min_pair/2, min_face])
    # Let a = min_pair/2, b = min_face
    # r_soft = a_min - log(sum(exp(-beta/2 * (a - a_min)), exp(-beta/2 * (b - a_min)))) / (beta/2)
    # This is getting complicated. Let's just use numerical gradient via L-BFGS-B.
    
    # Actually, let's return a dummy gradient and use numerical differentiation.
    return np.zeros_like(x)

def _build_initial_grid(N_local):
    """Build an initial grid configuration."""
    m = int(np.ceil(N_local ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N_local]
    return grid

def _optimize_from_init(init, beta_schedule, max_iter_per_beta=200):
    """
    Optimize from an initial configuration using a beta schedule.
    """
    current = init.copy()
    
    for beta in beta_schedule:
        x0 = current.flatten()
        
        def obj(x):
            return _objective(x, beta)
        
        try:
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            current = result.x.reshape(N, 3)
        except Exception:
            pass
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.RandomState(42)
    
    # Best solution found so far
    best_centres = None
    best_score = 0.0
    
    # Initial solution: grid
    init = _build_initial_grid(N)
    init_score = _score(init)
    if init_score > best_score:
        best_centres = init.copy()
        best_score = init_score
    
    # Beta schedule: start with low beta (smooth), increase to high beta (sharp)
    beta_schedule = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    # Run optimization
    refined = _optimize_from_init(init, beta_schedule, max_iter_per_beta=300)
    refined_score = _score(refined)
    if refined_score > best_score:
        best_centres = refined.copy()
        best_score = refined_score
    
    # Try multiple random restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial configuration
        init_r = _build_initial_grid(N)
        jitter = rng.uniform(-0.05, 0.05, size=init_r.shape)
        init_r += jitter
        init_r = np.clip(init_r, -0.49, 0.49)
        
        # Optimize
        refined_r = _optimize_from_init(init_r, beta_schedule, max_iter_per_beta=200)
        refined_r_score = _score(refined_r)
        if refined_r_score > best_score:
            best_centres = refined_r.copy()
            best_score = refined_r_score
    
    # Final polish with high beta
    if time.time() < deadline - 20 and best_centres is not None:
        final = _optimize_from_init(best_centres, [500.0, 1000.0, 2000.0], max_iter_per_beta=500)
        final_score = _score(final)
        if final_score > best_score:
            best_centres = final.copy()
            best_score = final_score
    
    # Ensure validity
    if best_centres is None:
        return _build_initial_grid(N)
    
    # Final clip to ensure inside cube
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
