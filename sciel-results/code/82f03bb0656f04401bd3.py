"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Best known radius from Eckard Specht
R_KNOWN = 0.1113823475125


def _score(c):
    """Compute the packing radius for centres c of shape (N,3)."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Boundary constraint: min_i min_k (0.5 - |c_ik|)
    bound = np.min(0.5 - np.abs(c))
    if bound < 0.0:
        return 0.0
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d) / 2.0
    return float(min(bound, pair_min))


def _objective(c, t):
    """
    Smooth surrogate objective: minimize -t + penalty for violations.
    t is a variable representing the radius. We want to maximize t subject to
    t <= 0.5 - |c_ik| for all i,k and t <= |c_i - c_j|/2 for all i<j.
    We use a smooth penalty: sum of soft violations.
    """
    c = c[:N]
    t = c[N]  # last element is t

    # Boundary violations: t - (0.5 - |c_ik|) should be <= 0
    bound_margin = 0.5 - np.abs(c)  # (N,3)
    bound_viol = np.maximum(0, t - np.min(bound_margin, axis=1))  # (N,)
    
    # Pairwise violations: t - |c_i - c_j|/2 should be <= 0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_viol = np.maximum(0, t - d / 2.0)
    
    # Smooth penalty (quadratic)
    penalty = np.sum(bound_viol ** 2) + np.sum(pair_viol ** 2)
    
    # We want to maximize t, so minimize -t + penalty
    return -t + penalty


def _objective_with_grad(c, t):
    """Objective with gradient for L-BFGS-B."""
    c = c[:N]
    t = c[N]
    
    bound_margin = 0.5 - np.abs(c)
    min_bound_idx = np.argmin(bound_margin, axis=1)
    min_bound_vals = np.min(bound_margin, axis=1)
    
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_dist = d / 2.0
    
    # Compute gradients numerically would be too slow; use finite differences via scipy
    # Actually, let's just use a simple approach: minimize with finite differences
    return -t + np.sum(np.maximum(0, t - min_bound_vals) ** 2) + np.sum(np.maximum(0, t - pair_dist) ** 2)


def _initial_grid():
    """Create an initial grid packing."""
    m = 5  # 5x5x4 = 100 points
    # Use a 5x5x4 grid
    xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, 5)
    ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, 5)
    zs = np.linspace(-0.5 + 0.125, 0.5 - 0.125, 4)
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    
    centres = np.array(centres[:N])
    return centres


def _local_optimize(c0, max_iter=500, tol=1e-10):
    """
    Local optimization using a smooth formulation.
    We optimize over all 3N+1 variables (centres + t).
    """
    n_vars = 3 * N + 1
    x0 = np.concatenate([c0.flatten(), [0.1]])
    
    def obj(x):
        c = x[:3*N].reshape(N, 3)
        t = x[3*N]
        
        # Boundary
        bound_margin = 0.5 - np.abs(c)
        bound_viol = np.maximum(0, t - np.min(bound_margin, axis=1))
        
        # Pairwise
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        pair_viol = np.maximum(0, t - d / 2.0)
        
        return -t + 10.0 * (np.sum(bound_viol ** 2) + np.sum(pair_viol ** 2))
    
    result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    
    c_opt = result.x[:3*N].reshape(N, 3)
    t_opt = result.x[3*N]
    
    # Project back to feasible region: ensure boundaries are respected
    c_opt = np.clip(c_opt, -0.5 + t_opt, 0.5 - t_opt)
    
    return c_opt, _score(c_opt)


def _perturb(c, scale=0.01):
    """Add random perturbation."""
    c_new = c + scale * np.random.randn(N, 3)
    return np.clip(c_new, -0.49, 0.49)


def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Start with a good initial guess
    np.random.seed(42)
    
    # Try multiple random initializations and local optimizations
    best_c = _initial_grid()
    best_score = _score(best_c)
    
    # Try several random restarts
    n_restarts = 0
    while time.time() - start_time < deadline - 30.0:
        n_restarts += 1
        
        # Random initialization
        c0 = np.random.uniform(-0.4, 0.4, size=(N, 3))
        
        # Local optimization
        c_opt, score = _local_optimize(c0, max_iter=200, tol=1e-9)
        
        if score > best_score:
            best_score = score
            best_c = c_opt.copy()
        
        # Also try perturbing the current best
        if n_restarts > 1:
            c_pert = _perturb(best_c, scale=0.005)
            c_opt2, score2 = _local_optimize(c_pert, max_iter=100, tol=1e-9)
            if score2 > best_score:
                best_score = score2
                best_c = c_opt2.copy()
    
    # Final polish with tighter tolerance if time permits
    if time.time() - start_time < deadline - 10.0:
        c_polish, score_polish = _local_optimize(best_c, max_iter=1000, tol=1e-12)
        if score_polish > best_score:
            best_score = score_polish
            best_c = c_polish
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
