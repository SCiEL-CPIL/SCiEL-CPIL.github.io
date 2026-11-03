"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_dists(c):
    """Compute pairwise distances for N points in 3D. Returns (N, N) matrix."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def _score(c):
    """Compute the packing radius for centres c in [-0.5, 0.5]^3."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside cube
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    # Wall distance
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    d = _pairwise_dists(c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    if min_pair <= 0:
        return 0.0
    return min(min_pair / 2.0, min_wall)

def _soft_energy(c, r, beta=50.0):
    """
    Smooth surrogate: penalize pairs closer than 2r and walls closer than r.
    Uses log-sum-exp style soft-min.
    Returns (energy, gradient).
    """
    c = np.asarray(c, dtype=np.float64)
    # Pairwise: want all distances >= 2r
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    np.fill_diagonal(d, np.inf)
    
    # Soft penalty for pairs: sum of softplus(2r - d_ij) for i<j
    # softplus(x) = log(1 + exp(x)) is smooth approximation of max(0, x)
    # We want to penalize when d < 2r, i.e., 2r - d > 0
    x_pair = 2.0 * r - d
    # Use only upper triangle
    iu = np.triu_indices(N, k=1)
    x_pair_u = x_pair[iu]
    
    # Soft sum: (1/beta) * log(sum(exp(beta * x))) approximates max(x)
    # But we want sum of softplus, not soft-max. Let's use sum of log(1+exp(beta*x))/beta
    # Actually, for a smooth penalty, let's use: sum_i softplus(beta * x_i) / beta
    # This is differentiable and approximates sum of max(0, x_i) as beta -> inf
    
    # To avoid overflow, use stable softplus
    def softplus(x):
        # log(1 + exp(x)) stable
        return np.where(x > 20, x, np.log1p(np.exp(x)))
    
    energy_pair = np.sum(softplus(beta * x_pair_u)) / beta
    grad_pair = np.zeros_like(c)
    # d/dx softplus(beta*x) = beta * sigmoid(beta*x) / beta = sigmoid(beta*x)
    # sigmoid(x) = 1/(1+exp(-x))
    sig = 1.0 / (1.0 + np.exp(-beta * x_pair_u))
    # Gradient wrt c: for pair (i,j), d = |c_i - c_j|
    # d(softplus(2r - d))/dc_i = -sigmoid(beta*(2r-d)) * (c_i - c_j) / d
    # We need to accumulate this
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d_ij = d[i, j]
        if d_ij < 1e-12:
            continue
        factor = -sig[idx] / d_ij
        grad_pair[i] += factor * (c[i] - c[j])
        grad_pair[j] += -factor * (c[i] - c[j])
    
    # Wall penalty: want each coord in [-0.5, 0.5], so 0.5 - |c_ik| >= r
    # Penalty: for each point and each dim, softplus(beta * (r - (0.5 - |c_ik|)))
    # = softplus(beta * (r + |c_ik| - 0.5))
    energy_wall = 0.0
    grad_wall = np.zeros_like(c)
    for i in range(N):
        for k in range(3):
            val = c[i, k]
            # distance to wall: 0.5 - |val|
            # penalty when 0.5 - |val| < r, i.e., |val| > 0.5 - r
            x_wall = r + np.abs(val) - 0.5
            energy_wall += softplus(beta * x_wall) / beta
            sig_w = 1.0 / (1.0 + np.exp(-beta * x_wall))
            # d/dc_i,k of |c_i,k| is sign(c_i,k)
            if val > 0:
                grad_wall[i, k] += sig_w
            elif val < 0:
                grad_wall[i, k] -= sig_w
            # else: val == 0, subgradient, set to 0
    
    total_energy = energy_pair + energy_wall
    total_grad = grad_pair + grad_wall
    return total_energy, total_grad

def _hard_score_fast(c):
    """Fast hard score computation."""
    c = np.asarray(c, dtype=np.float64)
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_pair / 2.0, min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start from a good initial configuration
    # Use a body-centered cubic-like arrangement or just a jittered grid
    np.random.seed(42)
    
    # Initialize with a grid plus small random jitter
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Take first N points
    c = grid[:N].copy()
    # Add small random jitter to break symmetry
    c += np.random.randn(N, 3) * 0.001
    
    # Clamp to cube
    c = np.clip(c, -0.49, 0.49)
    
    best_c = c.copy()
    best_score = _score(best_c)
    
    # Phase 1: Increase radius using smooth energy minimization
    # Start with a small radius and increase
    r_current = 0.05
    beta = 20.0
    
    while time.time() < deadline - 30.0:
        # Try to increase radius
        improved = False
        for attempt in range(5):
            if time.time() >= deadline - 30.0:
                break
            
            # Minimize energy at current radius
            def objective(x):
                e, g = _soft_energy(x, r_current, beta)
                return e, g
            
            try:
                result = minimize(
                    objective, c, jac=True, method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                c_new = result.x
                # Clamp
                c_new = np.clip(c_new, -0.49, 0.49)
                score_new = _score(c_new)
                
                if score_new > best_score:
                    best_c = c_new.copy()
                    best_score = score_new
                    improved = True
                c = c_new
            except Exception:
                pass
        
        if improved:
            r_current += 0.005
            beta = min(beta * 1.5, 200.0)
        else:
            # Try a slightly larger radius jump
            r_current += 0.01
            beta = min(beta * 1.5, 200.0)
    
    # Phase 2: Polish with hard constraints using local optimization
    # Use the best configuration and do a final polish
    c = best_c.copy()
    
    while time.time() < deadline - 5.0:
        # Compute current radius
        r_cur = _score(c)
        if r_cur <= 0:
            break
        
        # Try to increase radius slightly and minimize violations
        r_target = r_cur + 0.001
        beta_polish = 100.0
        
        def objective_polish(x):
            e, g = _soft_energy(x, r_target, beta_polish)
            return e, g
        
        try:
            result = minimize(
                objective_polish, c, jac=True, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_new = result.x
            c_new = np.clip(c_new, -0.49, 0.49)
            score_new = _score(c_new)
            
            if score_new > best_score:
                best_c = c_new.copy()
                best_score = score_new
            c = c_new
        except Exception:
            break
    
    # Final validation and return
    best_c = np.clip(best_c, -0.49, 0.49)
    final_score = _score(best_c)
    if final_score > 0:
        return best_c
    return best_c
# EVOLVE-BLOCK-END
