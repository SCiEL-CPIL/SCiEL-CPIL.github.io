"""Combined program: Normalizing Flow Latent Optimization with Differentiable Top-K for 100 Spheres."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _normalizing_flow_transform(z, scale=0.3):
    """
    Transform latent variables z to sphere centers c using a simple normalizing flow.
    z: (N, 3) latent variables
    Returns: c: (N, 3) sphere centers in [-0.5, 0.5]^3
    
    Uses a tanh-based bijection to map R^3 -> (-0.5, 0.5)^3 with learned scaling.
    """
    # First, apply a simple affine transformation
    # z is standardized
    # Apply tanh to bound to (-1, 1), then scale to (-0.5, 0.5)
    c = scale * np.tanh(z)
    # Center in the cube
    c += 0.0  # Already centered
    return c

def _normalizing_flow_grad(z, c, scale=0.3):
    """
    Compute the Jacobian of the normalizing flow transformation.
    For c = scale * tanh(z), dc/dz = scale * (1 - tanh(z)^2) = scale * (1 - (c/scale)^2)
    """
    # dc/dz = scale * (1 - (c/scale)^2)
    # = scale - c^2/scale
    jacobian = scale - (c**2) / scale
    return jacobian

def _compute_topk_obj_grad(c, k, alpha):
    """
    Compute differentiable top-k approximation of the minimum distance.
    
    Uses a smooth approximation: for the k smallest distances,
    f(c) = -1/alpha * log( sum_{i=1}^{k} exp(alpha * d_i / k) )
    where d_i are the k smallest distances.
    
    This approximates the minimum of the k smallest distances.
    """
    # Get all pairwise distances and wall distances
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0
    
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
        0.5 - c[:, 2],
        0.5 + c[:, 2]
    ])
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Find the k smallest distances
    # For differentiability, use a soft-min approximation
    # Sort distances
    sorted_dists = np.sort(all_dists)
    k_dists = sorted_dists[:k]
    
    # Smooth min of the k smallest
    # Use log-sum-exp with alpha
    max_k = np.max(k_dists)
    exp_terms = np.exp(alpha * (k_dists - max_k))
    Z = np.sum(exp_terms)
    if Z == 0:
        return 0.0, np.zeros_like(c)
    
    obj_val = -(max_k + np.log(Z) / alpha)
    
    # Gradient with respect to c
    grad_c = np.zeros_like(c)
    
    # We need gradients of the k smallest distances
    # This is complex; use a simpler approach with soft-min over all distances
    # weighted by their rank
    
    # Alternative: use the smooth min over all distances with emphasis on small ones
    max_all = np.max(all_dists)
    exp_all = np.exp(-alpha * (all_dists - max_all))
    Z_all = np.sum(exp_all)
    if Z_all == 0:
        return 0.0, np.zeros_like(c)
    
    weights_all = exp_all / Z_all
    
    # Pairwise gradient
    exp_pair = exp_all[:len(pair_dists)]
    weights_pair = exp_pair / Z_all
    
    pair_dists_raw = d[i_idx, j_idx]
    pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)
    
    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj
    norms = pair_dists_raw_safe[:, np.newaxis]
    u = diff / norms
    w = weights_pair[:, np.newaxis]
    contrib = w * u / 2.0
    
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    # Wall gradient
    exp_wall = exp_all[len(pair_dists):]
    weights_wall = exp_wall / Z_all
    ww = weights_wall.reshape(6, N)
    
    grad_c[:, 0] += ww[0] - ww[1]
    grad_c[:, 1] += ww[2] - ww[3]
    grad_c[:, 2] += ww[4] - ww[5]
    
    return obj_val, grad_c

def _score(c):
    """Compute the exact packing radius."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    bound = np.min(0.5 - np.abs(c))
    if bound < 0.0:
        return 0.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d) / 2.0
    return float(min(bound, pair_min))

def _initial_grid():
    """Create a 5x5x4 grid."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    return np.array(centres[:N])

def _latent_to_centers(z, scale=0.3):
    """Convert latent variables to center coordinates."""
    c = scale * np.tanh(z)
    return c

def _centers_to_latent(c, scale=0.3):
    """Convert center coordinates to latent variables."""
    # Inverse of tanh
    c_clipped = np.clip(c / scale, -0.9999, 0.9999)
    z = np.arctanh(c_clipped)
    return z

def construct():
    start_time = time.time()
    deadline = 145.0
    
    np.random.seed(42)
    
    # Start with grid initialization
    c0 = _initial_grid()
    best_c = c0.copy()
    best_score = _score(c0)
    
    # Convert to latent space
    scale = 0.3
    z0 = _centers_to_latent(c0, scale)
    
    # Phase 1: Optimize in latent space with smooth objective
    alphas = [10, 20, 50, 100, 200, 500]
    
    for alpha in alphas:
        if time.time() - start_time > deadline - 15.0:
            break
        
        def obj_latent(z_flat):
            z = z_flat.reshape(N, 3)
            c = _latent_to_centers(z, scale)
            # Penalty for being outside bounds (should be small due to tanh)
            bound_penalty = np.sum(np.maximum(0, np.abs(c) - 0.5)**2)
            val, _ = _compute_topk_obj_grad(c, k=10, alpha=alpha)
            return val + bound_penalty
        
        def grad_latent(z_flat):
            z = z_flat.reshape(N, 3)
            c = _latent_to_centers(z, scale)
            _, grad_c = _compute_topk_obj_grad(c, k=10, alpha=alpha)
            # Backprop through tanh
            jacobian = scale - (c**2) / scale
            grad_z = grad_c / jacobian
            # Add penalty gradient
            bound_penalty_grad = 2.0 * np.maximum(0, np.abs(c) - 0.5) * np.sign(c)
            grad_z += bound_penalty_grad / jacobian
            return grad_z.flatten()
        
        try:
            result = minimize(
                obj_latent,
                z0.flatten(),
                jac=grad_latent,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * (3 * N),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if result.success:
                z_new = result.x.reshape(N, 3)
                c_new = _latent_to_centers(z_new, scale)
                c_new = np.clip(c_new, -0.5, 0.5)
                
                score_new = _score(c_new)
                if score_new > best_score:
                    best_score = score_new
                    best_c = c_new.copy()
                    z0 = z_new.copy()
        except Exception:
            pass
    
    # Phase 2: Direct optimization in center space with high alpha
    if time.time() - start_time < deadline - 10.0:
        alphas_direct = [500, 1000, 2000]
        
        for alpha in alphas_direct:
            if time.time() - start_time > deadline - 5.0:
                break
            
            def obj_direct(c_flat):
                c = c_flat.reshape(N, 3)
                val, _ = _compute_topk_obj_grad(c, k=5, alpha=alpha)
                return val
            
            def grad_direct(c_flat):
                c = c_flat.reshape(N, 3)
                _, g = _compute_topk_obj_grad(c, k=5, alpha=alpha)
                return g.flatten()
            
            try:
                result = minimize(
                    obj_direct,
                    best_c.flatten(),
                    jac=grad_direct,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (3 * N),
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success:
                    c_new = result.x.reshape(N, 3)
                    c_new = np.clip(c_new, -0.5, 0.5)
                    score_new = _score(c_new)
                    if score_new > best_score:
                        best_score = score_new
                        best_c = c_new.copy()
            except Exception:
                pass
    
    # Phase 3: Local refinement with random restarts
    if time.time() - start_time < deadline - 5.0:
        for _ in range(3):
            if time.time() - start_time > deadline - 2.0:
                break
            
            c_pert = best_c + 0.005 * np.random.randn(N, 3)
            c_pert = np.clip(c_pert, -0.49, 0.49)
            
            alpha = 2000
            
            def obj_pert(c_flat):
                c = c_flat.reshape(N, 3)
                val, _ = _compute_topk_obj_grad(c, k=3, alpha=alpha)
                return val
            
            def grad_pert(c_flat):
                c = c_flat.reshape(N, 3)
                _, g = _compute_topk_obj_grad(c, k=3, alpha=alpha)
                return g.flatten()
            
            try:
                result = minimize(
                    obj_pert,
                    c_pert.flatten(),
                    jac=grad_pert,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (3 * N),
                    options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success:
                    c_new = result.x.reshape(N, 3)
                    c_new = np.clip(c_new, -0.5, 0.5)
                    score_new = _score(c_new)
                    if score_new > best_score:
                        best_score = score_new
                        best_c = c_new.copy()
            except Exception:
                pass
    
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c

# EVOLVE-BLOCK-END
