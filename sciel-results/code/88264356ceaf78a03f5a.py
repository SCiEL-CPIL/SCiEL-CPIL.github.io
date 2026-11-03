"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _smooth_energy(P, r, beta=50.0):
    """Smooth log-sum-exp energy function for differentiable optimization."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    violations_cc = 2.0 * r - dist
    dist_origin = np.linalg.norm(P, axis=1)
    violations_cb = dist_origin + r - 1.0
    
    def _lse(v, beta):
        v_scaled = beta * v
        v_max = np.max(v_scaled)
        if not np.isfinite(v_max) or v_max == -np.inf:
            return 0.0
        return (1.0 / beta) * (v_max + np.log(np.sum(np.exp(v_scaled - v_max))))
    
    E_cc = _lse(violations_cc, beta)
    E_cb = _lse(violations_cb, beta)
    return E_cc + E_cb

def _smooth_energy_grad(P, r, beta=50.0):
    """Gradient of the smooth log-sum-exp energy."""
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    violations_cc = 2.0 * r - dist
    v_scaled_cc = beta * violations_cc
    v_max_cc = np.max(v_scaled_cc)
    if np.isfinite(v_max_cc) and v_max_cc != -np.inf:
        weights_cc = np.exp(v_scaled_cc - v_max_cc)
        weights_cc /= np.sum(weights_cc)
        inv_dist = np.zeros_like(dist)
        np.divide(1.0, dist, out=inv_dist, where=dist > 0)
        
        W_cc = weights_cc * inv_dist
        sum_W = np.sum(W_cc, axis=1)
        term1 = (P.T @ W_cc).T
        term2 = P * sum_W[:, np.newaxis]
        grad += term2 - term1
    
    dist_origin = np.linalg.norm(P, axis=1)
    violations_cb = dist_origin + r - 1.0
    v_scaled_cb = beta * violations_cb
    v_max_cb = np.max(v_scaled_cb)
    if np.isfinite(v_max_cb) and v_max_cb != -np.inf:
        weights_cb = np.exp(v_scaled_cb - v_max_cb)
        weights_cb /= np.sum(weights_cb)
        inv_dist_origin = np.zeros_like(dist_origin)
        np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
        grad += (weights_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad

def _optimize_positions(P_init, r_target, time_limit=5.0, beta=50.0):
    """Optimize positions P to minimize smooth energy for a fixed r_target."""
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_energy(P, r_target, beta)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_energy_grad(P, r_target, beta).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())

def _check_feasible(P, r, tol=1e-8):
    """Check if packing with radius r is feasible."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    dist_origin = np.linalg.norm(P, axis=1)
    max_origin = np.max(dist_origin)
    
    return (min_pair >= 2.0 * r - tol) and (max_origin + r <= 1.0 + tol)

def _get_hex_initial_points():
    """Initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98

def _get_ring_initialization(n_rings=5):
    """
    Ring-based initialization: place circles in concentric rings.
    Ring 0: center
    Ring k: k * 2 * r_est circles at radius k * r_est
    where r_est is an estimated circle radius.
    """
    # Estimate: for N circles in unit disk, r ~ 1/sqrt(N) * factor
    r_est = 0.125  # close to known best
    
    points = []
    
    # Center circle
    points.append([0.0, 0.0])
    
    # Rings
    ring_counts = []
    remaining = N - 1
    k = 1
    while remaining > 0 and k <= n_rings:
        # Number of circles on ring k: approximately 2*pi*k*r_est / (2*r_est) = pi*k
        # But we need integer counts that fit
        count_k = max(1, int(np.pi * k * 0.9))
        if count_k > remaining:
            count_k = remaining
        ring_counts.append((k, count_k))
        remaining -= count_k
        k += 1
    
    # If we still have remaining, add to last ring
    if remaining > 0 and ring_counts:
        last_k, last_count = ring_counts[-1]
        ring_counts[-1] = (last_k, last_count + remaining)
    
    for k, count in ring_counts:
        ring_radius = k * 2.0 * r_est * 0.5  # spacing = 2*r_est
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        for angle in angles:
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            # Clamp to stay inside unit circle
            r = np.hypot(x, y)
            if r > 0.95:
                scale = 0.95 / r
                x *= scale
                y *= scale
            points.append([x, y])
    
    # If we have fewer than N points, pad with small perturbations
    while len(points) < N:
        idx = len(points)
        angle = idx * 2.4
        rad = 0.1 + 0.05 * idx
        x = rad * np.cos(angle)
        y = rad * np.sin(angle)
        points.append([x, y])
    
    points = np.array(points[:N])
    return points

def _perturb(P, sigma=0.01):
    """Add small Gaussian perturbation to escape local minima."""
    return P + sigma * np.random.randn(*P.shape)

def _local_refine(P, r_target, n_steps=50, lr=0.001):
    """
    Simple gradient descent local refinement with adaptive learning rate.
    """
    P_cur = P.copy()
    for step in range(n_steps):
        E = _smooth_energy(P_cur, r_target, beta=100.0)
        g = _smooth_energy_grad(P_cur, r_target, beta=100.0)
        
        # Adaptive LR: reduce if energy increased
        P_new = P_cur - lr * g
        E_new = _smooth_energy(P_new, r_target, beta=100.0)
        
        if E_new < E:
            P_cur = P_new
            lr = min(lr * 1.05, 0.01)
        else:
            lr *= 0.5
        
        if lr < 1e-6:
            break
    
    return P_cur

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer for safety
    
    np.random.seed(42)
    
    # Generate multiple initializations
    inits = []
    inits.append(('hex', _get_hex_initial_points()))
    inits.append(('ring', _get_ring_initialization(5)))
    inits.append(('ring6', _get_ring_initialization(6)))
    
    # Also add perturbed versions
    for name, P in list(inits[:2]):
        inits.append((name + '_pert', _perturb(P, 0.02)))
    
    # Evaluate all initializations
    P_best = None
    score_best = -np.inf
    for name, P in inits:
        score = _compute_score(P)
        if score > score_best:
            score_best = score
            P_best = P.copy()
    
    # Binary search on r with multi-restart local optimization
    r_low = score_best
    r_high = 0.135  # Known best is ~0.1258, leave some margin
    
    for iteration in range(15):
        if time.time() > deadline - 20.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        best_P_for_r = None
        best_energy = np.inf
        
        # Multi-restart: try several initializations
        n_restarts = min(len(inits), 4 if iteration < 5 else 5)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20.0:
                break
            
            name, P_base = inits[restart % len(inits)]
            if restart < len(inits):
                P_init = P_base.copy()
            else:
                P_init = _perturb(P_base, 0.005 * (restart + 1))
            
            P_opt, E = _optimize_positions(P_init, r_mid, time_limit=4.0, beta=50.0)
            
            if E < best_energy:
                best_energy = E
                best_P_for_r = P_opt
        
        if best_P_for_r is None:
            break
        
        # Check feasibility
        if _check_feasible(best_P_for_r, r_mid):
            r_low = r_mid
            score = _compute_score(best_P_for_r)
            if score > score_best:
                P_best = best_P_for_r.copy()
                score_best = score
        else:
            r_high = r_mid
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        r_target = score_best
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            if restart == 0:
                P_init = P_best.copy()
            else:
                P_init = _perturb(P_best, 0.001 * restart)
            
            # Use higher beta for sharper optimization
            P_opt, E = _optimize_positions(P_init, r_target, time_limit=3.0, beta=100.0)
            
            # Additional local refinement
            P_refined = _local_refine(P_opt, r_target, n_steps=30, lr=0.0005)
            
            score = _compute_score(P_refined)
            if score > score_best:
                P_best = P_refined
                score_best = score
    
    return P_best
# EVOLVE-BLOCK-END
