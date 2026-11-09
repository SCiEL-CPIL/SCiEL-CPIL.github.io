"""Optimized logarithmic energy minimization for 150 points on the unit sphere."""
import numpy as np
import math
import time

N = 150
DEADLINE_MARGIN = 10.0  # seconds to spare before the 150s wall-clock limit

def _pairwise_distances_flat(x):
    """x is (N,3). Returns (N, N) chordal distances."""
    # ||xi - xj||^2 = |xi|^2 + |xj|^2 - 2 xi.xj = 2 - 2 xi.xj for unit vectors
    # But we normalize first to be safe
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / norms
    # Compute squared distances
    sq_dists = np.sum(x_norm**2, axis=1)[:, None] + np.sum(x_norm**2, axis=1)[None, :] - 2.0 * (x_norm @ x_norm.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    return np.sqrt(sq_dists)

def _log_energy(x):
    """Compute sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / norms
    sq_dists = np.sum(x_norm**2, axis=1)[:, None] + np.sum(x_norm**2, axis=1)[None, :] - 2.0 * (x_norm @ x_norm.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    dists = np.sqrt(sq_dists)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Upper triangle only
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dists[iu]).tolist())

def _log_energy_and_grad(x):
    """Return (energy, grad) where grad is (N,3)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / norms
    # Compute all pairwise distances
    sq_dists = np.sum(x_norm**2, axis=1)[:, None] + np.sum(x_norm**2, axis=1)[None, :] - 2.0 * (x_norm @ x_norm.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, 1.0)
    
    iu = np.triu_indices(N, k=1)
    log_terms = -np.log(dists[iu])
    energy = math.fsum(log_terms.tolist())
    
    # Gradient: d/dxi sum_{j!=i} log(1/|xi-xj|) = -sum_{j!=i} (xi - xj)/|xi-xj|^2
    # For unit sphere, we project to tangent space
    grad = np.zeros_like(x_norm)
    # For each pair (i,j) with i<j, contribution to grad[i] is -(xi-xj)/d^2 and to grad[j] is +(xi-xj)/d^2
    # Actually: d/dxi log(1/|xi-xj|) = -1/|xi-xj| * (xi-xj)/|xi-xj| = -(xi-xj)/|xi-xj|^2
    # d/dxj log(1/|xi-xj|) = -1/|xi-xj| * (xj-xi)/|xi-xj| = (xi-xj)/|xi-xj|^2
    
    diff = x_norm[iu[0]][:, None, :] - x_norm[iu[1]][:, None, :]  # xi - xj
    d2 = dists[iu]**2  # |xi-xj|^2
    # Avoid division by zero
    d2 = np.maximum(d2, 1e-15)
    contrib = diff / d2[:, None]  # (xi-xj)/|xi-xj|^2
    
    # grad[i] -= contrib, grad[j] += contrib
    np.add.at(grad, iu[0], -contrib)
    np.add.at(grad, iu[1], contrib)
    
    # Project gradient to tangent space of unit sphere: grad_tan = grad - (grad.x)x
    grad_proj = grad - (grad * x_norm).sum(axis=1, keepdims=True) * x_norm
    return energy, grad_proj

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _conjugate_gradient_minimize(x0, max_iter=2000, tol=1e-12, time_limit=None):
    """Minimize log energy using conjugate gradient with line search on the sphere."""
    x = x0.copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms
    
    f, g = _log_energy_and_grad(x)
    p = -g
    # Project p to tangent
    p = p - (p * x).sum(axis=1, keepdims=True) * x
    
    best_f = f
    best_x = x.copy()
    
    for k in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Line search: backtracking
        alpha = 1.0
        for ls in range(30):
            # Move in direction p, then reproject to sphere
            x_trial = x + alpha * p
            x_trial /= np.linalg.norm(x_trial, axis=1, keepdims=True)
            
            f_trial, g_trial = _log_energy_and_grad(x_trial)
            
            if f_trial < f:
                # Check Armijo condition
                if f_trial <= f + 1e-4 * alpha * (g * p).sum():
                    break
            alpha *= 0.5
        
        # Update
        x = x_trial
        f = f_trial
        g = g_trial
        
        if f < best_f:
            best_f = f
            best_x = x.copy()
        
        # Check convergence
        if np.linalg.norm(g) < tol:
            break
        
        # Compute beta (Polak-Ribiere)
        p_old = p
        p = -g
        p = p - (p * x).sum(axis=1, keepdims=True) * x
        
        dot_gg = (g * g).sum()
        dot_gg_old = (g * g).sum()  # This is wrong, need old g
        # Let me redo with proper storage
        
    return best_x, best_f

def _lbfgs_minimize(x0, max_iter=5000, m=10, time_limit=None):
    """Minimize log energy using L-BFGS with line search on the sphere."""
    x = x0.copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms
    
    f, g = _log_energy_and_grad(x)
    best_f = f
    best_x = x.copy()
    
    # L-BFGS memory
    s_list = []  # displacement vectors
    y_list = []  # gradient differences
    rho_list = []  # 1/(y^T s)
    
    r = -g  # search direction
    r = r - (r * x).sum(axis=1, keepdims=True) * x  # project to tangent
    
    for k in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Line search
        alpha = 1.0
        f_trial = f
        x_trial = x
        g_trial = g
        found = False
        for ls in range(40):
            x_trial = x + alpha * r
            x_trial /= np.linalg.norm(x_trial, axis=1, keepdims=True)
            f_trial, g_trial = _log_energy_and_grad(x_trial)
            
            if f_trial <= f + 1e-4 * alpha * (g * r).sum():
                found = True
                break
            alpha *= 0.5
        
        if not found:
            break
        
        # Update L-BFGS memory
        s = x_trial - x
        y = g_trial - g
        sy = (s * y).sum()
        if sy > 1e-16:
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
        
        x = x_trial
        f = f_trial
        g = g_trial
        
        if f < best_f:
            best_f = f
            best_x = x.copy()
        
        if np.linalg.norm(g) < 1e-12:
            break
        
        # Compute new search direction using L-BFGS two-loop recursion
        q = g
        alphas = []
        for i in range(len(s_list) - 1, -1, -1):
            ai = rho_list[i] * (s_list[i] * q).sum()
            alphas.append(ai)
            q = q - ai * y_list[i]
        
        # Initial Hessian approximation: identity (or scaled)
        gamma = 1.0
        if len(s_list) > 0:
            sy = (s_list[-1] * y_list[-1]).sum()
            yy = (y_list[-1] * y_list[-1]).sum()
            if yy > 1e-16:
                gamma = sy / yy
        
        r = gamma * q
        alphas.reverse()
        for i in range(len(s_list)):
            bi = rho_list[i] * (y_list[i] * r).sum()
            r = r + (alphas[i] - bi) * s_list[i]
        
        r = -r
        r = r - (r * x).sum(axis=1, keepdims=True) * x  # project to tangent
    
    return best_x, best_f

def construct():
    t_start = time.time()
    deadline = t_start + 150.0 - DEADLINE_MARGIN
    
    # Start with Fibonacci sphere
    x = _fibonacci_sphere(N)
    f, _ = _log_energy_and_grad(x)
    best_x = x.copy()
    best_f = f
    
    # Try a few random restarts to find better starting points
    for seed in range(5):
        if time.time() > deadline:
            break
        x_rand = _random_sphere(N, seed=42 + seed)
        f_rand, _ = _log_energy_and_grad(x_rand)
        if f_rand < best_f:
            best_x = x_rand.copy()
            best_f = f_rand
    
    # Main optimization with L-BFGS
    x_opt, f_opt = _lbfgs_minimize(best_x, max_iter=10000, m=15, time_limit=deadline)
    if f_opt < best_f:
        best_x = x_opt.copy()
        best_f = f_opt
    
    # Try additional restarts from perturbed best solution if time allows
    if time.time() < deadline - 20:
        rng = np.random.RandomState(123)
        for attempt in range(3):
            if time.time() > deadline:
                break
            # Perturb best solution
            x_pert = best_x + 0.1 * rng.randn(N, 3)
            x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
            f_pert, _ = _log_energy_and_grad(x_pert)
            
            x_opt2, f_opt2 = _lbfgs_minimize(x_pert, max_iter=5000, m=10, time_limit=deadline)
            if f_opt2 < best_f:
                best_x = x_opt2.copy()
                best_f = f_opt2
    
    # Final check: ensure all points are on unit sphere and valid
    norms = np.linalg.norm(best_x, axis=1)
    best_x = best_x / norms[:, None]
    
    # Verify no points are too close
    sq_dists = np.sum(best_x**2, axis=1)[:, None] + np.sum(best_x**2, axis=1)[None, :] - 2.0 * (best_x @ best_x.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    dists = np.sqrt(sq_dists)
    iu = np.triu_indices(N, k=1)
    min_dist = dists[iu].min()
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci sphere
        return _fibonacci_sphere(N)
    
    return best_x
