"""Combined program for minimizing logarithmic energy of 150 points on the sphere."""
import numpy as np
import time
import math

N = 150
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

# EVOLVE-BLOCK-START

def _compute_energy_and_grad(X):
    """
    Compute logarithmic energy and its gradient for points on the sphere.
    X: (N, 3) array of points on unit sphere.
    Returns: energy (float), grad (N, 3) array.
    """
    n = len(X)
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Since dist is symmetric, sum over all i!=j is 2 * sum_{i<j}
    # So sum_{i<j} log(d) = 0.5 * sum_{i!=j} log(d)
    # log_energy = -0.5 * sum_{i!=j} log(dist[i,j])
    
    # Avoid log(0) on diagonal
    log_dist = np.zeros_like(dist)
    log_dist[mask] = np.log(dist[mask])
    
    energy = -0.5 * np.sum(log_dist)
    
    # Gradient: dE/dX_i = -sum_{j != i} (X_i - X_j) / ||X_i - X_j||^2
    # = -sum_{j != i} diff[i,j] / dist[i,j]^2
    dist_sq = dist**2
    dist_sq_safe = np.where(mask, dist_sq, 1.0)
    
    # grad_terms[i,j] = -diff[i,j] / dist_sq_safe[i,j]
    grad_terms = -diff / dist_sq_safe[:, :, np.newaxis]
    
    # Sum over j for each i
    grad = np.sum(grad_terms, axis=1)
    
    return energy, grad

def _normalize(X):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms

def _fibonacci_sphere(n, rot=None):
    """Generate n points on unit sphere via Fibonacci spiral, optionally rotated."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if rot is not None:
        pts = pts @ rot.T
    return _normalize(pts)

def _riemannian_gd(X_init, max_iters=10000, lr_init=0.01, lr_min=1e-12, time_budget=None, start_time=None):
    """
    Riemannian gradient descent with backtracking line search on the sphere.
    """
    if start_time is None:
        start_time = time.time()
    
    X = X_init.copy()
    E, grad = _compute_energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    # Project gradient to tangent space
    grad_dot_X = np.sum(grad * X, axis=1, keepdims=True)
    g_tan = grad - grad_dot_X * X
    
    lr = lr_init
    stall_count = 0
    
    for it in range(max_iters):
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        # Backtracking line search
        accepted = False
        current_lr = lr
        max_bt = 20
        for bt in range(max_bt):
            # Update: move in tangent direction, then project back to sphere
            X_new = X - current_lr * g_tan
            X_new = _normalize(X_new)
            
            E_new, grad_new = _compute_energy_and_grad(X_new)
            
            if E_new < E - 1e-15:
                X = X_new
                E = E_new
                grad = grad_new
                # Update tangent gradient
                grad_dot_X_new = np.sum(grad * X, axis=1, keepdims=True)
                g_tan = grad - grad_dot_X_new * X
                accepted = True
                stall_count = 0
                # Slightly increase learning rate if step was small
                if current_lr < lr * 0.5:
                    lr = min(current_lr * 1.5, 0.1)
                break
            else:
                current_lr *= 0.5
                if current_lr < lr_min:
                    break
        
        if not accepted:
            lr *= 0.5
            if lr < lr_min:
                break
            stall_count += 1
            if stall_count > 10:
                break
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        # Convergence check
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < 1e-12:
            break
    
    return best_X, best_E

def _simulated_annealing(X_init, max_iter=20000, T_init=0.1, T_min=1e-8, time_budget=None, start_time=None):
    """
    Simulated annealing on the sphere to escape local minima.
    Uses single-point moves with Rodrigues rotation.
    """
    if start_time is None:
        start_time = time.time()
    
    X = X_init.copy()
    E_current, _ = _compute_energy_and_grad(X)
    best_X = X.copy()
    E_best = E_current
    T = T_init
    n = len(X)
    
    for it in range(max_iter):
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        # Pick a random point to move
        i = np.random.randint(n)
        
        # Random perturbation angle
        angle = np.random.uniform(0.05, 0.3)
        
        # Random tangent direction
        tangent = np.random.randn(3)
        tangent -= np.dot(tangent, X[i]) * X[i]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-15:
            continue
        tangent /= tangent_norm
        
        # Rodrigues rotation of point i around tangent axis by angle
        v = tangent
        theta = angle
        x = X[i]
        x_new = (x * np.cos(theta) + 
                 np.cross(v, x) * np.sin(theta) + 
                 v * np.dot(v, x) * (1 - np.cos(theta)))
        x_new = _normalize(x_new.reshape(1, -1))[0]
        
        X_new = X.copy()
        X_new[i] = x_new
        
        E_new, _ = _compute_energy_and_grad(X_new)
        delta_E = E_new - E_current
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            X = X_new
            E_current = E_new
            if E_current < E_best:
                E_best = E_current
                best_X = X.copy()
        
        # Cool down
        T = max(T * 0.9997, T_min)
    
    return best_X, E_best

def _multi_start_optimize(X_init, time_budget, start_time):
    """
    Run a sequence of optimization phases: GD -> SA -> GD polish.
    Returns best points and energy found.
    """
    # Phase 1: Initial gradient descent
    gd1_budget = min(time_budget * 0.4, 8.0)
    X1, E1 = _riemannian_gd(X_init, max_iters=5000, lr_init=0.005,
                            time_budget=gd1_budget, start_time=start_time)
    
    # Phase 2: Simulated annealing
    sa_budget = min(time_budget * 0.3, 7.0)
    if (time.time() - start_time) + sa_budget < time_budget:
        X2, E2 = _simulated_annealing(X1, max_iter=10000, T_init=0.05,
                                      time_budget=sa_budget, start_time=start_time)
        if E2 < E1:
            X1, E1 = X2, E2
    
    # Phase 3: Final polish with gradient descent
    polish_budget = min(time_budget * 0.3, 6.0)
    remaining = time_budget - (time.time() - start_time)
    if remaining > 1.0:
        X3, E3 = _riemannian_gd(X1, max_iters=3000, lr_init=0.001,
                                time_budget=min(polish_budget, remaining - 1.0),
                                start_time=start_time)
        if E3 < E1:
            X1, E1 = X3, E3
    
    return X1, E1

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_X = None
    best_E = float('inf')
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Standard Fibonacci
    inits.append(_fibonacci_sphere(N))
    
    # 2. Rotated Fibonacci (several rotations)
    for k in range(6):
        Q, _ = np.linalg.qr(rng.randn(3, 3))
        inits.append(_fibonacci_sphere(N, rot=Q))
    
    # 3. Random points
    for k in range(4):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
    
    # Process each initialization
    num_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time for this restart
        time_per_restart = min(remaining / (num_inits - idx), 15.0)
        
        X_opt, E_opt = _multi_start_optimize(init_pts, time_per_restart, start_time)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    # Final high-precision polish if time remains
    if best_X is not None:
        remaining = deadline - (time.time() - start_time)
        if remaining > 3.0:
            X_final, E_final = _riemannian_gd(best_X, max_iters=5000, lr_init=0.0005,
                                              time_budget=min(remaining - 2.0, 10.0),
                                              start_time=start_time)
            if E_final < best_E:
                best_X = X_final
                best_E = E_final
    
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_X = _normalize(best_X)
    return best_X

# EVOLVE-BLOCK-END
