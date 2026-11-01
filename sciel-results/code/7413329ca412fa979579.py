import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _energy_and_grad(x):
    """
    x: shape (N, 3) on unit sphere (or close to it)
    Returns energy and gradient in R^3 (not projected).
    Energy: sum_{i<j} log(1/||xi - xj||)
    Gradient: dE/dx_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    energy = -0.5 * np.sum(np.log(dist[mask]))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    return energy, grad

def _tangent_project(x, g):
    """Project gradient g onto tangent space at x (pointwise)."""
    # g is (N,3), x is (N,3)
    # tangent component: g - (g.x) * x  (since x is unit)
    gx = np.einsum('ij,ij->i', g, x)
    return g - gx[:, np.newaxis] * x

def _retract(x, step):
    """Retract x + step back to sphere."""
    new_x = x + step
    norms = np.linalg.norm(new_x, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-15, 1.0, norms)
    return new_x / norms

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

def _random_sphere(seed):
    rng = np.random.RandomState(seed)
    x = rng.randn(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

def _icosahedron_based():
    """Generate points based on icosahedron vertices and face subdivisions."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Use Fibonacci as base, but perturb some points
    x = _fibonacci_spiral()
    rng = np.random.RandomState(42)
    for _ in range(3):
        idx = rng.choice(N, size=10, replace=False)
        x[idx] += rng.randn(10, 3) * 0.1
        norms = np.linalg.norm(x[idx], axis=1, keepdims=True)
        x[idx] = x[idx] / norms
    return x

def _optimize_lbfgs(x0, max_iter=200, tol=1e-12, deadline=None):
    """
    L-BFGS optimization with tangent-space projection and retraction.
    """
    x = x0.copy()
    best_x = x0.copy()
    best_e, best_g = _energy_and_grad(x)
    
    # Maintain L-BFGS history
    m = 10  # number of corrections to store
    S = []  # steps
    Y = []  # gradient differences
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(x)
        if e < best_e:
            best_e = e
            best_x = x.copy()
        
        # Project gradient to tangent space
        g_tan = _tangent_project(x, g)
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
        
        # L-BFGS two-loop recursion to compute search direction
        # p = H * g_tan where H is inverse Hessian approximation
        # For simplicity, use a basic L-BFGS implementation
        
        # Two-loop recursion
        r = g_tan.copy()
        alphas = []
        betas = []
        for S_i, Y_i in zip(reversed(S), reversed(Y)):
            alpha = np.dot(S_i.flatten(), r.flatten())
            beta = np.dot(Y_i.flatten(), r.flatten())
            alphas.append(alpha)
            r = r - beta * Y_i
            betas.append(beta)
        
        # Initial Hessian approximation: gamma * I
        if Y:
            gamma = np.dot(Y[-1].flatten(), S[-1].flatten()) / np.dot(Y[-1].flatten(), Y[-1].flatten())
        else:
            gamma = 1.0 / g_norm
        
        q = gamma * r
        alphas.reverse()
        betas.reverse()
        for S_i, Y_i, alpha, beta in zip(S, Y, alphas, betas):
            q = q + (alpha - gamma * beta) * S_i
        
        # Line search with Armijo condition
        step_dir = -q
        step_norm = np.linalg.norm(step_dir)
        if step_norm < 1e-15:
            break
        
        # Normalize step direction for line search
        alpha = min(1.0, 1.0 / step_norm) * 0.5
        
        for ls_it in range(30):
            new_x = _retract(x, alpha * step_dir)
            new_e, _ = _energy_and_grad(new_x)
            
            # Armijo condition
            if new_e <= e + 1e-4 * alpha * np.dot(g_tan.flatten(), step_dir.flatten()):
                break
            alpha *= 0.5
        
        # Update L-BFGS history
        S_new = new_x - x
        Y_new = _energy_and_grad(new_x)[1] - g
        S_new_tan = _tangent_project(new_x, S_new)  # approximate
        Y_new_tan = _tangent_project(new_x, Y_new)
        
        S.append(S_new)
        Y.append(Y_new)
        if len(S) > m:
            S.pop(0)
            Y.pop(0)
        
        x = new_x
    
    return best_x, best_e

def _newton_polish(x, max_iter=50, deadline=None):
    """
    Newton-style polish using the Hessian of the Lagrangian.
    For points on the sphere, the constrained Hessian is:
    H_constrained = H - lambda * I (approximately)
    We use a simplified approach: compute the Hessian of the energy
    and solve for the step in the tangent space.
    """
    x = x.copy()
    best_x = x.copy()
    best_e, best_g = _energy_and_grad(x)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(x)
        if e < best_e:
            best_e = e
            best_x = x.copy()
        
        g_tan = _tangent_project(x, g)
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-14:
            break
        
        # Approximate Hessian in tangent space
        # For log energy, Hessian at point k:
        # H_kk = sum_{j!=k} [1/dist_kj^2 * (I - (x_k-x_j)(x_k-x_j)^T/dist_kj^2)]
        # H_kl = -1/dist_kl^2 * (I - (x_k-x_l)(x_k-x_l)^T/dist_kl^2) for k!=l
        # This is expensive, so use a diagonal approximation
        
        # Diagonal Hessian approximation
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Diagonal Hessian: sum of 1/dist^2 for each point
        h_diag = np.sum(1.0 / dist_sq, axis=1)
        # Subtract the self-term contribution
        h_diag -= 1.0  # diagonal was set to 1, so 1/1 = 1
        
        # Solve H * step = -g_tan in tangent space
        # Use diagonal approximation
        step = -g_tan / h_diag[:, np.newaxis]
        
        # Project step to tangent space
        step = _tangent_project(x, step)
        
        # Damped Newton step
        step_norm = np.linalg.norm(step)
        if step_norm > 0.1:
            step *= 0.1 / step_norm
        
        new_x = _retract(x, step)
        new_e, _ = _energy_and_grad(new_x)
        
        if new_e < e:
            x = new_x
        else:
            # Halve the step
            step *= 0.5
            new_x = _retract(x, step)
            new_e, _ = _energy_and_grad(new_x)
            if new_e < e:
                x = new_x
            else:
                break
    
    return best_x, best_e

def _ultra_polish(x, max_iter=100, deadline=None):
    """
    Ultra-fine polish with very small steps and tight tolerance.
    """
    x = x.copy()
    best_x = x.copy()
    best_e, best_g = _energy_and_grad(x)
    
    step_scale = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(x)
        if e < best_e:
            best_e = e
            best_x = x.copy()
        
        g_tan = _tangent_project(x, g)
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-16:
            break
        
        # Gradient descent with adaptive step
        step = -g_tan * step_scale
        new_x = _retract(x, step)
        new_e, _ = _energy_and_grad(new_x)
        
        if new_e < e:
            x = new_x
            step_scale = min(step_scale * 1.1, 0.1)
        else:
            step_scale *= 0.5
            if step_scale < 1e-10:
                break
    
    return best_x, best_e

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate diverse initial guesses
    inits = []
    inits.append(_fibonacci_spiral())
    inits.append(_icosahedron_based())
    
    # Random initializations
    for seed in range(12):
        inits.append(_random_sphere(seed))
    
    # Perturbed Fibonacci
    rng = np.random.RandomState(99)
    for _ in range(5):
        x = _fibonacci_spiral()
        x += rng.randn(N, 3) * 0.05
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        inits.append(x / norms)
    
    # Deduplicate inits
    unique_inits = [inits[0]]
    for x in inits[1:]:
        is_dup = False
        for y in unique_inits:
            if np.linalg.norm(x - y, axis=1).max() < 0.3:
                is_dup = True
                break
        if not is_dup:
            unique_inits.append(x)
    
    # Optimize each initialization
    best_x = None
    best_e = float('inf')
    
    for x0 in unique_inits:
        if time.time() > deadline:
            break
        
        # L-BFGS optimization
        x_opt, e_opt = _optimize_lbfgs(x0, max_iter=150, tol=1e-12, deadline=deadline)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
        
        # Newton polish
        if time.time() < deadline:
            x_pol, e_pol = _newton_polish(x_opt, max_iter=30, deadline=deadline)
            if e_pol < best_e:
                best_e = e_pol
                best_x = x_pol.copy()
        
        # Ultra polish
        if time.time() < deadline:
            x_ultra, e_ultra = _ultra_polish(x_pol, max_iter=80, deadline=deadline)
            if e_ultra < best_e:
                best_e = e_ultra
                best_x = x_ultra.copy()
    
    if best_x is None:
        best_x = _fibonacci_spiral()
    
    # Final normalization
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_x = best_x / norms
    
    # Final check: ensure no points are too close
    # If any pair is closer than 1e-9, slightly perturb
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        # Perturb the closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        best_x[i] += 1e-8 * np.random.RandomState(123).randn(3)
        best_x[i] = best_x[i] / np.linalg.norm(best_x[i])
    
    return best_x
