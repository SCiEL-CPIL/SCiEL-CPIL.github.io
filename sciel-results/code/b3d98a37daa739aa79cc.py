"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_and_grad(x):
    """
    Compute log-energy sum_{i<j} log(1/|x_i - x_j|) and its Euclidean gradient.
    x: (N, 3) array on the unit sphere.
    """
    n = x.shape[0]
    # Pairwise squared distances
    # |x_i - x_j|^2 = 2 - 2 x_i . x_j
    dots = x @ x.T  # (n, n)
    dist2 = 2.0 - 2.0 * dots
    # Avoid self-distance (diagonal is 0)
    np.fill_diagonal(dist2, 1.0)
    
    # log(1/|d|) = -log(|d|) = -0.5 log(dist2)
    # Energy = sum_{i<j} -0.5 log(dist2_ij)
    # Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 * 0.5
    # Actually: d/dx_i [ -0.5 log(2 - 2 x_i.x_j) ] = -0.5 * (1/(2-2 x_i.x_j)) * (-2 x_j) 
    # Wait, let's be careful.
    # f = -0.5 log(dist2)
    # d f / d x_i = -0.5 * (1/dist2) * d(dist2)/d x_i
    # dist2 = 2 - 2 x_i . x_j
    # d(dist2)/d x_i = -2 x_j
    # So d f / d x_i = -0.5 * (1/dist2) * (-2 x_j) = x_j / dist2
    # Total gradient for point i: sum_{j != i} x_j / dist2_ij
    # But wait, the energy is sum_{i<j} f_ij.
    # For a fixed i, it appears in terms (i, j) for j>i and (k, i) for k<i.
    # Term (i, j): f = -0.5 log(2 - 2 x_i.x_j). Grad w.r.t x_i: x_j / dist2_ij.
    # Term (k, i): f = -0.5 log(2 - 2 x_k.x_i). Grad w.r.t x_i: x_k / dist2_ki.
    # So total grad_i = sum_{j != i} x_j / dist2_ij.
    
    # Let's compute the matrix G where G_ij = x_j / dist2_ij for i != j, and 0 for i=j.
    # Then grad_i = sum_j G_ij = sum of rows of G? No, G_ij is a vector x_j.
    # So grad_i = sum_{j != i} x_j / dist2_ij.
    
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j inv_dist2[i, j] * x[j]
    grad = inv_dist2 @ x
    
    # Energy
    # sum_{i<j} -0.5 log(dist2_ij)
    # We can compute sum of all off-diagonal elements of -0.5 log(dist2), then divide by 2.
    log_dist2 = np.log(dist2)
    np.fill_diagonal(log_dist2, 0.0)
    energy = -0.5 * np.sum(log_dist2) / 2.0
    
    return energy, grad


def _riemannian_grad(x, euclidean_grad):
    """
    Project Euclidean gradient onto the tangent space of the sphere at x.
    Tangent space at x: { v | v . x = 0 }.
    Projection: v_tan = v - (v . x) x.
    """
    # x: (N, 3), euclidean_grad: (N, 3)
    # (v . x) for each i
    dot = np.sum(euclidean_grad * x, axis=1, keepdims=True)
    return euclidean_grad - dot * x


def _retract(x, grad):
    """
    Retraction: move along gradient descent direction and renormalize.
    Simple: x_new = normalize(x - step * grad).
    We'll use a fixed small step or adaptive.
    """
    return x


def _run_l_bfgs_b(start, max_iter=1000, tol=1e-12):
    """
    Run L-BFGS-B with Riemannian gradient.
    We embed the sphere constraint by working in R^3 but using the projected gradient.
    L-BFGS-B minimizes a smooth function in R^n. We minimize the energy function
    but use the Riemannian gradient. This is a common heuristic (Euclidean L-BFGS
    with Riemannian gradient) which works well for sphere constraints.
    """
    n = start.shape[0]
    x0 = start.flatten()
    
    def func_grad(v):
        x = v.reshape(n, 3)
        e, g = _energy_and_grad(x)
        rg = _riemannian_grad(x, g)
        return e, rg.flatten()
    
    res = minimize(func_grad, x0, jac=True, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
    x_opt = res.x.reshape(n, 3)
    # Renormalize to be safe
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    x_opt = x_opt / norms
    return x_opt, res.fun


def construct():
    deadline = time.time() + 110.0  # 110s to leave margin for 120s limit
    best_x = _fibonacci_spiral(N)
    best_e, _ = _energy_and_grad(best_x)
    
    # Try L-BFGS-B from Fibonacci spiral
    try:
        x_opt, e_opt = _run_l_bfgs_b(best_x, max_iter=500, tol=1e-14)
        if np.all(np.isfinite(x_opt)) and np.all(np.linalg.norm(x_opt, axis=1) > 1e-9):
            # Check minimum distance
            min_d = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(x_opt[i] - x_opt[j])
                    if d < min_d:
                        min_d = d
            if min_d > 1e-9:
                best_x = x_opt
                best_e = e_opt
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        # Random start
        x_rand = np.random.randn(N, 3)
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        try:
            x_opt, e_opt = _run_l_bfgs_b(x_rand, max_iter=200, tol=1e-12)
            if np.all(np.isfinite(x_opt)):
                min_d = np.inf
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(x_opt[i] - x_opt[j])
                        if d < min_d:
                            min_d = d
                if min_d > 1e-9 and e_opt < best_e:
                    best_x = x_opt
                    best_e = e_opt
        except Exception:
            pass
    
    # Final polish: run L-BFGS-B with tighter tolerance on best result
    if time.time() < deadline - 5.0:
        try:
            x_opt, e_opt = _run_l_bfgs_b(best_x, max_iter=200, tol=1e-15)
            if np.all(np.isfinite(x_opt)) and e_opt < best_e:
                best_x = x_opt
                best_e = e_opt
        except Exception:
            pass
    
    # Ensure normalization
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / norms
    
    return best_x

# EVOLVE-BLOCK-END
