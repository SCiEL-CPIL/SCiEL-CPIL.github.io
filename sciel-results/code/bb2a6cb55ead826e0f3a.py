import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
_TIME_BUDGET = 140.0  # Safety margin before 150s kill

def _energy_and_grad(x):
    """
    Compute logarithmic energy and Riemannian gradient (tangent-space) for points on S^2.
    x: (N, 3) array of points (will be normalized internally).
    Returns: (energy: float, grad: (N,3) array in tangent space)
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / norms

    # Pairwise squared distances
    # dist_sq[i,j] = ||x_i - x_j||^2 = 2 - 2 x_i·x_j
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 1e-30, out=dist_sq)

    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(dist_sq[iu] + 1e-30)
    energy = -np.sum(np.log(d_pairs))

    # Gradient: for E = sum_{i<j} -log(||x_i - x_j||)
    # dE/dx_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # weights[i,j] = 1/dist_sq[i,j] for i!=j, 0 for i==j
    weights = 1.0 / (dist_sq + 1e-30)
    np.fill_diagonal(weights, 0.0)

    w_sum = weights.sum(axis=1)       # (N,)
    w_x = weights @ x                 # (N,3)
    grad_euclid = x * w_sum[:, np.newaxis] - w_x

    # Project to tangent space: grad_t = grad_e - (grad_e · x) x
    g_dot_x = np.sum(grad_euclid * x, axis=1, keepdims=True)
    grad_tangent = grad_euclid - g_dot_x * x

    return float(energy), grad_tangent


def _retract(x, v):
    """
    Retraction from tangent space back to sphere.
    x: (N,3) on sphere, v: (N,3) tangent vector.
    Returns retracted points on sphere.
    """
    x_new = x + v
    norms = np.linalg.norm(x_new, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return x_new / norms

def _optimize_l_bfgs(x0, max_iter=5000, gtol=1e-14):
    """
    L-BFGS-B optimization on Euclidean coordinates with projection.
    """
    def fun_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        e, g = _energy_and_grad(x)
        return e, g.flatten()

    x0_flat = x0.flatten()
    res = minimize(
        fun_and_grad,
        x0_flat,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': max_iter,
            'ftol': 1e-20,
            'gtol': gtol,
            'maxls': 40,
            'maxcor': 30,
            'disp': False
        }
    )
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    return x_opt, res.fun


def _riemannian_grad_descent(x0, steps=5000, time_limit=None):
    """
    Riemannian gradient descent with backtracking line search.
    """
    x = x0.copy()
    e, g = _energy_and_grad(x)
    best_e = e
    best_x = x.copy()

    lr = 0.01
    for step in range(steps):
        if time_limit is not None and time.time() > time_limit:
            break

        # Backtracking line search
        alpha = lr
        for _ in range(30):
            x_trial = _retract(x, -alpha * g)
            e_trial, _ = _energy_and_grad(x_trial)
            if e_trial < e:
                break
            alpha *= 0.5

        x = x_trial
        e = e_trial

        if e < best_e:
            best_e = e
            best_x = x.copy()

        # Adaptive step size
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-15:
            break
        # Decrease step if no improvement
        if e_trial >= e:
            lr = max(alpha, 1e-6)
        else:
            lr = min(alpha * 1.5, 0.1)

    return best_x, best_e


def _polish_newton(x0, max_iter=100, time_limit=None):
    """
    Newton-like polish using Hessian approximation via finite differences of gradient.
    For small N this can give quadratic convergence near the optimum.
    """
    x = x0.copy()
    e, g = _energy_and_grad(x)

    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break

        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            break

        # Approximate Hessian via finite differences (expensive but for polish only)
        # Use a simpler approach: damped Newton with diagonal Hessian approximation
        # H_ii ≈ (g_i^2 + eps) for each coordinate
        # This is essentially a scaled gradient step

        # Better: use the structure. For log energy on sphere, the Hessian is complex.
        # Instead, do a few steps of Riemannian conjugate gradient.
        # For now, use a damped step with adaptive sizing.
        scale = g_norm / (N * 3)
        step_size = min(1.0 / (g_norm + 1e-10), 0.1)

        x_trial = _retract(x, -step_size * g)
        e_trial, g_trial = _energy_and_grad(x_trial)

        if e_trial < e:
            x = x_trial
            e = e_trial
            g = g_trial
        else:
            # Try smaller step
            x_trial = _retract(x, -0.5 * step_size * g)
            e_trial, g_trial = _energy_and_grad(x_trial)
            if e_trial < e:
                x = x_trial
                e = e_trial
                g = g_trial
            else:
                break

    return x, e

def _energy_eval(x):
    """
    High precision energy calculation for final evaluation using math.fsum.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / norms
    iu = np.triu_indices(N, k=1)
    diff = x[iu[0]] - x[iu[1]]
    d_pairs = np.sqrt(np.sum(diff**2, axis=1) + 1e-30)
    return -math.fsum(np.log(d_pairs).tolist())

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_BUDGET

    rng = np.random.default_rng(42)

    best_x = None
    best_e = float('inf')

    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # 2. Perturbed Fibonacci Spirals
    inits = [x_fib]
    for scale in [0.01, 0.05, 0.1, 0.3, 0.7, 1.5]:
        noise = rng.normal(0, scale, (N, 3))
        x_pert = x_fib + noise
        x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
        inits.append(x_pert)

    # 3. Random Initializations (more diverse)
    for _ in range(12):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)

    # 4. Structured initializations: two antipodal Fibonacci + perturbation
    for scale in [0.02, 0.1]:
        x_struct = x_fib.copy()
        # Flip every other point
        x_struct[::2] = -x_struct[::2]
        noise = rng.normal(0, scale, (N, 3))
        x_struct += noise
        x_struct /= np.linalg.norm(x_struct, axis=1, keepdims=True)
        inits.append(x_struct)

    # Phase 1: L-BFGS on all initializations
    n_inits = len(inits)
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 30.0:
            break

        try:
            # Allocate time proportionally
            time_left = deadline - time.time() - 30.0
            if time_left < 3.0:
                break
            # Spend roughly equal time per init, but cap per-init time
            t_per_init = min(8.0, max(2.0, time_left / max(1, n_inits - idx)))
            max_iter = int(5000 * t_per_init / 8.0)

            x_opt, e_opt = _optimize_l_bfgs(x0, max_iter=max_iter, gtol=1e-12)

            # Verify with high precision energy
            e_precise = _energy_eval(x_opt)

            if e_precise < best_e:
                best_e = e_precise
                best_x = x_opt
        except Exception:
            continue

    if best_x is None:
        best_x = x_fib
        best_e = _energy_eval(best_x)

    # Phase 2: Riemannian gradient descent polish on best
    if time.time() < deadline - 15.0:
        try:
            t_rg = deadline - time.time() - 15.0
            x_rg, e_rg = _riemannian_grad_descent(best_x, steps=8000, time_limit=start_time + t_rg)
            e_rg_precise = _energy_eval(x_rg)
            if e_rg_precise < best_e:
                best_e = e_rg_precise
                best_x = x_rg
        except Exception:
            pass

    # Phase 3: Final L-BFGS polish with tight tolerance
    if time.time() < deadline - 5.0:
        try:
            t_final = deadline - time.time() - 5.0
            max_iter_final = int(10000 * t_final / 10.0)
            x_final, e_final = _optimize_l_bfgs(best_x, max_iter=max_iter_final, gtol=1e-15)
            e_final_precise = _energy_eval(x_final)
            if e_final_precise < best_e:
                best_e = e_final_precise
                best_x = x_final
        except Exception:
            pass

    # Phase 4: Final Newton-like polish
    if time.time() < deadline - 2.0:
        try:
            t_newton = deadline - time.time() - 2.0
            x_newton, e_newton = _polish_newton(best_x, max_iter=50, time_limit=start_time + t_newton)
            e_newton_precise = _energy_eval(x_newton)
            if e_newton_precise < best_e:
                best_e = e_newton_precise
                best_x = x_newton
        except Exception:
            pass

    # Ensure valid output
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms

    return best_x
