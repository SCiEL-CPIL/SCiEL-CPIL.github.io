"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START) - 2.0  # 2s safety margin

def _energy(pts):
    """Logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e10
            total += math.log(1.0 / d)
    return total

def _energy_grad(pts):
    """Gradient of log(1/d) energy w.r.t. each point (in R^3)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_vec = pts[i] - pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-15:
                continue
            # d/dx_i log(1/|x_i - x_j|) = -d_vec / |x_i - x_j|^2
            grad[i] -= d_vec / (d * d)
    return grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _sphere_fun_grad(x_flat):
    """Objective and gradient for L-BFGS on flat vector; projects to sphere."""
    pts = _project_to_sphere(x_flat.reshape(N, 3))
    e = _energy(pts)
    g = _energy_grad(pts)
    # Project gradient onto tangent space: g_tangent = g - (g.x)x
    pts_t = pts.T  # (3, N)
    g_t = g.T  # (3, N)
    dot = np.sum(pts_t * g_t, axis=0)  # (N,)
    g_tangent = g - dot[:, None] * pts
    return e, g_tangent.flatten()

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _lbfgs_optimize(pts, max_iter=200, gtol=1e-8):
    """Run L-BFGS-B on the sphere using projected gradients."""
    x0 = pts.flatten()
    best_e = _energy(pts)
    best_pts = pts.copy()
    
    for attempt in range(3):
        if _time_left() < 5.0:
            break
        def fun_grad(x):
            e, g = _sphere_fun_grad(x)
            return e, g
        try:
            res = minimize(fun_grad, x0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': gtol,
                                   'maxfun': max_iter * 3})
            if res.success or res.fun < best_e:
                new_pts = _project_to_sphere(res.x.reshape(N, 3))
                new_e = _energy(new_pts)
                if new_e < best_e:
                    best_e = new_e
                    best_pts = new_pts
                x0 = res.x
        except Exception:
            break
    return best_pts, best_e

def _steepest_descent_sphere(pts, lr=0.01, max_iter=500):
    """Steepest descent on sphere with adaptive step size."""
    best_e = _energy(pts)
    best_pts = pts.copy()
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        g = _energy_grad(pts)
        # Project to tangent space
        dot = np.sum(pts * g, axis=1, keepdims=True)
        g_tan = g - dot * pts
        # Adaptive step size
        for trial in range(10):
            step = lr * (0.5 ** trial)
            new_pts = _project_to_sphere(pts - step * g_tan)
            new_e = _energy(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_pts = new_pts
                pts = new_pts
                break
        else:
            break
    return best_pts, best_e

def _simulated_annealing(pts, t_start=0.5, t_end=0.001, max_iter=2000):
    """Simulated annealing with random perturbations on the sphere."""
    best_e = _energy(pts)
    best_pts = pts.copy()
    cur_e = best_e
    rng = np.random.RandomState(123)
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        t = t_start * (t_end / t_start) ** (it / max_iter)
        # Pick 1-3 random points to perturb
        n_perturb = rng.randint(1, 4)
        idx = rng.choice(N, n_perturb, replace=False)
        new_pts = pts.copy()
        for i in idx:
            # Random perturbation on sphere
            noise = rng.randn(3) * 0.1
            new_pts[i] = _project_to_sphere((pts[i] + noise).reshape(1, 3))[0]
        new_e = _energy(new_pts)
        delta = new_e - cur_e
        if delta < 0 or rng.rand() < math.exp(-delta / max(t, 1e-10)):
            pts = new_pts
            cur_e = new_e
            if cur_e < best_e:
                best_e = cur_e
                best_pts = pts.copy()
    return best_pts, best_e

def construct():
    global _START
    _START = time.time()
    
    # Phase 1: Start with Fibonacci spiral and optimize
    pts = _fibonacci_spiral()
    pts, e = _lbfgs_optimize(pts, max_iter=300, gtol=1e-8)
    
    # Phase 2: Steepest descent polish
    pts, e = _steepest_descent_sphere(pts, lr=0.05, max_iter=1000)
    
    # Phase 3: More L-BFGS iterations
    if _time_left() > 10.0:
        pts, e = _lbfgs_optimize(pts, max_iter=500, gtol=1e-10)
    
    # Phase 4: Simulated annealing to escape local minima
    if _time_left() > 15.0:
        pts_sa, e_sa = _simulated_annealing(pts, t_start=0.3, t_end=0.0005, max_iter=3000)
        if e_sa < e:
            pts, e = pts_sa, e_sa
            # Polish after SA
            if _time_left() > 5.0:
                pts, e = _lbfgs_optimize(pts, max_iter=200, gtol=1e-10)
    
    # Phase 5: Random restarts if time remains
    if _time_left() > 20.0:
        for seed in [7, 13, 42, 99, 123]:
            if _time_left() < 10.0:
                break
            pts_r = _random_points(seed)
            pts_r, e_r = _lbfgs_optimize(pts_r, max_iter=200, gtol=1e-8)
            if e_r < e:
                pts, e = pts_r, e_r
                pts, e = _steepest_descent_sphere(pts, lr=0.05, max_iter=500)
                if _time_left() > 5.0:
                    pts, e = _lbfgs_optimize(pts, max_iter=200, gtol=1e-10)
    
    # Final polish
    if _time_left() > 3.0:
        pts, e = _steepest_descent_sphere(pts, lr=0.01, max_iter=2000)
    
    # Ensure all points are on unit sphere
    pts = _project_to_sphere(pts)
    
    # Validate: no points too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                # Nudge apart
                pts[j] = _project_to_sphere(pts[j] + 1e-8 * np.random.randn(3))[0]
    
    return pts
# EVOLVE-BLOCK-END
