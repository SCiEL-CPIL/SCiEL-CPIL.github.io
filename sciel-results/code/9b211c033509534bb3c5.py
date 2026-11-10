# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time

K = 10
N_MU = 70
N_PHI = 130

def _compute_quadrature():
    mu, w = np.polynomial.legendre.leggauss(N_MU)
    w *= 0.5
    phi = np.arange(N_PHI) * (2.0 * np.pi / N_PHI)
    return mu, w, phi

def _compute_basis_values():
    mu, w, phi = _compute_quadrature()
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing='ij')
    
    n_basis = 2 * K + 1
    Y = np.zeros((n_basis, N_MU, N_PHI))
    
    for m in range(-K, K + 1):
        idx = m + K
        abs_m = abs(m)
        P = lpmv(abs_m, K, mu_grid)
        N_m = np.sqrt((2 * K + 1) / (4.0 * np.pi) *
                      float(np.math.factorial(K - abs_m)) /
                      float(np.math.factorial(K + abs_m)))
        
        if m == 0:
            Y[idx] = N_m * P
        elif m > 0:
            Y[idx] = np.sqrt(2.0) * N_m * P * np.cos(m * phi_grid)
        else:
            Y[idx] = np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi_grid)
    
    return Y, mu, w, phi

def _evaluate(coeffs):
    Y, mu, w, phi = _compute_quadrature_and_basis()
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    dphi = 2.0 * np.pi / N_PHI
    l2_sq = np.sum(f ** 2 * w[:, None] * dphi) / (4.0 * np.pi)
    l4_sq = np.sum(f ** 4 * w[:, None] * dphi) / (4.0 * np.pi)
    if l2_sq < 1e-15:
        return 1e10
    l4 = np.sqrt(np.sqrt(l4_sq))
    l2 = np.sqrt(l2_sq)
    return l4 / l2

def _compute_quadrature_and_basis():
    global _Y_cached, _mu_cached, _w_cached, _phi_cached
    if _Y_cached is None:
        _Y_cached, _mu_cached, _w_cached, _phi_cached = _compute_basis_values()
    return _Y_cached, _mu_cached, _w_cached, _phi_cached

_Y_cached = None
_mu_cached = None
_w_cached = None
_phi_cached = None

def _objective_and_gradient(coeffs):
    Y, mu, w, phi = _compute_quadrature_and_basis()
    dphi = 2.0 * np.pi / N_PHI
    wphi = w[:, None] * dphi
    
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    f2 = f ** 2
    f4 = f2 ** 2
    
    l2_sq = np.sum(f2 * wphi) / (4.0 * np.pi)
    l4_sq = np.sum(f4 * wphi) / (4.0 * np.pi)
    
    if l2_sq < 1e-15:
        return 1e10, np.zeros_like(coeffs)
    
    l4 = np.sqrt(np.sqrt(l4_sq))
    l2 = np.sqrt(l2_sq)
    R = l4 / l2
    
    w4pi = wphi / (4.0 * np.pi)
    f3 = f ** 3
    sum_fw = np.tensordot(f * w4pi, Y, axes=([0, 1], [0, 1]))
    sum_f3w = np.tensordot(f3 * w4pi, Y, axes=([0, 1], [0, 1]))
    
    dl2_dc = sum_fw / l2
    dl4_dc = sum_f3w / l4
    
    grad_R = (dl4_dc / l2) - (l4 / l2 ** 2) * dl2_dc
    
    return R, grad_R

def _project_to_sphere(x):
    nrm = np.linalg.norm(x)
    if nrm < 1e-15:
        return x
    return x / nrm

def _line_search(x0, direction, R0):
    x = x0 - direction
    x = _project_to_sphere(x)
    R, _ = _objective_and_gradient(x)
    step = 1.0
    c = 1e-4
    max_iter = 30
    for _ in range(max_iter):
        if R < R0 - c * step * np.linalg.norm(direction):
            return x, R
        step *= 0.5
        x = x0 - step * direction
        x = _project_to_sphere(x)
        R, _ = _objective_and_gradient(x)
    return x, R

def _gradient_descent(R0, grad0, x0, max_iter=5000, tol=1e-10):
    x = x0.copy()
    R = R0
    grad = grad0
    for it in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break
        direction = grad
        x_new, R_new = _line_search(x, direction, R)
        if R_new >= R:
            break
        x = x_new
        R = R_new
        _, grad = _objective_and_gradient(x)
    return x, R

def _polish(x, R, max_iter=100, tol=1e-12):
    x = x.copy()
    for it in range(max_iter):
        R, grad = _objective_and_gradient(x)
        if np.linalg.norm(grad) < tol:
            break
        step = 1e-3 / (1 + it * 0.1)
        x_new = x - step * grad
        x_new = _project_to_sphere(x_new)
        R_new, _ = _objective_and_gradient(x_new)
        if R_new < R:
            x = x_new
            R = R_new
        else:
            step *= 0.5
            x_new = x - step * grad
            x_new = _project_to_sphere(x_new)
            R_new, _ = _objective_and_gradient(x_new)
            if R_new < R:
                x = x_new
                R = R_new
    return x, R

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0
    
    x0 = np.ones(2 * K + 1, dtype=np.float64)
    x0 = _project_to_sphere(x0)
    
    R0, grad0 = _objective_and_gradient(x0)
    
    x, R = _gradient_descent(R0, grad0, x0, max_iter=2000)
    
    if time.time() > deadline - 120.0:
        return x
    
    x, R = _polish(x, R, max_iter=50)
    
    if time.time() > deadline - 60.0:
        return x
    
    R_final = _evaluate(x)
    
    max_c = np.max(np.abs(x))
    if max_c < 1e-3:
        x = x / max_c * 1.0
    elif max_c > 1e5:
        x = x / max_c * 1.0
    
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return np.ones(21, dtype=np.float64)
    
    return x

# EVOLVE-BLOCK-END
