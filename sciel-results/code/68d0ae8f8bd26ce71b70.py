"""Spectral deformation + Riemannian polish for log-energy minimization on S^2."""
import numpy as np
import math
from scipy.optimize import minimize

N = 150


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    Energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    Gradient wrt xi: sum_{j!=i} (xi - xj) / |xi-xj|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)

    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))

    inv_dist2 = 1.0 / (dist ** 2)
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)

    return energy, grad


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _build_spectral_basis(fib_pts):
    """Build a spectral basis for low-frequency deformations.
    
    Returns array of shape (N, K, 3) where K is the number of basis functions.
    Basis includes:
      - 3 rotation generators (l=1)
      - 5 quadrupole deformations (l=2)
      - 7 octupole deformations (l=3)
    Total K = 15
    """
    # Rotation generators: R_x, R_y, R_z applied to each point
    # R_x u = (0, -u_z, u_y), R_y u = (u_z, 0, -u_x), R_z u = (-u_y, u_x, 0)
    basis = np.zeros((N, 15, 3))
    
    # l=1: rotations
    basis[:, 0, :] = np.column_stack([-fib_pts[:, 2], fib_pts[:, 1], np.zeros(N)])  # R_x
    basis[:, 1, :] = np.column_stack([fib_pts[:, 2], np.zeros(N), -fib_pts[:, 0]])   # R_y
    basis[:, 2, :] = np.column_stack([-fib_pts[:, 1], fib_pts[:, 0], np.zeros(N)])   # R_z
    
    # l=2: quadrupole deformations (traceless symmetric matrices)
    # Q1 = diag(2, -1, -1), Q2 = diag(0, 1, -1), Q3 = [[0,1,0],[1,0,0],[0,0,0]]
    # Q4 = [[0,0,1],[0,0,0],[1,0,0]], Q5 = [[0,0,0],[0,0,1],[0,1,0]]
    Q1 = np.array([[2, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    Q2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    Q3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    Q4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
    Q5 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    
    for k, Q in enumerate([Q1, Q2, Q3, Q4, Q5]):
        basis[:, 3 + k, :] = fib_pts @ Q.T
    
    # l=3: octupole deformations (7 independent traceless symmetric rank-3 tensors)
    # Use simple polynomial basis: x^3, x^2y, xy^2, y^3, x^2z, y^2z, xyz
    x, y, z = fib_pts[:, 0], fib_pts[:, 1], fib_pts[:, 2]
    basis[:, 8, :] = np.column_stack([x**3, x**2 * y, x * y**2])
    basis[:, 9, :] = np.column_stack([y**3, x**2 * z, y**2 * z])
    basis[:, 10, :] = np.column_stack([x * z**2, y * z**2, z**3])
    basis[:, 11, :] = np.column_stack([x * y * z, x**2 * z, y**2 * z])
    basis[:, 12, :] = np.column_stack([x**2 * z, x * y * z, y**2 * z])
    basis[:, 13, :] = np.column_stack([x * y * z, x**2 * y, x * y**2])
    basis[:, 14, :] = np.column_stack([x * y * z, x * y**2, y**2 * z])
    
    return basis


def _spectral_deform(fib_pts, basis, alpha):
    """Apply spectral deformation: pts = normalize(fib_pts + sum_k alpha_k * basis[:, k, :])"""
    disp = np.einsum('ijk,k->ij', basis, alpha)
    raw = fib_pts + disp
    return _project_to_sphere(raw)


def _spectral_energy_grad(alpha, fib_pts, basis):
    """Compute energy and gradient w.r.t. alpha for spectral deformation."""
    pts = _spectral_deform(fib_pts, basis, alpha)
    energy, grad_x = _energy_and_grad(pts)
    
    # d(pts_i)/d(alpha_k) = [basis[i,k] - pts_i * (pts_i . basis[i,k])] / |raw_i|^2
    # Actually: raw_i = fib_pts[i] + sum_k alpha_k * basis[i,k]
    # pts_i = raw_i / |raw_i|
    # d(pts_i)/d(alpha_k) = basis[i,k]/|raw_i| - raw_i * (raw_i . basis[i,k]) / |raw_i|^3
    #                      = [basis[i,k] - pts_i * (pts_i . basis[i,k])] / |raw_i|
    
    disp = np.einsum('ijk,k->ij', basis, alpha)
    raw = fib_pts + disp
    raw_norm = np.linalg.norm(raw, axis=1, keepdims=True)
    raw_norm = np.maximum(raw_norm, 1e-15)
    
    # d(pts_i)/d(alpha_k) for each i, k
    # = basis[i,k] / raw_norm_i - pts_i * (pts_i . basis[i,k]) / raw_norm_i
    # = (basis[i,k] - pts_i * (pts_i . basis[i,k])) / raw_norm_i
    
    # Compute pts_i . basis[i,k] for all i, k
    # shape: (N, K)
    dot_pb = np.einsum('ik,ikj->ij', pts, basis)  # (N, K)
    
    # (basis[i,k] - pts_i * dot_pb[i,k]) / raw_norm_i
    # shape: (N, K, 3)
    dpda = (basis - pts[:, np.newaxis, :] * dot_pb[:, :, np.newaxis]) / raw_norm[:, np.newaxis, :]
    
    # grad_alpha[k] = sum_i grad_x[i] . dpda[i, k, :]
    grad_alpha = np.einsum('ij,ikj->k', grad_x, dpda)
    
    return energy, grad_alpha


def _riemannian_polish(pts, max_iter=3000, lr_init=0.01, patience=400):
    """Riemannian gradient descent with backtracking line search."""
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    lr = lr_init
    no_improve = 0
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space
        grad_proj = grad - np.sum(pts * grad, axis=1, keepdims=True) * pts
        
        # Backtracking line search
        step_lr = lr
        accepted = False
        for _ in range(25):
            new_pts = pts + step_lr * grad_proj
            new_pts = _project_to_sphere(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-10:
                accepted = True
                break
            step_lr *= 0.5
        
        if not accepted:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                break
            continue
        
        pts = new_pts
        if new_energy < best_energy - 1e-12:
            best_energy = new_energy
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if it % 100 == 0 and it > 0:
            lr *= 0.92
        
        if no_improve > patience:
            break
    
    return best_pts, best_energy


def construct():
    # Step 1: Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = _project_to_sphere(fib_pts)
    
    # Step 2: Build spectral basis
    basis = _build_spectral_basis(fib_pts)
    K = basis.shape[1]
    
    # Step 3: Spectral optimization using L-BFGS-B
    # We optimize alpha in R^K to minimize energy
    def fun(alpha):
        e, _ = _spectral_energy_grad(alpha, fib_pts, basis)
        return e
    
    def jac(alpha):
        _, g = _spectral_energy_grad(alpha, fib_pts, basis)
        return g
    
    alpha0 = np.zeros(K)
    result = minimize(fun, alpha0, jac=jac, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    alpha_opt = result.x
    pts = _spectral_deform(fib_pts, basis, alpha_opt)
    
    # Step 4: Riemannian polish
    pts, energy1 = _riemannian_polish(pts, max_iter=3000, lr_init=0.01, patience=400)
    
    # Step 5: Fine polish with smaller learning rate
    pts, energy2 = _riemannian_polish(pts, max_iter=2000, lr_init=0.001, patience=300)
    
    # Step 6: Verify and fix any issues
    pts = _project_to_sphere(pts)
    
    # Check minimum distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        for _ in range(200):
            close_pairs = np.argwhere(dist < 1e-9)
            if len(close_pairs) == 0:
                break
            idx = close_pairs[0]
            noise = np.random.randn(3) * 1e-5
            pts[idx[0]] += noise
            pts[idx[0]] /= np.linalg.norm(pts[idx[0]])
            pts[idx[1]] += noise
            pts[idx[1]] /= np.linalg.norm(pts[idx[1]])
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1.0)
    
    return pts
