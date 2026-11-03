"""Combined logarithmic energy minimization: multi-start (Fibonacci + geodesic dome + random),
tangent-space gradient with Adam, then L-BFGS-B polish. Explicit 120 s deadline guard."""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _energy_and_grad(pts):
    """Log energy and tangent-space gradient for points on the unit sphere.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    grad_i = sum_{j != i} (x_j - x_i) / d_ij^2, projected to tangent space."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())

    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)

    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts
    grad -= pts * row_sums[:, np.newaxis]

    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


def _energy_only(pts):
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _fibonacci_spiral(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    rng = np.random.RandomState(seed)
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    edges = [(0, 11), (0, 5), (0, 1), (0, 7), (0, 10), (1, 5), (5, 11), (11, 10), (10, 7),
             (7, 1), (1, 6), (6, 10), (2, 11), (2, 5), (2, 9), (2, 4), (2, 8), (3, 5),
             (3, 11), (3, 4), (3, 9), (3, 6), (4, 9), (5, 4), (11, 4), (10, 7), (6, 1),
             (8, 4), (9, 8), (8, 5), (6, 9), (7, 2), (8, 3)]

    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])

    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]

    return verts


def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _adam_optimize(pts0, max_iter=2000, lr_start=0.02, lr_end=0.0005,
                   beta1=0.9, beta2=0.999, eps=1e-8, deadline=None):
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)

    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    for t in range(1, max_iter + 1):
        if deadline is not None and time.time() > deadline:
            break

        energy, grad = _energy_and_grad(pts)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)

        lr = lr_schedule(t)
        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()

        pts = pts_new

    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=300):
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 3,
                                'gtol': 1e-12, 'ftol': 1e-12})
        if res.x is not None:
            X_res = _project_to_sphere(res.x.reshape(N, 3))
            E_res = _energy_only(X_res)
            if np.all(np.isfinite(X_res)) and E_res < 1e17:
                return X_res, E_res
    except Exception:
        pass
    return pts, _energy_only(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 s safety margin

    best_X = None
    best_E = float('inf')

    # Initializations
    inits = []
    inits.append(_fibonacci_spiral(N, offset=0.0))
    inits.append(_fibonacci_spiral(N, offset=0.25))
    inits.append(_geodesic_dome(N, seed=0))
    inits.append(_random_points(N, seed=42))

    for init in inits:
        if time.time() > deadline:
            break

        X_adam, E_adam = _adam_optimize(init, max_iter=1500, deadline=deadline)
        if E_adam < best_E:
            best_E = E_adam
            best_X = X_adam

        # L-BFGS polish on the best result so far
        if time.time() < deadline - 5.0:
            X_pol, E_pol = _lbfgs_polish(best_X, deadline=deadline, maxiter=200)
            if E_pol < best_E:
                best_E = E_pol
                best_X = X_pol

    # Final high-precision polish pass
    if best_X is not None and time.time() < deadline - 2.0:
        X_final, E_final = _lbfgs_polish(best_X, deadline=deadline, maxiter=500)
        if E_final < best_E:
            best_E = E_final
            best_X = X_final

    if best_X is None:
        best_X = _fibonacci_spiral(N)

    best_X = _project_to_sphere(best_X)

    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                best_X[j] += 1e-8 * np.random.randn(3)
                best_X[j] = _project_to_sphere(best_X[j])

    return best_X

# EVOLVE-BLOCK-END
