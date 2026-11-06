"""Logarithmic energy minimisation for N points on the unit sphere."""
import time
import numpy as np
import math

N = 150
# Deadline guard: leave a safety margin (10 s) under the 120 s limit
_DEADLINE = time.time() + 110.0


def _energy(pts):
    """Sum_{i<j} log(1 / |x_i - x_j|) using math.fsum."""
    terms = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e18
            terms.append(-math.log(d))
    return math.fsum(terms)


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _grad(pts):
    """Gradient of E wrt each point (tangent-projected)."""
    m = len(pts)
    # distances
    diff = pts[:, None, :] - pts[None, :, :]        # (m,m,3)
    dist = np.linalg.norm(diff, axis=2)              # (m,m)
    np.fill_diagonal(dist, 1.0)                      # avoid div-by-zero
    inv_d2 = 1.0 / (dist * dist)                     # (m,m)
    # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)     # (m,3)
    # project to tangent: remove radial component
    grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    return grad


def _gd(pts, max_iters=400000, tol=1e-12):
    """Tangent-projected gradient descent with backtracking line search."""
    pts = pts.copy()
    E = _energy(pts)
    best_pts = pts.copy()
    best_E = E
    lr = 0.01
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        g = _grad(pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        # backtracking
        step = lr
        for _ in range(30):
            new_pts = pts - step * g
            # re-normalise
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_E = _energy(new_pts)
            if new_E < E - 1e-10 * step * g_norm * g_norm:
                pts = new_pts
                E = new_E
                if E < best_E:
                    best_E = E
                    best_pts = pts.copy()
                break
            step *= 0.5
        # occasionally increase lr if stuck
        if step < lr * 0.01:
            lr = min(lr * 1.5, 0.05)
        elif it % 1000 == 0:
            lr *= 0.95
    return best_pts, best_E


def _perturb(pts, seed, sigma=0.05):
    rng = np.random.default_rng(seed)
    p = pts + rng.normal(0, sigma, pts.shape)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    return p


def construct():
    t0 = time.time()
    # deterministic good start
    pts0 = _fibonacci_sphere(N)
    E0 = _energy(pts0)
    best_pts, best_E = _gd(pts0, max_iters=200000)
    
    # a few perturbations to escape local minima
    for seed in [1, 2, 3]:
        if time.time() > _DEADLINE - 15:
            break
        p = _perturb(pts0, seed, sigma=0.08)
        p2, E2 = _gd(p, max_iters=150000)
        if E2 < best_E:
            best_pts, best_E = p2, E2
    
    # final local refinement
    if time.time() < _DEADLINE - 5:
        best_pts, best_E = _gd(best_pts, max_iters=100000)
    
    # normalise
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
