"""Rewritten program for logarithmic energy minimization of 150 points on the sphere."""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def energy(pts):
        """Compute log(1/d) = -log(d) energy using math.fsum for precision."""
        dot = pts @ pts.T
        dist2 = 2.0 - 2.0 * dot
        np.fill_diagonal(dist2, 1.0)
        dist2 = np.maximum(dist2, 1e-24)
        dist = np.sqrt(dist2)
        iu = np.triu_indices(N, k=1)
        terms = (-np.log(dist[iu[0], iu[1]])).tolist()
        return math.fsum(terms)

    def gradient(pts):
        """Compute gradient of energy on the sphere (tangent space)."""
        dot = pts @ pts.T
        dist2 = 2.0 - 2.0 * dot
        np.fill_diagonal(dist2, 1.0)
        dist2 = np.maximum(dist2, 1e-24)
        inv_dist2 = 1.0 / dist2
        np.fill_diagonal(inv_dist2, 0.0)
        grad = pts * inv_dist2.sum(axis=1, keepdims=True) - inv_dist2 @ pts
        dot_p = np.einsum('ij,ij->i', grad, pts)
        grad_t = grad - dot_p[:, np.newaxis] * pts
        return grad_t

    def fibonacci_sphere(offset=0.0):
        i = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_sphere(n, seed=None):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((n, 3))
        return normalize(pts)

    def lbfgs_optimize(pts, max_iter=3000, tol=1e-14):
        """Optimize using L-BFGS-B on the sphere via retraction."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_e = energy(pts)

        def obj_and_grad(x):
            p = normalize(x.reshape(N, 3))
            e = energy(p)
            g = gradient(p)
            return e, g.reshape(-1)

        x0 = pts.reshape(-1)
        e0, g0 = obj_and_grad(x0)

        try:
            res = minimize(
                lambda x: obj_and_grad(x)[0],
                x0,
                jac=lambda x: obj_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxfun': max_iter * 2}
            )
            p_opt = normalize(res.x.reshape(N, 3))
            e_opt = energy(p_opt)
            if e_opt < best_e:
                best_e = e_opt
                best_pts = p_opt
        except Exception:
            pass

        return best_pts, best_e

    def backtracking_polish(pts, max_iter=3000, lr_init=1e-3):
        """Riemannian gradient descent with backtracking line search for final polish."""
        pts = pts.copy()
        lr = lr_init
        best_pts = pts.copy()
        best_e = energy(pts)
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            g = gradient(pts)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-14:
                break

            # Backtracking line search
            alpha = lr
            e_curr = energy(pts)
            for _ in range(50):
                pts_new = normalize(pts - alpha * g)
                e_new = energy(pts_new)
                # Armijo condition
                if e_new < e_curr - 1e-4 * alpha * g_norm ** 2:
                    best_e = e_new
                    best_pts = pts_new.copy()
                    pts = pts_new
                    lr = min(alpha * 1.5, 1e-2)
                    break
                alpha *= 0.5
            else:
                break

            if it % 100 == 0 and time.time() > deadline - 5.0:
                break

        return best_pts, best_e

    def gradient_descent_polish(pts, max_iter=5000, lr_init=1e-5):
        """Final polish with small gradient steps and adaptive learning rate."""
        pts = pts.copy()
        lr = lr_init
        best_pts = pts.copy()
        best_e = energy(pts)
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            g = gradient(pts)
            pts_new = normalize(pts - lr * g)
            e_new = energy(pts_new)
            if e_new < best_e:
                best_e = e_new
                best_pts = pts_new.copy()
                pts = pts_new
                lr = min(lr * 1.02, 1e-3)
            else:
                lr *= 0.7
                if lr < 1e-12:
                    break
        return best_pts, best_e

    def conjugate_gradient_on_sphere(pts, max_iter=2000, tol=1e-12):
        """Conjugate gradient on the sphere for faster convergence."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_e = energy(pts)
        lr = 1e-4
        p = -gradient(pts)

        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            g = gradient(pts)
            g_norm = np.linalg.norm(g)
            if g_norm < tol:
                break

            pts_new = normalize(pts + lr * p)
            e_new = energy(pts_new)

            if e_new < best_e:
                best_e = e_new
                best_pts = pts_new.copy()
                pts = pts_new
                g_new = gradient(pts)
                beta = np.dot(g_new, g_new) / max(np.dot(g, g), 1e-16)
                p = -g_new + beta * p
                lr = min(lr * 1.01, 1e-2)
            else:
                lr *= 0.5
                if lr < 1e-14:
                    break

        return best_pts, best_e

    # Generate diverse initializations
    inits = []

    # Fibonacci sphere with various offsets
    for offset in [0.0, 0.1, 0.25, 0.4]:
        fib = normalize(fibonacci_sphere(offset))
        inits.append(fib)

    # Random initializations with different seeds
    for seed in [42, 123, 999, 7, 2024, 555, 314, 271, 888, 666]:
        inits.append(random_sphere(N, seed=seed))

    # Perturbed Fibonacci
    rng = np.random.default_rng(42)
    for noise_scale in [0.01, 0.05, 0.1, 0.2]:
        perturbed = normalize(inits[0] + noise_scale * rng.standard_normal((N, 3)))
        inits.append(perturbed)

    # Optimize each initialization
    best_pts = None
    best_e = float('inf')

    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
        e_init = energy(init_pts)
        if e_init < best_e:
            best_e = e_init
            best_pts = init_pts.copy()

        # L-BFGS optimization
        pts_opt, e_opt = lbfgs_optimize(init_pts, max_iter=2000, tol=1e-12)
        if e_opt < best_e:
            best_e = e_opt
            best_pts = pts_opt

    # Final polish on best result using conjugate gradient then backtracking polish
    if best_pts is not None and time.time() < deadline - 15.0:
        best_pts, best_e = conjugate_gradient_on_sphere(best_pts, max_iter=3000, tol=1e-13)

    if best_pts is not None and time.time() < deadline - 8.0:
        best_pts, best_e = backtracking_polish(best_pts, max_iter=2000, lr_init=1e-3)

    if best_pts is not None and time.time() < deadline - 3.0:
        best_pts, best_e = gradient_descent_polish(best_pts, max_iter=10000, lr_init=1e-6)

    # Ensure validity
    best_pts = normalize(best_pts)

    # Check no points too close
    dot = best_pts @ best_pts.T
    dist2 = 2.0 - 2.0 * dot
    np.fill_diagonal(dist2, 1.0)
    min_dist = np.sqrt(dist2.min())
    if min_dist < 1e-9:
        # Nudge apart
        rng = np.random.default_rng(0)
        for i in range(N):
            for j in range(i + 1, N):
                if dist2[i, j] < 1e-18:
                    best_pts[j] = normalize(best_pts[j] + 1e-6 * rng.standard_normal(3))
                    best_pts = normalize(best_pts)
                    # Recompute distances
                    dot = best_pts @ best_pts.T
                    dist2 = 2.0 - 2.0 * dot
                    np.fill_diagonal(dist2, 1.0)

    return best_pts
# EVOLVE-BLOCK-END
