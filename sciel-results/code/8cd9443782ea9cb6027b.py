"""Improved program for logarithmic energy minimization on the sphere."""
import time
import math
import numpy as np

N = 100
DEADLINE_FRACTION = 0.85  # safety margin

def construct():
    t_start = time.time()
    total_budget = 150.0 * DEADLINE_FRACTION  # ~127.5s

    def elapsed():
        return time.time() - t_start

    def remaining():
        return total_budget - elapsed()

    def energy(pts):
        """Compute logarithmic energy using math.fsum for precision."""
        n = len(pts)
        terms = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-15:
                    return float('inf')
                terms.append(math.log(1.0 / d))
        return math.fsum(terms)

    def energy_grad(pts):
        """Compute energy and gradient on the sphere (using ambient gradient projected)."""
        n = len(pts)
        e = 0.0
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-15:
                    return float('inf'), grad
                # log(1/d) = -log(d), d = |xi - xj|
                # dE/dxi = -(1/d) * (xi - xj)/d = -(xi - xj)/d^2
                # dE/dxj = +(xi - xj)/d^2
                inv_d2 = 1.0 / (d * d)
                e -= math.log(d)
                grad[i] -= diff * inv_d2
                grad[j] += diff * inv_d2
        # Project gradient to tangent space
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        return e, grad

    def project(pts):
        """Project points to unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return pts / norms

    def fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def gradient_descent(pts, max_iters=5000, tol=1e-14):
        """Steepest descent with backtracking line search on the sphere."""
        pts = project(pts.copy())
        e, grad = energy_grad(pts)
        if not np.isfinite(e):
            return pts, e
        lr = 0.1
        for it in range(max_iters):
            if remaining() < 5.0:
                break
            # Backtracking line search
            alpha = lr
            improved = False
            for _ in range(30):
                new_pts = project(pts - alpha * grad)
                new_e, _ = energy_grad(new_pts)
                if np.isfinite(new_e) and new_e < e:
                    pts = new_pts
                    e = new_e
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                # Try smaller step
                alpha = 1e-8
                new_pts = project(pts - alpha * grad)
                new_e, _ = energy_grad(new_pts)
                if np.isfinite(new_e) and new_e < e:
                    pts = new_pts
                    e = new_e
                else:
                    break
            grad = energy_grad(pts)[1]
            if np.linalg.norm(grad) < tol:
                break
        return pts, e

    def simulated_annealing(pts, max_iters=3000, t_start=0.01, t_end=1e-8):
        """Simulated annealing with random perturbations."""
        pts = project(pts.copy())
        e, _ = energy_grad(pts)
        if not np.isfinite(e):
            return pts, e
        best_pts = pts.copy()
        best_e = e
        for it in range(max_iters):
            if remaining() < 5.0:
                break
            temp = t_start * (t_end / t_start) ** (it / max_iters)
            # Random perturbation
            idx = np.random.randint(N)
            noise = np.random.randn(3) * 0.05
            new_pts = pts.copy()
            new_pts[idx] = project(new_pts[idx] + noise)
            new_e, _ = energy_grad(new_pts)
            if not np.isfinite(new_e):
                continue
            if new_e < e or np.random.random() < math.exp(-(new_e - e) / temp):
                pts = new_pts
                e = new_e
                if e < best_e:
                    best_e = e
                    best_pts = pts.copy()
        return best_pts, best_e

    def random_restart(n_restarts=5):
        """Generate diverse random initial configurations."""
        configs = [fibonacci_spiral()]
        for _ in range(n_restarts):
            # Random points on sphere
            pts = np.random.randn(N, 3)
            pts = project(pts)
            configs.append(pts)
        return configs

    # Main optimization loop
    best_pts = None
    best_e = float('inf')

    # Start with Fibonacci spiral
    pts0 = fibonacci_spiral()
    pts_opt, e_opt = gradient_descent(pts0, max_iters=3000)
    best_pts = pts_opt.copy()
    best_e = e_opt

    # Try multiple random restarts
    n_restarts = 8
    for r in range(n_restarts):
        if remaining() < 15.0:
            break
        pts_rand = project(np.random.randn(N, 3))
        # First do some gradient descent
        pts_gd, e_gd = gradient_descent(pts_rand, max_iters=2000)
        if e_gd < best_e:
            best_e = e_gd
            best_pts = pts_gd.copy()
        # Then try simulated annealing from the best so far
        pts_sa, e_sa = simulated_annealing(pts_gd, max_iters=2000, t_start=0.005, t_end=1e-9)
        if e_sa < best_e:
            best_e = e_sa
            best_pts = pts_sa.copy()

    # Final polish with tighter tolerance
    if remaining() > 10.0:
        pts_final, e_final = gradient_descent(best_pts, max_iters=10000, tol=1e-16)
        if e_final < best_e:
            best_e = e_final
            best_pts = pts_final.copy()

    # Ensure output is valid
    best_pts = project(best_pts)
    if not np.all(np.isfinite(best_pts)):
        return fibonacci_spiral()

    return best_pts
