import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # 150s limit, keep 15s safety margin
    deadline = start_time + 130.0

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def pair_distances(pts):
        """Compute 150x150 distance matrix using d^2 = 2 - 2*(x_i . x_j)."""
        gram = pts @ pts.T
        d2 = 2.0 - 2.0 * gram
        d2 = np.maximum(d2, 0.0)
        np.fill_diagonal(d2, 1.0)
        return np.sqrt(d2)

    def energy(pts):
        """Compute log(1/d) = -log(d) energy for all pairs using math.fsum."""
        dist = pair_distances(pts)
        dist = np.maximum(dist, 1e-12)
        e = -np.log(dist)
        iu = np.triu_indices(N, k=1)
        terms = e[iu[0], iu[1]].tolist()
        return math.fsum(terms)

    def gradient(pts):
        """Gradient of energy w.r.t. points, projected to tangent space.
        grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
        """
        dist = pair_distances(pts)
        dist = np.maximum(dist, 1e-12)
        inv_d2 = 1.0 / (dist * dist)
        np.fill_diagonal(inv_d2, 0.0)
        row_sums = inv_d2.sum(axis=1)
        grad = -pts * row_sums[:, np.newaxis] + inv_d2 @ pts
        # Project to tangent space
        dot = np.einsum('ij,ij->i', grad, pts)
        grad = grad - dot[:, np.newaxis] * pts
        return grad

    def fibonacci_sphere():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_sphere(n, seed=None):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((n, 3))
        return normalize(pts)

    def obj_grad(x_flat):
        """Objective and gradient for L-BFGS-B on flattened 3N vector."""
        pts = x_flat.reshape(N, 3)
        pts = normalize(pts)
        e = energy(pts)
        g = gradient(pts)
        return e, g.flatten()

    def lbfgs_optimize(pts_init, time_limit, max_iter=1000, ftol=1e-12, gtol=1e-10):
        """L-BFGS-B optimization with time guard."""
        x0 = pts_init.flatten()
        
        def _callback(xk):
            if time.time() > time_limit:
                raise RuntimeError("time limit")
            return False
        
        try:
            res = minimize(
                fun=lambda x: obj_grad(x)[0],
                x0=x0,
                jac=lambda x: obj_grad(x)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': max_iter,
                    'ftol': ftol,
                    'gtol': gtol,
                    'maxfun': max_iter * 2,
                    'callback': _callback
                }
            )
            return normalize(res.x.reshape(N, 3)), energy(normalize(res.x.reshape(N, 3)))
        except RuntimeError:
            # Return best found so far (use current x0 if no progress)
            return normalize(pts_init), energy(pts_init)
        except Exception:
            return normalize(pts_init), energy(pts_init)

    def gd_polish(pts, max_iter=2000, lr_init=1e-5, tol=1e-12, time_limit=None):
        """Riemannian Gradient Descent with backtracking line search for final polish."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_e = energy(pts)
        lr = lr_init
        
        if time_limit is None:
            time_limit = deadline - 1.0

        for k in range(max_iter):
            if time.time() > time_limit:
                break
            
            g = gradient(pts)
            g_norm = np.linalg.norm(g)
            if g_norm < tol:
                break
            
            # Backtracking line search
            step = 1.0
            min_step = 1e-12
            c1 = 1e-4
            e_curr = energy(pts)
            slope = -np.sum(g * g)
            
            found_step = False
            while step >= min_step:
                new_pts = normalize(pts - lr * step * g)
                e_new = energy(new_pts)
                if not np.isfinite(e_new):
                    step *= 0.5
                    continue
                if e_new <= e_curr + c1 * (lr * step) * slope:
                    found_step = True
                    break
                step *= 0.5
            
            if not found_step:
                lr *= 0.5
                if lr < 1e-15:
                    break
                continue
            
            pts = new_pts
            e_new = energy(pts)
            if e_new < best_e:
                best_e = e_new
                best_pts = pts.copy()
            
            if step > 0.9:
                lr = min(lr * 1.1, 1e-3)
            else:
                lr = max(lr * 0.9, 1e-15)

        return best_pts, best_e

    def simulated_annealing(initial_pts, max_iters=2000, time_limit=None, seed=0):
        """Simulated Annealing on the sphere with periodic L-BFGS polish."""
        rng = np.random.default_rng(seed)
        pts = initial_pts.copy()
        best_pts = pts.copy()
        best_e = energy(pts)
        
        if time_limit is None:
            time_limit = deadline - 5.0

        T = 1.0
        alpha = 0.997
        step_size = 0.04 
        
        accept_count = 0
        total_count = 0
        target_accept = 0.5
        
        for k in range(max_iters):
            if time.time() > time_limit:
                break
            
            if total_count > 100:
                current_accept = accept_count / total_count
                if current_accept > target_accept + 0.1:
                    step_size *= 1.2
                elif current_accept < target_accept - 0.1:
                    step_size *= 0.8
                accept_count = 0
                total_count = 0

            noise = rng.standard_normal((N, 3)) * step_size
            new_pts = normalize(pts + noise)
            
            e_new = energy(new_pts)
            e_curr = energy(pts)
            delta_e = e_new - e_curr
            
            total_count += 1
            if delta_e < 0 or (T > 1e-10 and rng.random() < math.exp(-delta_e / T)):
                pts = new_pts
                accept_count += 1
                if e_new < best_e:
                    best_e = e_new
                    best_pts = pts.copy()
            
            T *= alpha
            
            # Periodic L-BFGS polish
            if k % 150 == 0 and k > 0 and time.time() < time_limit - 3:
                polished_pts, polished_e = lbfgs_optimize(pts, time_limit=time_limit - 2, max_iter=200, ftol=1e-10, gtol=1e-8)
                if polished_e < best_e:
                    best_e = polished_e
                    best_pts = polished_pts.copy()
                pts = polished_pts

        return best_pts, best_e

    # Main search loop
    best_pts = None
    best_e = float('inf')
    
    # 1. Start with Fibonacci sphere
    fib = normalize(fibonacci_sphere())
    e_fib = energy(fib)
    if e_fib < best_e:
        best_e = e_fib
        best_pts = fib.copy()
        
    # Polish the Fibonacci start with L-BFGS-B
    if time.time() < deadline - 20:
        pts_polished, e_polished = lbfgs_optimize(fib, time_limit=deadline - 15, max_iter=500, ftol=1e-12, gtol=1e-10)
        if e_polished < best_e:
            best_e = e_polished
            best_pts = pts_polished.copy()

    # 2. Run SA instances with L-BFGS polishing
    num_sa_runs = 6
    seeds = [42, 123, 999, 7, 2024, 555]
    
    for i in range(num_sa_runs):
        if time.time() > deadline - 15:
            break
            
        if i % 2 == 0:
            rng_init = np.random.default_rng(seeds[i])
            init_pts = normalize(best_pts + 0.03 * rng_init.standard_normal((N, 3)))
        else:
            init_pts = random_sphere(N, seed=seeds[i])
        
        remaining = deadline - time.time() - 15
        t_limit = min(deadline - 15, time.time() + remaining / (num_sa_runs - i))
        
        pts_sa, e_sa = simulated_annealing(init_pts, max_iters=1200, time_limit=t_limit, seed=seeds[i])
        
        if e_sa < best_e:
            best_e = e_sa
            best_pts = pts_sa.copy()
            
        if time.time() < deadline - 5:
            pts_final, e_final = lbfgs_optimize(pts_sa, time_limit=deadline - 3, max_iter=300, ftol=1e-12, gtol=1e-10)
            if e_final < best_e:
                best_e = e_final
                best_pts = pts_final.copy()

    # Final high-precision polish: L-BFGS then GD
    if time.time() < deadline - 2:
        best_pts, best_e = lbfgs_optimize(best_pts, time_limit=deadline - 1, max_iter=1000, ftol=1e-14, gtol=1e-12)
    
    if time.time() < deadline - 0.5:
        best_pts, best_e = gd_polish(best_pts, max_iter=5000, lr_init=1e-6, tol=1e-14, time_limit=deadline - 0.1)

    # Ensure validity
    best_pts = normalize(best_pts)
    
    # Check no points too close
    dist = pair_distances(best_pts)
    min_dist = dist.min()
    if min_dist < 1e-9:
        rng = np.random.default_rng(0)
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-9:
                    best_pts[j] = normalize(best_pts[j] + 1e-6 * rng.standard_normal(3))
                    best_pts = normalize(best_pts)
                    dist = pair_distances(best_pts)

    return best_pts
# EVOLVE-BLOCK-END
