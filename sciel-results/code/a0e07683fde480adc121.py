"""Dual-representation optimization for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Vectorized log-energy and Riemannian gradient (projected to tangent space).
    
    Energy: E = sum_{i<j} log(1/|x_i - x_j|)
    dE/dx_i = sum_{j != i} -(x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    dot_products = pts @ pts.T
    dists_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)

    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts

    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts

    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(np.maximum(dists_sq[iu], 1e-30))
    energy = np.sum(np.log(1.0 / d))

    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    dot_products = pts @ pts.T
    dists_sq = 2.0 - 2.0 * dot_products
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(np.maximum(dists_sq[iu], 1e-30))
    return np.sum(np.log(1.0 / d))


def _energy_with_penalty(pts, lambda_min=0.0):
    """Log-energy plus a smooth penalty for close points (soft-min via log-sum-exp)."""
    n = len(pts)
    dot_products = pts @ pts.T
    dists_sq = 2.0 - 2.0 * dot_products
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(np.maximum(dists_sq[iu], 1e-15))
    energy = np.sum(np.log(1.0 / d))
    
    if lambda_min > 0:
        tau = 0.1
        neg_d_over_tau = -d / tau
        max_val = np.max(neg_d_over_tau)
        lse = max_val + np.log(np.sum(np.exp(neg_d_over_tau - max_val)))
        energy -= lambda_min * lse
    
    return energy


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        u3 = halton(i, base_start + 2)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _riemannian_cg(pts, time_limit=15.0, max_iter=500):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)

    d = -grad.copy()
    prev_grad = None

    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break

        # Backtracking line search with Armijo condition
        alpha = 0.5
        c1 = 1e-4
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = None

        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)

            grad_dot_d = np.sum(grad * d)
            if new_energy <= energy + c1 * alpha * grad_dot_d:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
                break

            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad

            alpha *= 0.5

        if best_alpha == 0.0:
            alpha = 1e-3
            new_pts = _exponential_map(current, alpha * (-grad))
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                current = new_pts
                energy = new_energy
                grad = new_grad
                d = -grad.copy()
                prev_grad = None
            else:
                break
            continue

        current = best_pts
        energy = best_energy
        grad = best_grad

        if np.linalg.norm(grad) < 1e-15:
            break

        d_new = -grad
        beta = 0.0
        if prev_grad is not None:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)

        d = d_new + beta * d
        prev_grad = grad.copy()

    return current, energy


def _lbfgs_optimize(pts, time_limit=30.0, m=20, maxiter=3000, gtol=1e-13, ftol=1e-16,
                    lambda_min=0.0):
    """Optimize using L-BFGS-B with projected gradient and optional min-distance penalty."""
    start = time.time()

    x0 = pts.flatten()

    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        
        if lambda_min > 0:
            n = len(p)
            dot_products = p @ p.T
            dists_sq = 2.0 - 2.0 * dot_products
            iu = np.triu_indices(n, k=1)
            d = np.sqrt(np.maximum(dists_sq[iu], 1e-15))
            tau = 0.1
            neg_d_over_tau = -d / tau
            max_val = np.max(neg_d_over_tau)
            weights = np.exp(neg_d_over_tau - max_val)
            Z = np.sum(weights)
            W = np.zeros((n, n))
            W[iu] = weights / Z
            W = W + W.T
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            d_full = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(d_full, 1.0)
            W_grad = W / (tau * d_full)
            np.fill_diagonal(W_grad, 0.0)
            grad_penalty = lambda_min * (W_grad @ diff)
            dp = np.sum(grad_penalty * p, axis=1)
            grad_penalty -= dp[:, np.newaxis] * p
            grad += grad_penalty
        
        return energy, grad.flatten()

    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                            'maxfun': maxiter * 5, 'maxls': 50, 'm': m})

    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)

    return x_opt, energy


def _dual_optimize(pts, time_limit=40.0):
    """Alternate between L-BFGS-B (with penalty) and Riemannian CG."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    # Phase 1: L-BFGS with soft-min penalty to avoid close points
    time_left = time_limit - (time.time() - start)
    if time_left > 12:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left * 0.5, 20.0),
                                           lambda_min=1.0)

    # Phase 2: Riemannian CG on pure log-energy
    time_left = time_limit - (time.time() - start)
    if time_left > 8:
        current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.6, 12.0))

    # Phase 3: L-BFGS without penalty for fine-tuning
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left, 15.0),
                                           m=25, maxiter=5000, gtol=1e-14, ftol=1e-17,
                                           lambda_min=0.0)

    return current, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish using L-BFGS-B with tight tolerances, then Newton steps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    try:
        res = minimize(fun_and_grad, current.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15,
                                'maxfun': 10000, 'maxls': 50, 'm': 30})
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        new_energy = _log_energy_fast(x_opt)
        if new_energy < energy:
            current = x_opt
            energy = new_energy
    except Exception:
        pass

    # Second very tight pass
    if time.time() - start < time_limit * 0.7:
        try:
            res2 = minimize(fun_and_grad, current.flatten(), method='L-BFGS-B', jac=True,
                            options={'maxiter': 1000, 'ftol': 1e-19, 'gtol': 1e-16,
                                     'maxfun': 5000, 'maxls': 50, 'm': 30})
            x_opt2 = res2.x.reshape(N, 3)
            x_opt2 = _normalize(x_opt2)
            new_energy2 = _log_energy_fast(x_opt2)
            if new_energy2 < energy:
                current = x_opt2
                energy = new_energy2
        except Exception:
            pass

    # Final Newton-style polish
    if time.time() - start < time_limit * 0.9:
        current = _newton_polish(current, time_remaining=time_limit - (time.time() - start))
        energy = _log_energy_fast(current)

    return current, energy


def _newton_polish(pts, time_remaining=2.0, max_steps=30):
    """A few Riemannian Newton steps using approximate Hessian (diagonal in tangent basis)."""
    start = time.time()
    current = pts.copy()
    n = len(current)

    for step in range(max_steps):
        if time.time() - start > time_remaining - 0.1:
            break

        energy, grad = _energy_and_grad(current)
        if np.linalg.norm(grad) < 1e-15:
            break

        grad_norm_sq = np.sum(grad * grad)
        scale = 1.0 / (1.0 + 0.1 * grad_norm_sq)
        step_vec = -scale * grad

        # Line search
        alpha = 1.0
        best_energy = energy
        best_pts = current.copy()
        for ls in range(20):
            new_pts = _exponential_map(current, alpha * step_vec)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
            if new_energy <= energy + 1e-4 * alpha * np.sum(grad * step_vec):
                break
            alpha *= 0.5

        if best_energy >= energy:
            break

        current = best_pts
        energy = best_energy

    return current


def _farthest_point_sampling(n, seed=42):
    """Generate points via farthest point sampling for better initial diversity."""
    rng = np.random.default_rng(seed)
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])

    for _ in range(1, n):
        candidates = rng.standard_normal((200, 3))
        candidates = _normalize(candidates)

        min_dists = np.full(len(candidates), np.inf)
        for p in pts:
            d = np.linalg.norm(candidates - p, axis=1)
            min_dists = np.minimum(min_dists, d)

        idx = np.argmax(min_dists)
        pts.append(candidates[idx])

    return np.array(pts)


def _perturb_and_optimize(pts, time_limit, rng):
    """Perturb points and optimize - helps escape local minima."""
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Multiple perturbation scales
    for scale in [0.01, 0.005, 0.002]:
        if time.time() > time_limit - 5:
            break
        perturbed = current + rng.standard_normal((N, 3)) * scale
        perturbed = _normalize(perturbed)
        
        time_left = min(time_limit - time.time() - 3, 15.0)
        if time_left < 3:
            break
        
        try:
            opt_pts, opt_energy = _dual_optimize(perturbed, time_limit=time_left)
            if opt_energy < energy:
                current = opt_pts
                energy = opt_energy
        except Exception:
            pass
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    initializations = []

    # 1. Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib.copy())

    # 2. Halton sequence
    pts_halton = _halton_sphere(N)
    initializations.append(pts_halton.copy())

    # 3. Random initializations
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)

    # 4. Perturbed Fibonacci
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)

    # 5. Farthest point sampling
    try:
        pts_fps = _farthest_point_sampling(N, seed=99)
        initializations.append(pts_fps)
    except Exception:
        pass

    # 6. Different Fibonacci variants
    for offset in [0.3, 0.7]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))

    # 7. Additional structured initializations
    rng_local = np.random.default_rng(777)
    pts_ico = rng_local.standard_normal((N, 3))
    pts_ico = _normalize(pts_ico)
    initializations.append(pts_ico)
    
    # Low-discrepancy: Sobol-like via golden ratio in 3D
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i_arr = np.arange(N)
    t1 = (i_arr * golden) % 1.0
    t2 = (i_arr * golden * golden) % 1.0
    t3 = (i_arr * golden * golden * golden) % 1.0
    theta = 2.0 * np.pi * t1
    phi = np.arccos(2.0 * t2 - 1.0)
    pts_gold = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    initializations.append(_normalize(pts_gold))

    # 8. Additional random seeds
    for seed in [111, 222, 333]:
        rng_local = np.random.default_rng(seed)
        pts_rand = rng_local.standard_normal((N, 3))
        initializations.append(_normalize(pts_rand))

    best_pts = initializations[0].copy()
    best_energy = _log_energy_fast(best_pts)

    # Track which initializations have been tried
    tried = set()

    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 25:
            break

        time_left = deadline - time.time() - 5
        if time_left < 10:
            break

        try:
            opt_pts, opt_energy = _dual_optimize(init_pts, time_limit=min(time_left, 35.0))

            if time.time() < deadline - 3:
                polish_time_total = min(deadline - time.time() - 1, 6.0)
                if polish_time_total > 0.5:
                    polish_time1 = polish_time_total * 0.6
                    polished_pts, polished_energy = _polish(opt_pts, time_limit=polish_time1)
                    if polished_energy < opt_energy:
                        opt_pts = polished_pts
                        opt_energy = polished_energy

                    if time.time() < deadline - 2:
                        polish_time2 = min(deadline - time.time() - 1, polish_time_total * 0.4)
                        if polish_time2 > 0.3:
                            polished_pts2, polished_energy2 = _polish(opt_pts, time_limit=polish_time2)
                            if polished_energy2 < opt_energy:
                                opt_pts = polished_pts2
                                opt_energy = polished_energy2

            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                tried.add(idx)

        except Exception:
            try:
                time_left = deadline - time.time() - 5
                if time_left > 5:
                    opt_pts, opt_energy = _lbfgs_optimize(init_pts, time_limit=min(time_left, 20.0))
                    if time.time() < deadline - 2:
                        polish_time = min(deadline - time.time() - 1, 4.0)
                        if polish_time > 0.5:
                            polished_pts, polished_energy = _polish(opt_pts, time_limit=polish_time)
                            if polished_energy < opt_energy:
                                opt_pts = polished_pts
                                opt_energy = polished_energy
                    if opt_energy < best_energy:
                        best_energy = opt_energy
                        best_pts = opt_pts.copy()
                        tried.add(idx)
            except Exception:
                pass

    # Perturbation-based local search on the best solution
    if time.time() < deadline - 15:
        rng_perturb = np.random.default_rng(42)
        pert_time = min(deadline - time.time() - 5, 25.0)
        if pert_time > 5:
            try:
                pert_pts, pert_energy = _perturb_and_optimize(best_pts, deadline - 5.0, rng_perturb)
                if pert_energy < best_energy:
                    best_energy = pert_energy
                    best_pts = pert_pts.copy()
            except Exception:
                pass

    # Final high-precision polish on best solution
    if time.time() < deadline - 5:
        final_time = min(deadline - time.time() - 2, 12.0)
        if final_time > 1.0:
            try:
                polish_time1 = final_time * 0.5
                final_pts, final_energy = _polish(best_pts, time_limit=polish_time1)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pts = final_pts

                if time.time() < deadline - 3:
                    polish_time2 = min(deadline - time.time() - 1, final_time * 0.5)
                    if polish_time2 > 0.5:
                        final_pts2, final_energy2 = _polish(best_pts, time_limit=polish_time2)
                        if final_energy2 < best_energy:
                            best_energy = final_energy2
                            best_pts = final_pts2
            except Exception:
                pass

    # Very final: one more Newton polish with remaining time
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 0.5:
            try:
                best_pts = _newton_polish(best_pts, time_remaining=remaining)
                best_energy = _log_energy_fast(best_pts)
            except Exception:
                pass

    best_pts = _normalize(best_pts)

    return best_pts
