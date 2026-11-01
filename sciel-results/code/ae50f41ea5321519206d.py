import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_S = 143.0  # safety margin before 150s limit


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


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)

    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts

    # Project gradient to tangent space: g - (g . p) p
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts

    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))

    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _riemannian_cg(pts, max_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """Riemannian conjugate gradient on the sphere for log-energy minimization."""
    if start_time is None:
        start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    def _time_exceeded():
        if time_limit is None:
            return False
        return time.time() - start_time > time_limit
    
    energy, grad = _energy_and_grad(pts)
    p = -grad
    
    for it in range(max_iter):
        if _time_exceeded():
            break
        
        best_alpha = 0.01
        best_e = energy
        for alpha in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            new_pts = _exponential_map(pts, alpha * p)
            new_pts = _normalize(new_pts)
            new_e, _ = _energy_and_grad(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_alpha = alpha
            if _time_exceeded():
                break
        
        if best_alpha == 0.0:
            break
        
        new_pts = _exponential_map(pts, best_alpha * p)
        new_pts = _normalize(new_pts)
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if new_energy >= energy:
            found = False
            for alpha in [best_alpha * 0.5, best_alpha * 0.25, best_alpha * 0.1]:
                trial_pts = _exponential_map(pts, alpha * p)
                trial_pts = _normalize(trial_pts)
                trial_e, _ = _energy_and_grad(trial_pts)
                if trial_e < energy:
                    new_pts = trial_pts
                    new_energy, new_grad = _energy_and_grad(trial_pts)
                    found = True
                    break
            if not found:
                break
        
        beta = np.sum(new_grad * (new_grad - grad)) / (np.sum(grad * grad) + 1e-15)
        beta = max(0.0, min(beta, 1.0))
        
        grad_norm = np.linalg.norm(new_grad)
        if grad_norm < tol:
            pts = new_pts
            break
        
        p = -new_grad + beta * p
        dot_p = np.sum(p * new_pts, axis=1, keepdims=True)
        p -= dot_p * new_pts
        
        pts = new_pts
        energy = new_energy
        grad = new_grad
    
    return pts, _log_energy_fast(pts)


def _lbfgs_sphere(pts, time_limit=15.0):
    """L-BFGS-B optimization on the sphere."""
    start = time.time()
    x0 = pts.flatten().copy()

    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return float('inf'), np.zeros_like(x)
        pts_local = x.reshape(N, 3)
        pts_local = _normalize(pts_local)
        energy, grad = _energy_and_grad(pts_local)
        return energy, grad.flatten()

    result = minimize(
        obj_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 3000,
            'ftol': 1e-15,
            'gtol': 1e-11,
            'maxfun': 50000,
        }
    )

    if result.success or result.fun < float('inf'):
        best_pts = _normalize(result.x.reshape(N, 3))
        best_energy = _log_energy_fast(best_pts)
    else:
        best_pts = pts.copy()
        best_energy = _log_energy_fast(pts)

    return best_pts, best_energy


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
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])

    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _spherical_design_init(n, seed=42):
    """Generate points using structured approach with random rotations."""
    rng = np.random.default_rng(seed)
    pts = _fibonacci_sphere(n)
    for _ in range(3):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        pts = pts @ R.T
    return _normalize(pts)


def _fps_fast(n, seed=42, batch_size=1000):
    """Fast farthest point sampling using vectorized operations."""
    rng = np.random.default_rng(seed)
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    pts_arr = np.array(pts)

    for _ in range(n - 1):
        candidates = rng.standard_normal((batch_size, 3))
        candidates = _normalize(candidates)
        min_dists = np.full(batch_size, np.inf)
        chunk = 50
        for i in range(0, len(pts_arr), chunk):
            chunk_pts = pts_arr[i:i+chunk]
            d = np.linalg.norm(candidates[:, np.newaxis, :] - chunk_pts[np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, np.min(d, axis=1))
        best_idx = np.argmax(min_dists)
        pts_arr = np.vstack([pts_arr, candidates[best_idx:best_idx+1]])

    return _normalize(pts_arr)


def _basin_hopping(pts, time_limit=20.0, max_restarts=15):
    """Basin hopping with Riemannian CG + L-BFGS local optimization."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    time_left_init = time_limit - (time.time() - start) - 2.0
    if time_left_init > 5.0:
        cg_time = min(5.0, time_left_init * 0.35)
        current, _ = _riemannian_cg(current, max_iter=300, tol=1e-9,
                                     time_limit=cg_time, start_time=time.time())
        lbfgs_time = min(10.0, time_left_init - cg_time - 1.0)
        if lbfgs_time > 2.0:
            current, energy = _lbfgs_sphere(current, time_limit=lbfgs_time)

    best_pts = current.copy()
    best_energy = energy

    temp = 0.4
    decay = 0.88
    reheat_every = 4
    reheat_temp = 0.25

    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2.0:
            break

        if rh > 0 and rh % reheat_every == 0:
            temp = reheat_temp

        perturb_scale = max(0.02, min(0.20, 0.06 * temp + 0.02))

        rng = np.random.default_rng(rh * 7919 + 1)
        perturbations = rng.standard_normal((N, 3))
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        pert_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
        pert_norms = np.maximum(pert_norms, 1e-15)
        perturbations = perturbations / pert_norms * perturb_scale

        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)

        time_left = time_limit - (time.time() - start) - 1.0
        if time_left < 4.0:
            break

        cg_t = min(3.0, time_left * 0.3)
        new_opt_pts, _ = _riemannian_cg(new_pts, max_iter=150, tol=1e-8,
                                          time_limit=cg_t, start_time=time.time())
        lbfgs_t = min(7.0, time_left - cg_t - 0.5)
        if lbfgs_t > 2.0:
            new_opt_pts, new_energy = _lbfgs_sphere(new_opt_pts, time_limit=lbfgs_t)
        else:
            new_energy = _log_energy_fast(new_opt_pts)
        new_energy = _log_energy_fast(new_opt_pts)

        delta_e = new_energy - energy
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(temp, 1e-10)):
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()

        temp *= decay

    return best_pts, best_energy


def _dual_optimize(pts, time_limit=30.0):
    """Dual-representation optimization: alternate between ambient L-BFGS and
    Riemannian CG with perturbation to escape local minima."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    for phase in range(3):
        remaining = time_limit - (time.time() - start) - 1.0
        if remaining < 3.0:
            break
        
        lbfgs_t = min(remaining * 0.4, 12.0)
        if lbfgs_t > 2.0:
            current, energy = _lbfgs_sphere(current, time_limit=lbfgs_t)
        
        cg_t = min(remaining * 0.3, 8.0)
        if cg_t > 1.5:
            current, _ = _riemannian_cg(current, max_iter=500, tol=1e-12,
                                         time_limit=cg_t, start_time=time.time())
            e = _log_energy_fast(current)
            if e < energy:
                energy = e
        
        perturb_t = min(remaining * 0.2, 5.0)
        if perturb_t > 2.0:
            rng = np.random.default_rng(phase * 104729 + 7)
            perturbations = rng.standard_normal((N, 3))
            dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
            perturbations -= dot_prods * current
            pert_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
            pert_norms = np.maximum(pert_norms, 1e-15)
            perturbations = perturbations / pert_norms * 0.04
            
            new_pts = _exponential_map(current, perturbations)
            new_pts = _normalize(new_pts)
            
            new_pts, new_energy = _lbfgs_sphere(new_pts, time_limit=perturb_t * 0.7)
            if new_energy < energy:
                current = new_pts
                energy = new_energy
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_S

    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_spherical_design_init(N, seed=42))
    inits.append(_spherical_design_init(N, seed=123))
    inits.append(_spherical_design_init(N, seed=777))

    inits.append(_fps_fast(N, seed=7))
    inits.append(_fps_fast(N, seed=13))
    inits.append(_fps_fast(N, seed=99))

    for s in range(3):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)

    for s in range(3):
        rng = np.random.default_rng(s + 100)
        base = _fibonacci_sphere(N) if s % 2 == 0 else _halton_sphere(N)
        pts = base + rng.standard_normal((N, 3)) * 0.10
        pts = _normalize(pts)
        inits.append(pts)

    best_pts = None
    best_energy = float('inf')

    total_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        remaining = deadline - time.time() - 15.0
        if remaining < 8.0:
            break

        if idx < 4:
            time_per_restart = min(20.0, remaining / (total_inits - idx) * 1.6)
        elif idx < 8:
            time_per_restart = min(14.0, remaining / (total_inits - idx) * 1.3)
        else:
            time_per_restart = min(10.0, remaining / (total_inits - idx) * 1.1)

        time_per_restart = max(5.0, time_per_restart)

        try:
            if idx < 4:
                opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            else:
                opt_pts, opt_energy = _dual_optimize(init_pts, time_limit=time_per_restart)
            opt_energy = _log_energy_fast(opt_pts)

            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()

        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)

    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        try:
            cg_polish_time = min(6.0, remaining * 0.3)
            if cg_polish_time > 1.5:
                polished_pts, _ = _riemannian_cg(best_pts, max_iter=500, tol=1e-13,
                                                  time_limit=cg_polish_time,
                                                  start_time=time.time())
                polished_e = _log_energy_fast(polished_pts)
                if polished_e < best_energy:
                    best_energy = polished_e
                    best_pts = polished_pts

            lbfgs_polish_time = min(8.0, deadline - time.time() - 4.0)
            if lbfgs_polish_time > 2.0:
                final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=lbfgs_polish_time)
                final_energy = _log_energy_fast(final_pts)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pts = final_pts

            remaining2 = deadline - time.time() - 3.0
            if remaining2 > 2.5:
                cg2_time = min(4.0, remaining2)
                if cg2_time > 1.0:
                    final2_pts, _ = _riemannian_cg(best_pts, max_iter=400, tol=1e-14,
                                                    time_limit=cg2_time,
                                                    start_time=time.time())
                    final2_e = _log_energy_fast(final2_pts)
                    if final2_e < best_energy:
                        best_energy = final2_e
                        best_pts = final2_pts
        except Exception:
            pass

    return _normalize(best_pts)
