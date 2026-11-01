import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE_MARGIN = 5.0  # seconds to leave before hard timeout


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


def _tangent_proj(v, pts):
    """Project each row of v (N,3) into the tangent space at the corresponding pts."""
    d = np.sum(v * pts, axis=1, keepdims=True)
    return v - d * pts


def _hvp(pts, v):
    """Hessian-vector product of the log-energy restricted to the sphere.
    v is (N,3). Returns the tangent-space Hessian applied to v (N,3)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)

    # Per-point coefficient c_i = sum_j (1/d2_ij)
    c = np.sum(inv_d2, axis=1)

    # Ambient Hessian applied to v:
    # (H v)_i = -c_i v_i + 2 * sum_j (1/d2_ij) ( (x_i - x_j) . v_j ) (x_i - x_j)
    dot = diff @ v                                   # (N,N)
    weighted = (2.0 * inv_d2) * dot                  # (N,N)
    Hv = -c[:, np.newaxis] * v + (weighted @ diff)   # (N,3)

    # Project to tangent space
    return _tangent_proj(Hv, pts)


def _newton_polish(pts, time_limit=8.0, tol=1e-12, max_cg=60, max_steps=12, m=15):
    """Vectorized Newton polish using a CG-solve for the Hessian inverse.
    Works on the sphere via tangent-space Hessian and exponential map."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    best_pts = current.copy()
    best_energy = energy

    # Precompute a basis (u, w) per point for the 2D tangent space
    u = np.zeros((N, 3))
    w = np.zeros((N, 3))
    for i in range(N):
        p = current[i]
        a = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(a, p)) > 0.9:
            a = np.array([1.0, 0.0, 0.0])
        u[i] = a - np.dot(a, p) * p
        u[i] /= (np.linalg.norm(u[i]) + 1e-30)
        w[i] = np.cross(p, u[i])
        w[i] /= (np.linalg.norm(w[i]) + 1e-30)

    # Flatten a tangent vector (N,3) into 2N coords
    def flat(v):
        return np.concatenate([np.sum(v * u, axis=1), np.sum(v * w, axis=1)])

    def unflat(fv):
        return fv[:N, np.newaxis] * u + fv[N:2 * N, np.newaxis] * w

    # Precompute per-point 2x2 Gram of Hessian (constant during polish since H
    # depends only on current, not the step).
    def compute_gram():
        G = np.zeros((2 * N, 2 * N))
        for i in range(N):
            e_u = np.zeros((N, 3)); e_u[i] = u[i]
            e_w = np.zeros((N, 3)); e_w[i] = w[i]
            Hu = _hvp(current, e_u)
            Hw = _hvp(current, e_w)
            G[i, i] = np.dot(Hu, e_u[i])
            G[i, N + i] = np.dot(Hu, e_w[i])
            G[N + i, i] = np.dot(Hw, e_u[i])
            G[N + i, N + i] = np.dot(Hw, e_w[i])
        return G

    # CG solve G x = b  (G is SPD)
    def cg_solve(G, b):
        x = np.zeros_like(b)
        r = b - G @ x
        p = r.copy()
        rs = np.dot(r, r)
        for _ in range(max_cg):
            Ap = G @ p
            pAp = np.dot(p, Ap)
            if abs(pAp) < 1e-30:
                break
            alpha = rs / pAp
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.dot(r, r)
            if math.sqrt(rs_new) < 1e-15:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x

    for step in range(max_steps):
        if time.time() - start > time_limit - 0.5:
            break
        gflat = flat(grad)
        if np.linalg.norm(gflat) < tol:
            break
        G = compute_gram()
        try:
            dflat = -cg_solve(G, gflat)
        except Exception:
            dflat = -gflat
        d = unflat(dflat)
        d = _tangent_proj(d, current)
        if np.linalg.norm(d) < 1e-16:
            break

        # Line search with Armijo
        alpha = 1.0
        best_a = 0.0
        best_e = energy
        best_g = grad
        best_p = current.copy()
        dir_deriv = np.sum(grad * d)
        if dir_deriv > 0:
            d = -d
            dir_deriv = -dir_deriv
        c1 = 1e-4
        for ls in range(30):
            np_ = _exponential_map(current, alpha * d)
            ne, ng = _energy_and_grad(np_)
            if ne < best_e:
                best_e, best_g, best_p, best_a = ne, ng, np_, alpha
            if ne > energy + c1 * alpha * dir_deriv:
                break
            alpha *= 0.5
            if alpha < 1e-15:
                break
        if best_a == 0.0:
            break
        current, energy, grad = best_p, best_e, best_g
        if energy < best_energy:
            best_energy = energy
            best_pts = current.copy()

    return best_pts, best_energy


def _lbfgs_sphere(pts, time_limit=10.0, max_iter=200, m=10):
    """L-BFGS on the sphere using exponential maps and parallel transport."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)

    s_history = []
    y_history = []

    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break

        q = grad.copy()
        rho = []
        for k in range(len(s_history) - 1, -1, -1):
            s_k = s_history[k]
            y_k = y_history[k]
            ys = np.sum(y_k * s_k)
            if ys < 1e-30:
                rho.append(0.0)
                continue
            rho_k = 1.0 / ys
            rho.append(rho_k)
            alpha_k = rho_k * np.sum(s_k * q)
            q -= alpha_k * y_k

        r = q
        d = r.copy()
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            if rho[k] == 0.0:
                continue
            ys = np.sum(y_k * s_k)
            if ys < 1e-30:
                continue
            beta_k = (1.0 / ys) * np.sum(y_k * r)
            d += (rho[k] - beta_k) * s_k

        dot_d = np.sum(d * current, axis=1)
        d -= dot_d[:, np.newaxis] * current

        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break

        alpha = 1.0
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv

        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            if new_energy > energy + c1 * alpha * directional_deriv:
                break
            alpha *= 0.5
            if alpha < 1e-15:
                break

        if best_alpha == 0.0:
            for small_alpha in [1e-3, 1e-4, 1e-5, 1e-6]:
                new_pts = _exponential_map(current, small_alpha * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = small_alpha
                    break
            if best_alpha == 0.0:
                break

        s_vec = best_pts - current
        y_vec = best_grad - grad
        dot_s = np.sum(s_vec * current, axis=1)
        s_vec -= dot_s[:, np.newaxis] * current
        dot_y = np.sum(y_vec * best_pts, axis=1)
        y_vec -= dot_y[:, np.newaxis] * best_pts

        s_history.append(s_vec)
        y_history.append(y_vec)
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)

        current = best_pts
        energy = best_energy
        grad = best_grad

        if np.linalg.norm(grad) < 1e-13:
            break

    return current, energy


def _fps_fast(n, seed=42, batch_size=1000):
    """Fast farthest point sampling using vectorized operations."""
    rng = np.random.default_rng(seed)
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / (np.linalg.norm(pts[0]) + 1e-30)
    pts_arr = np.array(pts)

    for _ in range(n - 1):
        candidates = rng.standard_normal((batch_size, 3))
        candidates = _normalize(candidates)
        min_dists = np.full(batch_size, np.inf)
        chunk = 50
        for i in range(0, len(pts_arr), chunk):
            chunk_pts = pts_arr[i:i + chunk]
            d = np.linalg.norm(candidates[:, np.newaxis, :] - chunk_pts[np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, np.min(d, axis=1))
        best_idx = np.argmax(min_dists)
        pts_arr = np.vstack([pts_arr, candidates[best_idx:best_idx + 1]])

    return _normalize(pts_arr)


def _basin_hopping(pts, time_limit=30.0, max_restarts=10, reheat_every=4, reheat_temp=0.35):
    """Local optimization followed by basin hopping with adaptive perturbation + reheat."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _lbfgs_sphere(current, time_limit=min(15, time_left - 2))

    best_pts = current.copy()
    best_energy = energy

    temp = 0.6
    decay = 0.82

    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 5:
            break

        if rh > 0 and rh % reheat_every == 0:
            temp = reheat_temp

        # Adaptive perturbation scale tied to temperature and current energy
        base_scale = max(0.04, min(0.30, 0.05 * (1.0 + temp) + 0.02))
        rng = np.random.default_rng(rh * 7919 + 1000)
        perturbations = rng.standard_normal((N, 3)) * base_scale
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current

        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)

        opt_time = min(12, time_left - 3)
        if opt_time < 3:
            break
        new_opt_pts, new_energy = _lbfgs_sphere(new_pts, time_limit=opt_time)

        delta_e = new_energy - energy
        accept = delta_e < 0
        if not accept and temp > 1e-6:
            try:
                accept = np.random.rand() < math.exp(-delta_e / temp)
            except OverflowError:
                accept = False

        if accept:
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()

        temp *= decay

    # Final polish within basin hopping
    time_left = time_limit - (time.time() - start)
    if time_left > 3:
        try:
            polished, pol_energy = _lbfgs_sphere(best_pts, time_limit=max(2.0, time_left - 1.0), max_iter=200, m=15)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_pts = polished
        except Exception:
            pass

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


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN

    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))

    for s in range(3):
        rng = np.random.default_rng(s * 7 + 1)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)

    for s in range(2):
        rng = np.random.default_rng(s * 13 + 200)
        scale = 0.1 + 0.08 * s
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)

    # Fast farthest-point sampling initializations (diverse, well-spread)
    for s in range(2):
        try:
            inits.append(_fps_fast(N, seed=500 + s * 37))
        except Exception:
            pass

    best_pts = None
    best_energy = float('inf')

    n_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 18:
            break

        time_remaining_for_inits = remaining - 12  # reserve for final Newton polish
        inits_remaining = n_inits - idx
        time_per_restart = time_remaining_for_inits / inits_remaining
        time_per_restart = min(24.0, max(10.0, time_per_restart))

        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)

    # Final polish: high-precision Newton refinement using remaining time
    time_left = deadline - time.time()
    if time_left > 2:
        try:
            final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=max(2.0, time_left * 0.4), max_iter=300, m=15)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass

    time_left = deadline - time.time()
    if time_left > 2:
        try:
            new_pts, new_energy = _newton_polish(best_pts, time_limit=max(1.5, time_left - 1.0))
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
        except Exception:
            pass

    # One more Newton pass if time remains for extra precision
    time_left = deadline - time.time()
    if time_left > 1.5:
        try:
            new_pts, new_energy = _newton_polish(best_pts, time_limit=max(1.0, time_left - 0.5))
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
        except Exception:
            pass

    return _normalize(best_pts)
