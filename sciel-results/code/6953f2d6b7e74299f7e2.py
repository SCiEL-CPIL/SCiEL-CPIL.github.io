"""Logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
_START = time.time()
_DEADLINE = _START + 140.0


def _remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_fast(pts):
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    n = pts.shape[0]
    pts = _normalize(pts)
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))

    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)

    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)

    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


def _precise_energy(pts):
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _exp_map(pts, tangent):
    """Exponential map on the unit sphere."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    mask_small = norms < 1e-8
    theta = norms_safe
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    v_normalized = tangent / norms_safe
    new_pts = pts * cos_theta + v_normalized * sin_theta
    if np.any(mask_small):
        new_pts[mask_small] = pts[mask_small] + tangent[mask_small]
    return _normalize(new_pts)


def _log_map(pts, ref):
    """Logarithmic map: tangent vector from ref to pts on the sphere."""
    d = np.sum(pts * ref, axis=1, keepdims=True)
    d = np.clip(d, -1.0 + 1e-15, 1.0 - 1e-15)
    theta = np.arccos(d)
    theta_safe = np.where(np.abs(theta) < 1e-15, 1e-15, theta)
    sin_theta = np.sin(theta_safe)
    tangent = (pts - d * ref) / sin_theta
    tangent = tangent * theta_safe
    return tangent


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _langevin_exp(pts, deadline, rng, max_iter=2000, dt=0.003, gamma=0.5, T_init=0.4, T_min=1e-7):
    """Langevin dynamics with exponential map on the sphere."""
    current = _normalize(pts.copy())
    velocity = np.zeros_like(current)
    best_e = _energy_fast(current)
    best_pts = current.copy()
    n = N

    for it in range(max_iter):
        if time.time() > deadline:
            break

        frac = it / max_iter
        T = T_init * (T_min / T_init) ** (frac ** 0.35)

        e, g = _energy_grad(current)
        F = -g

        # Half-step velocity
        velocity += 0.5 * dt * F

        # Project velocity to tangent space
        dot_v = np.sum(velocity * current, axis=1, keepdims=True)
        velocity = velocity - dot_v * current

        # Full position step via exp map
        new_pts = _exp_map(current, dt * velocity)

        # New force
        new_e, new_g = _energy_grad(new_pts)
        new_F = -new_g

        # Half-step velocity with new force
        velocity += 0.5 * dt * new_F

        # Project to tangent space of new_pts
        dot_v = np.sum(velocity * new_pts, axis=1, keepdims=True)
        velocity = velocity - dot_v * new_pts

        # Damping
        velocity *= (1.0 - gamma * dt)

        # Noise (Langevin kick)
        if T > T_min:
            noise_scale = np.sqrt(2.0 * gamma * T * dt)
            noise = noise_scale * rng.standard_normal((n, 3))
            # Project noise to tangent space
            dot_noise = np.sum(noise * new_pts, axis=1, keepdims=True)
            noise = noise - dot_noise * new_pts
            velocity += noise

        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()

        current = new_pts

    return best_pts, best_e


def _riemannian_lbfgs(pts, deadline, max_iter=5000, ftol=1e-12, gtol=1e-8):
    """
    Riemannian L-BFGS on the sphere using exponential map retraction.
    This is the key structural difference: optimization happens in the
    tangent space with proper geodesic updates.
    """
    n = pts.shape[0]
    current = _normalize(pts.copy())
    best_pts = current.copy()
    best_e = _precise_energy(current)

    # L-BFGS memory
    m = 20
    S = []  # position differences (tangent vectors)
    Y = []  # gradient differences
    W = []  # curvature terms
    g_prev = None
    q = None
    r = None

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, g = _energy_grad(current)
        g_flat = g.reshape(-1)
        g_norm = np.linalg.norm(g_flat)

        if g_norm < gtol:
            break

        # L-BFGS two-loop recursion for direction
        if g_prev is None:
            d = -g_flat
        else:
            # Two-loop recursion
            q = g_flat.copy()
            alpha = np.zeros(m)
            for i in range(len(S) - 1, -1, -1):
                alpha[i] = np.dot(S[i], q) / np.dot(Y[i], S[i])
                q -= alpha[i] * Y[i]
            # Scale by inverse Hessian approximation
            if r is not None and np.dot(r, r) > 1e-30:
                gamma_ = np.dot(S[-1], Y[-1]) / np.dot(Y[-1], Y[-1])
            else:
                gamma_ = 1.0
            r = gamma_ * q
            for i in range(len(S)):
                beta = np.dot(Y[i], r) / np.dot(Y[i], S[i])
                r += (alpha[i] - beta) * S[i]
            d = -r

        # Project direction to tangent space
        d_3d = d.reshape(n, 3)
        dot_d = np.sum(d_3d * current, axis=1, keepdims=True)
        d_3d = d_3d - dot_d * current

        # Backtracking line search with Armijo condition
        alpha_step = 0.5
        c1 = 1e-4
        for ls in range(40):
            new_pts = _exp_map(current, alpha_step * d_3d)
            new_e = _energy_fast(new_pts)
            if new_e < e + c1 * alpha_step * np.dot(d_flat if False else g_flat, d_3d.reshape(-1)):
                break
            alpha_step *= 0.5
        else:
            # Line search failed, try smaller
            alpha_step = 0.5 * (0.5 ** 40)
            new_pts = _exp_map(current, alpha_step * d_3d)
            new_e = _energy_fast(new_pts)

        # Update L-BFGS memory
        if new_e < e - 1e-15:
            new_e2, new_g2 = _energy_grad(new_pts)
            new_g_flat = new_g2.reshape(-1)

            # Compute tangent-space displacement
            disp = _log_map(new_pts, current).reshape(-1)
            grad_diff = new_g_flat - g_flat

            S.append(disp)
            Y.append(grad_diff)

            if len(S) > m:
                S.pop(0)
                Y.pop(0)

            g_prev = g_flat
            current = new_pts

            if new_e < best_e:
                best_e = new_e
                best_pts = new_pts.copy()
        else:
            # No improvement, reduce step
            g_prev = g_flat
            # Don't update current, just break out or continue with smaller step
            if alpha_step < 1e-12:
                break

    return best_pts, best_e


def _newton_polish(pts, deadline, max_iter=30):
    """Newton refinement with diagonal Hessian approximation."""
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)
    eps = 1e-5

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, g = _energy_grad(best_pts)
        g_flat = g.reshape(-1)
        g_norm = np.linalg.norm(g_flat)

        if g_norm < 1e-14:
            break

        # Diagonal Hessian via finite differences (only first 5 iters)
        if it < 5:
            h_diag = np.zeros_like(g_flat)
            dim = g_flat.shape[0]
            for i in range(0, dim, 3):
                for k in range(3):
                    idx = i + k
                    p_plus = best_pts.copy()
                    p_plus.flat[idx] += eps
                    p_plus = _normalize(p_plus)
                    _, g_plus = _energy_grad(p_plus)

                    p_minus = best_pts.copy()
                    p_minus.flat[idx] -= eps
                    p_minus = _normalize(p_minus)
                    _, g_minus = _energy_grad(p_minus)

                    h_diag[idx] = (g_plus.reshape(-1)[idx] - g_minus.reshape(-1)[idx]) / (2 * eps)
            h_diag = np.abs(h_diag) + 1e-10
        else:
            h_diag = np.abs(g_flat) + 1e-10

        step = -g_flat / h_diag
        step_3d = step.reshape(n, 3)

        # Project to tangent space
        dot_s = np.sum(step_3d * best_pts, axis=1, keepdims=True)
        step_3d = step_3d - dot_s * best_pts

        step_norm = np.linalg.norm(step_3d)
        if step_norm < 1e-14:
            break

        max_step = 1e-3
        if step_norm > max_step:
            step_3d *= max_step / step_norm

        new_pts = _exp_map(best_pts, step_3d)
        new_e = _precise_energy(new_pts)

        if new_e < best_energy:
            best_pts = new_pts
            best_energy = new_e
        else:
            new_pts = _exp_map(best_pts, 0.5 * step_3d)
            new_e = _precise_energy(new_pts)
            if new_e < best_energy:
                best_pts = new_pts
                best_energy = new_e
            else:
                break

    return best_pts, best_energy


def _optimize_from_start(pts, deadline):
    """Multi-phase optimization: Riemannian L-BFGS then Newton polish."""
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)

    # Phase 1: Riemannian L-BFGS (coarse)
    if time.time() < deadline - 10:
        try:
            r1_pts, r1_e = _riemannian_lbfgs(best_pts, deadline - 5, max_iter=3000, ftol=1e-10, gtol=1e-6)
            if r1_e < best_energy:
                best_pts = r1_pts
                best_energy = r1_e
        except Exception:
            pass

    # Phase 2: Riemannian L-BFGS (fine)
    if time.time() < deadline - 5:
        try:
            r2_pts, r2_e = _riemannian_lbfgs(best_pts, deadline - 2, max_iter=5000, ftol=1e-13, gtol=1e-9)
            if r2_e < best_energy:
                best_pts = r2_pts
                best_energy = r2_e
        except Exception:
            pass

    # Phase 3: Newton polish
    if time.time() < deadline - 1:
        try:
            np_pts, np_e = _newton_polish(best_pts, deadline)
            if np_e < best_energy:
                best_pts = np_pts
                best_energy = np_e
        except Exception:
            pass

    return best_pts, best_energy


def _spectral_init(n, rng, l_max=8):
    """Structured smooth initial configuration."""
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_phi = 2.0 * rng.uniform(0, 1, n) - 1.0
    sin_phi = np.sqrt(1.0 - cos_phi ** 2)

    pts = np.column_stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])

    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            amp = rng.normal(0, 0.01 / (l + 1))
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arctan2(pts[:, 1], pts[:, 0])
            c = pts[:, 2]
            mod = amp * np.cos(m * t + phase) * (c ** abs(m))
            pts[:, 2] += mod * 0.1
            pts[:, 0] += amp * 0.05 * np.sin(m * t + phase) * sin_phi

    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    rng = np.random.default_rng(12345)

    initializations = []

    # 1. Fibonacci spiral variants
    for offset in [0.0, 0.15, 0.3, 0.5, 0.7]:
        initializations.append(_fibonacci_spiral(N, offset))

    # 2. Perturbed Fibonacci
    base = _fibonacci_spiral(N)
    for scale in [0.005, 0.01, 0.02, 0.05]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))

    # 3. Spectral/smooth initializations
    for seed in range(4):
        r = np.random.default_rng(seed * 100 + 7)
        initializations.append(_spectral_init(N, r, l_max=6))

    # 4. Random initializations
    for i in range(8):
        initializations.append(_random_sphere(N, rng))

    # Score and sort
    scored_inits = []
    for init in initializations:
        e_quick = _energy_fast(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    # Phase A: Langevin exploration on top initializations
    for e_quick, init_pts in scored_inits[:4]:
        if time.time() > deadline - 40:
            break
        try:
            lang_pts, lang_e = _langevin_exp(init_pts, deadline - 20, rng,
                                             max_iter=1500, dt=0.002,
                                             gamma=0.5, T_init=0.3, T_min=1e-7)
            polished_pts, polished_e = _optimize_from_start(lang_pts, deadline)
            if polished_e < best_energy:
                best_energy = polished_e
                best_pts = polished_pts
        except Exception:
            pass

    # Phase B: Direct Riemannian L-BFGS on all initializations
    for e_quick, init_pts in scored_inits:
        if time.time() > deadline - 15:
            break

        optimized_pts, e_precise = _optimize_from_start(init_pts, deadline)

        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    # Final polish
    if time.time() < deadline - 10:
        try:
            final_pts, final_e = _optimize_from_start(best_pts, deadline)
            if final_e < best_energy:
                best_pts = final_pts
                best_energy = final_e
        except Exception:
            pass

    # Final normalization and validation
    best_pts = _normalize(best_pts)
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _normalize(_fibonacci_spiral(N))

    return best_pts
