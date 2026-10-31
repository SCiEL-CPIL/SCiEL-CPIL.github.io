import time
import numpy as np
from scipy.optimize import minimize

N = 38
KNOWN_GLOBAL_MIN = -173.928427


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)

    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad


def _make_fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron geometry for LJ38.
    Generate FCC lattice points, take the 38 closest to origin, center,
    and scale so the minimum nearest-neighbor distance is 2^(1/6).
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    # Center before sorting by distance
    pts -= np.mean(pts, axis=0)

    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    pts = pts[:N].copy()
    # Re-center after selection
    pts -= np.mean(pts, axis=0)

    # Scale so minimum nearest-neighbor distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))

    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-12:
        scale = target_r / min_r
        pts *= scale

    return pts


def _relax(pos, max_iter=3000, gtol=1e-12, ftol=1e-16):
    """L-BFGS-B local relaxation with analytic gradient."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out


def _optimize_scale(pos, t0=None, deadline=None):
    """
    Optimize the global scale of the cluster by minimizing energy as a function
    of a single scale factor s applied to centered positions.
    Uses a 1D golden-section / scipy minimize_scalar approach.
    """
    if t0 is None:
        t0 = time.time()
    if deadline is None:
        deadline = t0 + 10.0

    # Center the cluster
    centroid = np.mean(pos, axis=0)
    centered = pos - centroid

    def scale_energy(log_s):
        s = np.exp(log_s)
        p = centered * s + centroid
        e, _ = _lj_energy_and_grad(p)
        return e

    # Initial scale estimate: target min distance ~ 2^(1/6)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    s0 = target_r / min_r if min_r > 1e-12 else 1.0
    log_s0 = np.log(s0)

    # Use scipy minimize_scalar on the log-scale
    from scipy.optimize import minimize_scalar

    try:
        res = minimize_scalar(
            scale_energy,
            bounds=(log_s0 - 0.15, log_s0 + 0.15),
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 50},
        )
        if res.success and res.fun < scale_energy(log_s0) - 1e-12:
            s_opt = np.exp(res.x)
            new_pos = centered * s_opt + centroid
            return new_pos, res.fun
    except Exception:
        pass

    return pos, scale_energy(log_s0)


def _polish(pos, max_iter=8000):
    """Extremely tight polish pass."""
    return _relax(pos, max_iter=max_iter, gtol=1e-14, ftol=1e-18)


def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin within 150 s hard limit

    # ---- 1. Build the FCC truncated-octahedron geometry ----
    fcc = _make_fcc_truncated_octahedron()

    # ---- 1b. Optimize the global scale of the FCC fragment ----
    if time.time() < deadline - 120.0:
        fcc, _ = _optimize_scale(fcc, t0=t0, deadline=deadline - 120.0)
        fcc = _ensure_min_sep(fcc)

    # ---- 2. Initial high-precision relaxation ----
    best_pos, best_energy = _relax(fcc, max_iter=3000, gtol=1e-12, ftol=1e-16)

    # ---- 2b. Post-relaxation scale optimization ----
    if time.time() < deadline - 110.0:
        best_pos, e_scaled = _optimize_scale(best_pos, t0=t0, deadline=deadline - 110.0)
        if e_scaled < best_energy:
            best_energy = e_scaled
            best_pos = _ensure_min_sep(best_pos)
            # Re-relax after scale change
            best_pos, best_energy = _relax(best_pos, max_iter=2000, gtol=1e-12, ftol=1e-16)

    # Fast path: if we're at the known global minimum, polish and return
    if best_energy < KNOWN_GLOBAL_MIN + 1e-6:
        best_pos = _ensure_min_sep(best_pos)
        if time.time() < deadline - 5.0:
            p, e = _polish(best_pos)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        best_pos = _ensure_min_sep(best_pos)
        return best_pos

    # ---- 3. Basin hopping from current best ----
    rng = np.random.default_rng(42)
    current_best = best_pos.copy()
    bh_accepted = 0

    for i in range(40):
        if time.time() > deadline - 15.0:
            break

        # Adaptive perturbation scale: start moderate, decrease over time
        base_scale = 0.15
        scale = base_scale * (1.0 - 0.5 * i / 40.0)
        scale = max(scale, 0.05)

        # 60% from current best, 40% from original FCC
        if i % 5 == 0:
            base = fcc
        else:
            base = current_best

        perturbed = base + scale * rng.standard_normal(base.shape)
        perturbed = _ensure_min_sep(perturbed)

        try:
            p, e = _relax(perturbed, max_iter=2000, gtol=1e-10, ftol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
                current_best = p.copy()
                bh_accepted += 1
        except Exception:
            continue

    # ---- 4. Final high-precision polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 8.0:
        try:
            p, e = _polish(best_pos)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 5. Post-polish scale optimization + re-relax ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 4.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos, t0=t0, deadline=deadline - 4.0)
            if e_scaled < best_energy - 1e-10:
                p_scaled = _ensure_min_sep(p_scaled)
                p_final, e_final = _relax(p_scaled, max_iter=3000, gtol=1e-14, ftol=1e-18)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pos = p_final.copy()
        except Exception:
            pass

    # ---- 6. Micro-polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 1.0:
        try:
            p, e = _relax(best_pos, max_iter=3000, gtol=1e-14, ftol=1e-18)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
