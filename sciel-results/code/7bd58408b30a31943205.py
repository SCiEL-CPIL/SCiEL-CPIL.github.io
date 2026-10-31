"""
LJ38 cluster global minimum search.

Strategy:
1. Construct the known FCC truncated-octahedron geometry analytically with
   precise nearest-neighbor scaling to 2^(1/6).
2. High-precision L-BFGS relaxation from this structure.
   If the energy is already at the known global minimum (~ -173.928427),
   return immediately (fast path).
3. Otherwise, basin hopping from the current best structure:
   - Perturb with Gaussian noise, relax, accept if lower energy.
   - This provides robustness against any numerical issues or if the
     initial FCC construction doesn't converge to the global minimum.
4. Final micro-polish with extremely tight tolerances.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
KNOWN_GLOBAL_MIN = -173.928427


def _lj_energy(pos):
    """Total LJ energy for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    return 0.5 * np.sum(pair_energy)


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


def _make_fcc_cluster():
    """
    Build the 38-atom FCC truncated-octahedron geometry.
    Returns 38 points centered at origin, scaled so NN distance = 2^(1/6).
    """
    # FCC lattice points: all (i,j,k) with i+j+k even, within a cube
    # The 38-atom truncated octahedron is the set of FCC points closest to origin
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:N]].copy()
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


def _polish(pos, max_iter=5000):
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


def _basin_hopping(pos0, max_iter=30, rng=None, deadline=None):
    """
    Basin hopping: perturb, relax, accept if lower energy.
    Returns the best position and energy found.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    best_pos = pos0.copy()
    best_energy, _ = _lj_energy_and_grad(pos0)
    current_pos = pos0.copy()
    current_energy = best_energy

    # Adaptive perturbation scale
    scale = 0.5

    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Perturb
        noise = scale * rng.standard_normal((N, 3))
        perturbed = current_pos + noise

        # Relax
        try:
            relaxed, e = _relax(perturbed, max_iter=2000, gtol=1e-10, ftol=1e-14)
        except Exception:
            continue

        # Accept if lower
        if e < current_energy:
            current_pos = relaxed
            current_energy = e
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
                # Increase scale slightly on improvement
                scale = min(scale * 1.1, 1.5)
        else:
            # Decrease scale on rejection
            scale = max(scale * 0.9, 0.1)

    return best_pos, best_energy


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin within 150 s hard limit

    # ---- 1. Build the FCC truncated-octahedron geometry ----
    fcc = _make_fcc_cluster()

    # ---- 2. High-precision relaxation from FCC ----
    pos, energy = _relax(fcc, max_iter=3000, gtol=1e-12, ftol=1e-16)

    # Fast path: if we're at the known global minimum, polish and return
    if energy < KNOWN_GLOBAL_MIN + 1e-6:
        pos = _ensure_min_sep(pos)
        if time.time() < deadline - 5.0:
            p, e = _polish(pos, max_iter=5000)
            if e < energy:
                energy = e
                pos = p.copy()
        pos = _ensure_min_sep(pos)
        return pos

    # ---- 3. Basin hopping from the current best structure ----
    rng = np.random.default_rng(42)
    best_pos, best_energy = _basin_hopping(
        pos, max_iter=30, rng=rng, deadline=deadline - 10.0
    )

    # ---- 4. Final high-precision polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 5.0:
        try:
            p, e = _polish(best_pos, max_iter=5000)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 5. Micro-polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 2.0:
        try:
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-14, ftol=1e-18)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
