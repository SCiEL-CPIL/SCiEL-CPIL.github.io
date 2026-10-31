"""LJ38 global minimum search: FCC truncated-octahedron geometry with robust
median-NN scaling, basin hopping for robustness, multi-scale perturbation,
and high-precision L-BFGS polish within a 120 s wall-clock budget."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE_S = 150.0
MARGIN = 15.0


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _relax(pos, maxiter=20000, deadline=None, gtol=1e-10, ftol=1e-16):
    """L-BFGS-B relaxation with optional deadline and precision control."""
    pos = np.array(pos, dtype=float).copy()

    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol,
                                'maxls': 50})
        return res.x.reshape(-1, 3), res.fun
    except TimeoutError:
        return pos, _lj_energy_and_grad(pos)[0]
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def _make_fcc_truncated_octahedron():
    """
    Construct the FCC truncated-octahedron geometry for 38 atoms.
    Uses the median nearest-neighbour distance for robust scaling to the
    LJ minimum r0 = 2^(1/6).
    """
    # Generate FCC lattice points in a cubic box
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts.append(np.array([i, j, k], dtype=float))
                pts.append(np.array([i + 0.5, j + 0.5, k], dtype=float))
                pts.append(np.array([i + 0.5, j, k + 0.5], dtype=float))
                pts.append(np.array([i, j + 0.5, k + 0.5], dtype=float))
    pts = np.array(pts)
    pts = np.unique(np.round(pts, 12), axis=0)

    # Center
    center = pts.mean(axis=0)
    pts = pts - center

    # Select 38 closest to centroid
    dists = np.sum(pts ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    pos = pts[idx].copy()

    # Compute nearest-neighbour distances to robustly scale
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    # For each atom, its nearest neighbour distance
    nn_dists = np.min(r, axis=1)
    # Use median for robustness
    median_nn = np.median(nn_dists)

    # Scale so that median NN distance = r0
    r0 = 2.0 ** (1.0 / 6.0)
    if median_nn > 1e-12:
        scale = r0 / median_nn
        pos = pos * scale

    # Re-center
    pos -= pos.mean(axis=0)
    return pos


def _perturb(pos, sigma, rng):
    """Add Gaussian noise to positions."""
    return pos + rng.randn(*pos.shape) * sigma


def _basin_hopping(pos, deadline, rng, n_iter=50, max_sigma=0.3):
    """
    Simple basin hopping: perturb, relax, accept if lower energy.
    Returns best position and energy found.
    """
    best_pos = pos.copy()
    best_e = _lj_energy_and_grad(pos)[0]
    
    for i in range(n_iter):
        if time.time() > deadline:
            break
        
        # Anneal: start with small perturbations, increase over time
        progress = i / max(n_iter - 1, 1)
        sigma = max_sigma * (0.1 + 0.9 * progress)
        
        pos_p = _perturb(best_pos, sigma, rng)
        pos_r, e = _relax(pos_p, maxiter=5000, deadline=deadline,
                          gtol=1e-8, ftol=1e-14)
        
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
    
    return best_pos, best_e


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_S - MARGIN

    best_pos = None
    best_e = np.inf

    # 1) FCC truncated octahedron: initial relax + high-precision polish
    try:
        pos0 = _make_fcc_truncated_octahedron()
        pos_r, e = _relax(pos0, maxiter=20000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
        # High-precision polish pass
        if time.time() < deadline - 5.0:
            pos_r2, e2 = _relax(pos_r, maxiter=10000, deadline=deadline,
                                gtol=1e-12, ftol=1e-16)
            if e2 < best_e:
                best_e = e2
                best_pos = pos_r2.copy()
        # Ultra-high precision final polish
        if time.time() < deadline - 2.0 and best_pos is not None:
            pos_r3, e3 = _relax(best_pos, maxiter=15000, deadline=deadline,
                                gtol=1e-14, ftol=1e-18)
            if e3 < best_e:
                best_e = e3
                best_pos = pos_r3.copy()
    except Exception:
        pass

    if best_pos is not None and time.time() > deadline:
        return best_pos

    # 2) Basin hopping from the FCC result for robustness
    if time.time() < deadline - 10.0:
        try:
            rng_bh = np.random.RandomState(42)
            pos_bh, e_bh = _basin_hopping(best_pos, deadline, rng_bh,
                                          n_iter=30, max_sigma=0.25)
            if e_bh < best_e:
                best_e = e_bh
                best_pos = pos_bh.copy()
                # Polish the basin hopping result
                if time.time() < deadline - 3.0:
                    pos_r, e = _relax(best_pos, maxiter=15000, deadline=deadline,
                                      gtol=1e-12, ftol=1e-16)
                    if e < best_e:
                        best_e = e
                        best_pos = pos_r.copy()
        except Exception:
            pass

    if time.time() > deadline:
        if best_pos is not None:
            return best_pos

    # 3) Icosahedral start (alternative basin)
    try:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        r0 = 2.0 ** (1.0 / 6.0)
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts[0]) * r0
        pos_list = [np.zeros(3)]
        pos_list.extend(list(verts))
        n_outer = N - len(pos_list)
        golden = (1.0 + np.sqrt(5.0)) / 2.0
        for i in range(n_outer):
            z = 1.0 - 2.0 * (i + 0.5) / n_outer
            radius = np.sqrt(1.0 - z * z)
            theta = 2.0 * np.pi * i / golden
            v = 2.0 * r0 * np.array([radius * np.cos(theta),
                                      radius * np.sin(theta),
                                      z])
            pos_list.append(v)
        pos_ico = np.array(pos_list[:N])
        pos_ico -= pos_ico.mean(axis=0)
        
        pos_r, e = _relax(pos_ico, maxiter=15000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
    except Exception:
        pass

    if time.time() > deadline:
        if best_pos is not None:
            return best_pos

    # 4) Multi-scale perturbations around the best result
    rng = np.random.RandomState(12345)
    base = best_pos if best_pos is not None else _make_fcc_truncated_octahedron()
    for sigma in [0.01, 0.02, 0.04, 0.06, 0.10, 0.15]:
        if time.time() > deadline:
            break
        for _ in range(2):
            if time.time() > deadline:
                break
            pos_p = _perturb(base, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=8000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    # 5) Random starts (catch-all)
    for _ in range(5):
        if time.time() > deadline:
            break
        pos_rand = rng.randn(N, 3) * 2.0
        pos_r, e = _relax(pos_rand, maxiter=4000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    if best_pos is None:
        # Fallback: cubic lattice
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: sum((c - (side - 1) / 2.0) ** 2 for c in p))
        return np.array(pts[:N], dtype=float) * 1.12

    # Final ultra-high-precision polish if time remains
    if time.time() < deadline - 2.0:
        pos_r, e = _relax(best_pos, maxiter=20000, deadline=deadline,
                          gtol=1e-14, ftol=1e-18)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
    # One more polish with even tighter tolerance if still time
    if time.time() < deadline - 1.0:
        pos_r, e = _relax(best_pos, maxiter=10000, deadline=deadline,
                          gtol=1e-15, ftol=1e-20)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    # Ensure minimum pairwise distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        scale = 0.5 / min_r * 1.001
        best_pos = best_pos * scale

    return best_pos
