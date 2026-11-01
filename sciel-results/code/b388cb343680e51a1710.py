"""Lennard-Jones 55-atom cluster global minimum search.

Keeps the Mackay icosahedron multi-start design and adds:
- explicit wall-clock deadline guard,
- optimized shell radii via L-BFGS,
- diverse perturbations with time-aware multi-start,
- final high-precision polish pass.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 55

# Deadline guard: evaluator kills after 150s; use 145s for safety.
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _get_icosahedron_vectors():
    """
    Returns 12 unit vectors along 5-fold axes (icosahedron vertices)
    and 30 unit vectors along 2-fold axes (edge midpoints).
    """
    phi = (1 + np.sqrt(5)) / 2

    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=float)

    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    dists = np.linalg.norm(
        verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2
    )
    min_dist = np.min(dists[dists > 1e-8])

    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(dists[i, j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)

    edge_mids = np.array(edge_mids)
    return verts, edge_mids


def _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283):
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    """
    v5, v2 = _get_icosahedron_vectors()

    points = [np.zeros(3)]

    for v in v5:
        points.append(v * r1)

    for v in v2:
        points.append(v * r2)

    for v in v5:
        points.append(v * r3)

    return np.array(points, dtype=float)


def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    return 4.0 * np.sum(np.where(mask, r12 - r6, 0.0))


def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(np.where(mask, r12 - r6, 0.0))

    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4.0 * (-12.0 * r2_inv7 + 6.0 * r2_inv4)
    coeff = np.where(mask, coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5


def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytical gradient."""
    if _remaining() < 2.0:
        return pos

    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()

    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)


def _optimize_radii(base_pos, maxiter=2000):
    """
    Optimize the three shell radii of the Mackay icosahedron
    to minimize LJ energy, then relax the full geometry.
    """
    if _remaining() < 10.0:
        return base_pos, _lj_energy(base_pos)

    v5, v2 = _get_icosahedron_vectors()

    def objective(radii):
        r1, r2, r3 = radii
        if r1 < 0.5 or r2 < 0.5 or r3 < 0.5:
            return 1e10
        points = [np.zeros(3)]
        for v in v5:
            points.append(v * r1)
        for v in v2:
            points.append(v * r2)
        for v in v5:
            points.append(v * r3)
        pos = np.array(points, dtype=float)
        return _lj_energy(pos)

    r0 = np.array([1.135, 1.162, 1.283])
    result = minimize(
        objective, r0, method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    best_radii = np.clip(result.x, 0.5, 3.0)
    best_pos = _mackay_icosahedron_55(*best_radii)
    best_energy = _lj_energy(best_pos)
    return best_pos, best_energy


def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    with Gaussian noise at multiple scales and relaxing with L-BFGS.
    """
    global _start_time
    _start_time = time.time()

    rng = np.random.default_rng(42)

    # Start from well-established Mackay icosahedron radii
    base = _mackay_icosahedron_55(1.135, 1.162, 1.283)

    best_energy = float('inf')
    best_pos = base.copy()

    # Step 1: Optimize radii, then relax
    if _remaining() > 20.0:
        try:
            opt_pos, opt_e = _optimize_radii(base, maxiter=2000)
            if _is_valid(opt_pos) and opt_e < best_energy:
                best_energy = opt_e
                best_pos = opt_pos.copy()
        except Exception:
            pass

    # Step 2: Relax the base geometry
    if _remaining() > 10.0:
        relaxed = _relax(base)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()

    # Step 3: Multi-start with Gaussian noise at various scales
    noise_scales = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.17, 0.22]

    for scale in noise_scales:
        if _remaining() < 15.0:
            break
        noise = rng.normal(0, scale, size=base.shape)
        pos = base + noise

        relaxed = _relax(pos)

        if not _is_valid(relaxed):
            continue

        energy = _lj_energy(relaxed)

        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()

    # Step 4: Additional trials with moderate noise for diversity
    for trial in range(15):
        if _remaining() < 15.0:
            break
        scale = rng.uniform(0.01, 0.15)
        noise = rng.normal(0, scale, size=base.shape)
        pos = base + noise

        relaxed = _relax(pos)

        if not _is_valid(relaxed):
            continue

        energy = _lj_energy(relaxed)

        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()

    # Step 5: Final high-precision polish pass
    if best_energy < float('inf') and _remaining() > 10.0:
        refined = _relax(best_pos, maxiter=30000, ftol=1e-14, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined

    return best_pos
