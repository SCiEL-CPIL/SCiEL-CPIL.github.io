"""Morse cluster global minima: Multi-start with diverse structural prototypes
(FCC, HCP, Icosahedral) and L-BFGS-B refinement, with vectorized energy/gradient.
A hard wall-clock deadline guard returns the best finite result before the kill."""

import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 310.0  # Safety margin before 330s kill
_DEADLINE = _START + _BUDGET


def _remaining():
    return max(0.0, _DEADLINE - time.time())


def _morse_energy(rho, coords):
    """Exact Morse energy; returns large penalty if any pair < 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return 1e10
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)) / 2.0)


def _morse_energy_and_grad(rho, coords):
    """Morse energy and gradient with overlap penalty."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)

    min_d = np.min(d)
    if min_d < 0.5:
        return 1e10, np.zeros_like(coords)

    e = np.exp(rho * (1.0 - d))
    energy = float(np.sum(e * (e - 2.0)) / 2.0)

    # dV/dr = rho * e * (e - 1)
    dV_dr = rho * e * (e - 1.0)

    # diff[i,j] = r_i - r_j
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    mask = d > 0
    d_safe = np.where(mask, d, 1.0)

    grad = np.sum(dV_dr[:, :, np.newaxis] * diff / d_safe[:, :, np.newaxis], axis=1)

    return energy, grad


def _push_apart(coords, min_dist=0.51, iterations=100):
    """Push overlapping atoms apart."""
    coords = np.array(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return coords
    for _ in range(iterations):
        d = cdist(coords, coords)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d >= min_dist:
            break
        idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = idx
        if i == j:
            continue
        diff = coords[i] - coords[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            coords[i] += 0.01 * np.random.randn(3)
            continue
        diff /= dist
        coords[i] += 0.05 * diff
        coords[j] -= 0.05 * diff
    return coords


def _local_minimize(rho, x0, maxiter=150):
    """
    Perform local minimization using L-BFGS-B.
    Returns (energy, coords).
    """
    n = len(x0)
    x0 = _push_apart(x0)
    x0_flat = x0.flatten()

    def func_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        e, g = _morse_energy_and_grad(rho, x)
        return e, g.flatten()

    try:
        res = minimize(func_and_grad, x0_flat, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': 1e-6, 'ftol': 1e-12})
        x_opt = res.x.reshape(n, 3)
        e_opt = _morse_energy(rho, x_opt)
        if e_opt < 1e8:
            return e_opt, x_opt
    except Exception:
        pass

    # Fallback
    try:
        e_rel, x_rel = relax(x0, 'morse', rho)
        return e_rel, x_rel
    except Exception:
        return _morse_energy(rho, x0), x0


def _generate_fcc_seeds(n, rng, count=4):
    """Generate multiple FCC-like seeds with rotations and perturbations."""
    seeds = []
    pts = []
    limit = int(np.ceil((n ** (1 / 3)) * 1.5)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                pts.append([i, j, k])
                pts.append([(i + 0.5), (j + 0.5), k])
                pts.append([(i + 0.5), j, (k + 0.5)])
                pts.append([i, (j + 0.5), (k + 0.5)])

    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_fcc = pts[order[:n]].copy()
    # Scale so nearest neighbor ~ 1.0
    if n >= 2:
        d = cdist(base_fcc, base_fcc)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d > 0:
            base_fcc /= min_d

    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])
        rotated = base_fcc @ R.T
        noisy = rotated + rng.normal(0, 0.05, rotated.shape)
        seeds.append(noisy)

    return seeds


def _generate_hcp_seeds(n, rng, count=3):
    """Generate HCP-like seeds."""
    seeds = []
    a = 1.0
    c = a * np.sqrt(8 / 3)
    pts = []
    limit = int(np.ceil((n ** (1 / 3)) * 1.5)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                x = i * a + 0.5 * (j % 2) * a
                y = j * a * np.sqrt(3) / 2
                z = k * c
                pts.append([x, y, z])
                x2 = i * a + 0.5 * a + 0.5 * (j % 2) * a
                y2 = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                z2 = k * c + c / 3
                pts.append([x2, y2, z2])

    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_hcp = pts[order[:n]].copy()
    if n >= 2:
        d = cdist(base_hcp, base_hcp)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d > 0:
            base_hcp /= min_d

    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])
        rotated = base_hcp @ R.T
        noisy = rotated + rng.normal(0, 0.05, rotated.shape)
        seeds.append(noisy)

    return seeds


def _generate_icosahedral_seeds(n, rng, count=3):
    """Generate Icosahedral-like seeds (nested shells)."""
    seeds = []

    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])

    # Generate points on nested shells
    all_pts = []
    shells = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0]
    for r in shells:
        # Fibonacci sphere distribution
        cnt = int(12 * r ** 2)
        for i in range(cnt):
            y = 1.0 - (2.0 * i + 1.0) / cnt
            rr = np.sqrt(max(0.0, 1.0 - y * y))
            phi_angle = 2.0 * np.pi * i / ((1.0 + np.sqrt(5.0)) / 2.0)
            all_pts.append([np.cos(phi_angle) * rr * r, y * r, np.sin(phi_angle) * rr * r])

    all_pts = np.array(all_pts)
    all_pts -= all_pts.mean(axis=0)
    dists = np.linalg.norm(all_pts, axis=1)
    order = np.argsort(dists)
    base_icc = all_pts[order[:n]].copy()
    if n >= 2:
        d = cdist(base_icc, base_icc)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d > 0:
            base_icc /= min_d

    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])
        rotated = base_icc @ R.T
        noisy = rotated + rng.normal(0, 0.05, rotated.shape)
        seeds.append(noisy)

    return seeds


def _generate_random_compact_seeds(n, rng, count=3):
    """Generate random compact seeds."""
    seeds = []
    for _ in range(count):
        box_size = (n * 0.5) ** (1 / 3) * 2
        pts = rng.uniform(-box_size, box_size, size=(n, 3))
        seeds.append(pts)
    return seeds


def construct(instance):
    """Build a low-energy Morse cluster for the given instance."""
    n = instance["n"]
    rho = instance.get("rho", 6)

    rng = np.random.default_rng(42)

    # Generate diverse candidate seeds
    candidates = []
    candidates.extend(_generate_fcc_seeds(n, rng, count=4))
    candidates.extend(_generate_hcp_seeds(n, rng, count=3))
    candidates.extend(_generate_icosahedral_seeds(n, rng, count=3))
    candidates.extend(_generate_random_compact_seeds(n, rng, count=3))

    # Evaluate each candidate with polishing, respecting the time budget
    best_E = np.inf
    best_pos = None
    per_candidate = max(1.5, _remaining() / (len(candidates) + 1))

    for idx, cand in enumerate(candidates):
        if _remaining() < 3.0:
            break
        cand = np.asarray(cand, dtype=float)
        if len(cand) < n:
            extra = rng.uniform(-1, 1, (n - len(cand), 3))
            cand = np.vstack([cand, extra])
        elif len(cand) > n:
            cand = cand[:n]

        try:
            E, pos = _local_minimize(rho, cand, maxiter=100)
            if E < best_E:
                best_E = E
                best_pos = pos
        except Exception:
            try:
                E = _morse_energy(rho, cand)
                if E < best_E:
                    best_E = E
                    best_pos = cand.copy()
            except Exception:
                pass

    # Final safety
    if best_pos is None or not np.all(np.isfinite(best_pos)):
        best_pos = _generate_fcc_seeds(n, rng, count=1)[0]
        best_E = _morse_energy(rho, best_pos)

    return np.asarray(best_pos[:n], dtype=float)
