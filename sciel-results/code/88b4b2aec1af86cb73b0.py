"""Morse cluster global minima: diverse lattice seeds + perturb-refine cycles."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 290.0
    return _T0


def _time_left():
    global _HARD_DEADLINE
    return max(0.0, _HARD_DEADLINE - time.time())


def _enforce_min_dist(P, min_d=0.5):
    """Ensure all pair distances >= min_d by small local pushes."""
    P = np.array(P, dtype=float)
    n = P.shape[0]
    for _ in range(30):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                d = P[i] - P[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-12:
                        d = np.array([1.0, 0, 0])
                    else:
                        d = d / dist
                    push = (min_d - dist) / 2.0 + 0.005
                    P[i] += push * d
                    P[j] -= push * d
                    ok = False
        if ok:
            break
    return P


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice, centered."""
    pts = []
    a = spacing
    basis = [np.array([0, 0, 0]), np.array([0, a/2, a/2]),
             np.array([a/2, 0, a/2]), np.array([a/2, a/2, 0])]
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                base = np.array([i, j, k]) * a
                for b in basis:
                    pts.append(base + b)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _hcp_points(n, spacing=1.0):
    """Generate n points from an HCP lattice, centered."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0)/2.0 * j
                z = c * (k + (1.0/3.0) * ((j % 2) + 1))
                pts.append(np.array([x, y, z]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _bcc_points(n, spacing=1.0):
    """Generate n points from a BCC lattice, centered."""
    pts = []
    a = spacing
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                base = np.array([i, j, k]) * a
                pts.append(base)
                pts.append(base + np.array([a/2, a/2, a/2]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice, centered."""
    pts = []
    a = spacing
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                pts.append(np.array([i, j, k]) * a)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _random_compact(n, spacing=1.0, seed=42):
    """Generate a random compact cluster with minimum distance enforcement."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    max_tries = 30000
    tries = 0
    while len(points) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.52 * spacing for q in points):
                points.append(p)
        tries += 1
    if len(points) < n:
        for i in range(n - len(points)):
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = radius * 0.9
            p = np.array([r * np.sin(phi) * np.cos(theta),
                          r * np.sin(phi) * np.sin(theta),
                          r * np.cos(phi)])
            points.append(p)
    return np.array(points)


def _relax_one(P, potential, parameter):
    """Relax a single configuration, returning (energy, positions)."""
    try:
        energy, positions = relax(P, potential, parameter)
        return energy, positions
    except Exception:
        return np.inf, P


def construct(instance):
    """Multi-start portfolio with perturb-refine cycles."""
    _init_timer()
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", 6))

    rng = np.random.default_rng(size * 1000 + int(parameter))

    # Phase 1: diverse lattice + random initializations
    candidates = []
    candidates.append(_fcc_points(size, 1.0))
    candidates.append(_hcp_points(size, 1.0))
    candidates.append(_bcc_points(size, 1.0))
    candidates.append(_sc_points(size, 1.0))

    # Random compact with varied seeds
    for seed in [size, size + 7, size + 13, size + 23, size + 42]:
        candidates.append(_random_compact(size, 1.0, seed))

    # Perturbed FCC variants
    for scale in [0.08, 0.15]:
        fcc_pert = _fcc_points(size, 1.0) + rng.normal(0, scale, (size, 3))
        candidates.append(fcc_pert)

    best_energy = np.inf
    best_positions = None

    for i, cand in enumerate(candidates):
        if _time_left() < 3.0:
            break
        cand = _enforce_min_dist(cand, 0.5)
        energy, positions = _relax_one(cand, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()

    # Phase 2: perturb-best-and-relax cycles (refinement)
    if best_positions is not None:
        # Decreasing perturbation scales for progressive refinement
        scales = [0.25, 0.18, 0.12, 0.08, 0.05, 0.03]
        for scale in scales:
            if _time_left() < 3.0:
                break
            # Try multiple perturbations at this scale
            n_tries = 3 if scale > 0.1 else 4
            for _ in range(n_tries):
                if _time_left() < 3.0:
                    break
                perturbed = best_positions + rng.normal(0, scale, (size, 3))
                perturbed = _enforce_min_dist(perturbed, 0.5)
                energy, positions = _relax_one(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()

    # Phase 3: if time remains, try larger perturbations from scratch
    if best_positions is not None and _time_left() > 8.0:
        for scale in [0.4, 0.5]:
            if _time_left() < 4.0:
                break
            for _ in range(2):
                if _time_left() < 4.0:
                    break
                perturbed = best_positions + rng.normal(0, scale, (size, 3))
                perturbed = _enforce_min_dist(perturbed, 0.5)
                energy, positions = _relax_one(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()

    if best_positions is None:
        best_positions = _enforce_min_dist(candidates[0], 0.5)

    return best_positions
# EVOLVE-BLOCK-END
