"""Improved program for Morse cluster global minima using lattice initialization, diverse restarts, and time management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # Safety margin for 300s limit


def _time_left():
    """Return seconds remaining before hard deadline."""
    return _HARD_DEADLINE - time.time()


def _morse_energy(positions, rho):
    """Compute Morse energy for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Compute all pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Compute energy
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    energy = np.sum(exp_x * (exp_x - 2.0))
    return 0.5 * energy  # Each pair counted twice


def _morse_energy_gradient(positions, rho):
    """Compute Morse energy gradient."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            r = positions[i] - positions[j]
            rij = np.linalg.norm(r)
            if rij < 1e-10:
                continue
            x = rho * (1.0 - rij)
            exp_x = np.exp(x)
            # dE/dr = 2 * exp(x) * (exp(x) - 1) * rho
            dE_dr = 2.0 * exp_x * (exp_x - 1.0) * rho
            # grad_i = -dE_dr * r_hat, grad_j = dE_dr * r_hat
            r_hat = r / rij
            grad[i] -= dE_dr * r_hat
            grad[j] += dE_dr * r_hat
    return grad


def _enforce_min_dist(positions, min_dist=0.5, max_iter=1000):
    """Push apart any atoms closer than min_dist."""
    positions = positions.copy()
    n = len(positions)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    push = (min_dist - dist) / 2.0
                    d_hat = d / dist
                    positions[i] += push * d_hat
                    positions[j] -= push * d_hat
                    moved = True
                elif dist <= 1e-10:
                    # Random small displacement
                    positions[i] += np.random.randn(3) * 0.1
                    moved = True
        if not moved:
            break
    return positions


def _fcc_points(n, spacing=1.0):
    """Generate n points in an FCC-like arrangement."""
    points = []
    # Generate FCC lattice points
    a = spacing
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                # FCC basis
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + bx, j + by, k + bz]) * a
                    points.append(p)
    points = np.array(points)
    # Center at origin
    points -= np.mean(points, axis=0)
    # Sort by distance from center and take first n
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _bcc_points(n, spacing=1.0):
    """Generate n points in a BCC-like arrangement."""
    points = []
    a = spacing
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                # BCC basis
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = np.array([i + bx, j + by, k + bz]) * a
                    points.append(p)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _hcp_points(n, spacing=1.0):
    """Generate n points in an HCP-like arrangement."""
    points = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-3, 4):
                # HCP: layers at z = k*c/2, alternating AB
                z = k * c / 2.0
                if k % 2 == 0:
                    # A layer
                    for bx, by in [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]:
                        p = np.array([i + bx, j + by, z]) * a
                        points.append(p)
                else:
                    # B layer
                    for bx, by in [(0.5, 0), (0, 0.5), (1, 0.5), (0.5, 1), (0.5, 0.5)]:
                        p = np.array([i + bx, j + by, z]) * a
                        points.append(p)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _random_compact(n, spacing=1.0, seed=None):
    """Generate n random points in a compact sphere with minimum distance."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    max_tries = 10000
    for _ in range(max_tries):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
    # If we couldn't get enough, fill with random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    points = np.array(points[:n])
    return points


def _relax_with_scipy(positions, potential, rho, max_iter=2000):
    """Relax using scipy minimize with L-BFGS-B."""
    n = len(positions)
    
    def objective(x):
        pos = x.reshape(n, 3)
        e = _morse_energy(pos, rho)
        return e
    
    def gradient(x):
        pos = x.reshape(n, 3)
        g = _morse_energy_gradient(pos, rho)
        return g.ravel()
    
    result = minimize(
        objective,
        positions.ravel(),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.fun, result.x.reshape(n, 3)


def _perturb_refine(positions, rho, potential, scales, rng, max_iter_per_scale=500):
    """Perturb with decreasing scales and refine."""
    best_pos = positions.copy()
    best_e = _morse_energy(positions, rho)
    n = len(positions)
    
    for scale in scales:
        for _ in range(3):  # 3 perturbations per scale
            perturbed = best_pos + rng.normal(0, scale, (n, 3))
            perturbed = _enforce_min_dist(perturbed, 0.5)
            try:
                e, pos = relax(perturbed, potential, rho)
                if e < best_e:
                    best_e = e
                    best_pos = pos.copy()
            except:
                pass
    return best_e, best_pos


def construct(instance):
    """Find low-energy arrangements using diverse lattice seeds and perturb-refine."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    if _time_left() < 10:
        # Not enough time, return a quick valid result
        pts = _random_compact(n, 1.0, seed=n)
        pts = _enforce_min_dist(pts, 0.5)
        return pts
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse initial configurations
    candidates = []
    
    # FCC
    try:
        fcc = _fcc_points(n, 1.0)
        fcc = _enforce_min_dist(fcc, 0.5)
        candidates.append(fcc)
    except:
        pass
    
    # BCC
    try:
        bcc = _bcc_points(n, 1.0)
        bcc = _enforce_min_dist(bcc, 0.5)
        candidates.append(bcc)
    except:
        pass
    
    # HCP
    try:
        hcp = _hcp_points(n, 1.0)
        hcp = _enforce_min_dist(hcp, 0.5)
        candidates.append(hcp)
    except:
        pass
    
    # Random compact (multiple)
    for seed in [n, n + 1, n + 2, n + 3]:
        try:
            rc = _random_compact(n, 1.0, seed=seed)
            rc = _enforce_min_dist(rc, 0.5)
            candidates.append(rc)
        except:
            pass
    
    # Relax each candidate
    best_e = np.inf
    best_pos = candidates[0] if candidates else _random_compact(n, 1.0, seed=0)
    
    for i, cand in enumerate(candidates):
        if _time_left() < 15:
            break
        try:
            e, pos = relax(cand, potential, rho)
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
        except:
            pass
    
    # Perturb-refine on the best
    if _time_left() > 30:
        scales = [0.3, 0.2, 0.1, 0.05]
        try:
            e, pos = _perturb_refine(best_pos, rho, potential, scales, rng)
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
        except:
            pass
    
    # Additional random restarts if time permits
    if _time_left() > 20:
        for seed in [n + 10, n + 20, n + 30]:
            if _time_left() < 10:
                break
            try:
                rc = _random_compact(n, 1.0, seed=seed)
                rc = _enforce_min_dist(rc, 0.5)
                e, pos = relax(rc, potential, rho)
                if e < best_e:
                    best_e = e
                    best_pos = pos.copy()
            except:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
