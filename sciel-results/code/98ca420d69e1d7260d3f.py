"""Morse cluster global minima search with multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # safety margin before 330s kill


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _morse_energy(pos, rho):
    """Compute Morse potential energy efficiently."""
    n = len(pos)
    if n < 2:
        return 0.0
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(d ** 2, axis=2))
    np.fill_diagonal(r, 1e10)
    u = rho * (1.0 - r)
    e = np.exp(u)
    energy = np.sum(e * (e - 2.0))
    return 0.5 * energy


def _fix_min_dist(pos, min_d=0.501):
    """Separate any atoms closer than min_d using simple repulsion."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(50):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-8:
                        # Random small displacement
                        pos[j] += np.random.randn(3) * 0.01
                    else:
                        push = (min_d - dist) / dist * 0.5
                        pos[i] += push * d
                        pos[j] -= push * d
                    moved = True
        if not moved:
            break
    return pos


def _make_icosahedral_shell(n, rng):
    """Generate points on concentric shells with icosahedral symmetry."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    points = []
    # Start with center if n is large enough
    shells = []
    remaining = n
    # Use multiple shells
    if n >= 13:
        points.append(np.zeros(3))
        remaining -= 1
        shells.append(1.0)  # First shell radius
    if remaining >= 12:
        for v in ico:
            points.append(v * 1.0)
        remaining -= 12
        shells.append(1.8)
    if remaining >= 30:
        # Second shell - use octahedron-like
        octa = np.array([
            [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
            [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
            [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
            [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]
        ], dtype=float)
        octa = octa / np.linalg.norm(octa, axis=1, keepdims=True)
        for v in octa[:remaining]:
            points.append(v * 1.8)
        remaining -= min(30, remaining)
        shells.append(2.5)
    
    # Fill remaining with random points on expanding shells
    while len(points) < n:
        shell_idx = len(shells)
        if shell_idx == 0:
            r = 1.0
        else:
            r = 1.0 + 0.8 * (shell_idx - 1)
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p) * r
        points.append(p)
        if len(points) % 10 == 0:
            shells.append(r + 0.8)
    
    return np.array(points[:n])


def _make_random_cluster(n, rng, spacing=1.0):
    """Generate a random compact cluster with minimum distance constraint."""
    points = []
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    attempts = 0
    while len(points) < n and attempts < n * 1000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.7 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
    # If we couldn't place enough, fill with random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _optimize(pos, rho, max_time=30.0):
    """Optimize a single starting configuration."""
    start_t = time.time()
    pos = _fix_min_dist(pos)
    
    # Use L-BFGS-B with gradient
    try:
        result = minimize(
            fun=lambda x: _morse_energy(x.reshape(-1, 3), rho),
            x0=pos.flatten(),
            jac=lambda x: morse_energy_gradient(x.reshape(-1, 3), rho).flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        pos_opt = result.x.reshape(-1, 3)
        pos_opt = _fix_min_dist(pos_opt)
        energy = _morse_energy(pos_opt, rho)
        return energy, pos_opt
    except Exception:
        return _morse_energy(pos, rho), pos


def construct(instance):
    """Multi-start global optimization for Morse clusters."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance["potential"]
    
    best_energy = np.inf
    best_positions = None
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Icosahedral-based start
    starts.append(_make_icosahedral_shell(n, rng))
    
    # 2-5. Random compact clusters
    for _ in range(4):
        starts.append(_make_random_cluster(n, rng))
    
    # 6. FCC-like lattice fragment
    a = 1.0
    fcc_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                fcc_points.append(np.array([i*a, j*a, k*a]))
                fcc_points.append(np.array([i*a, (j+0.5)*a, (k+0.5)*a]))
                fcc_points.append(np.array([(i+0.5)*a, j*a, (k+0.5)*a]))
                fcc_points.append(np.array([(i+0.5)*a, (j+0.5)*a, k*a]))
    fcc_points = np.array(fcc_points)
    # Keep only points within a sphere
    center = fcc_points.mean(axis=0)
    fcc_points = fcc_points - center
    mask = np.linalg.norm(fcc_points, axis=1) <= 2.5
    fcc_points = fcc_points[mask]
    if len(fcc_points) >= n:
        idx = rng.choice(len(fcc_points), n, replace=False)
        starts.append(fcc_points[idx])
    elif len(fcc_points) > 0:
        # Pad with random
        pad = _make_random_cluster(n - len(fcc_points), rng)
        starts.append(np.vstack([fcc_points, pad]))
    
    # Optimize each start
    for i, start in enumerate(starts):
        if _remaining_time() < 15:
            break
        try:
            energy, pos_opt = _optimize(start, rho, max_time=30.0)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos_opt
        except Exception:
            pass
    
    # If we have time, do more random restarts
    extra_restarts = 0
    while _remaining_time() > 20 and extra_restarts < 10:
        start = _make_random_cluster(n, rng)
        try:
            energy, pos_opt = _optimize(start, rho, max_time=20.0)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos_opt
        except Exception:
            pass
        extra_restarts += 1
    
    if best_positions is None:
        # Fallback: simple random
        best_positions = _make_random_cluster(n, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
