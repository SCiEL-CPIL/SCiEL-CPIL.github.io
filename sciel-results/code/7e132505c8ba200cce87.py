"""Morse cluster global minima search with multi-start basin hopping and local relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 330s kill


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _morse_energy(pos, rho):
    """Compute Morse potential energy: E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)"""
    n = len(pos)
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(d ** 2, axis=2))
    np.fill_diagonal(r, 1e10)
    u = rho * (1.0 - r)
    e = np.exp(u)
    energy = np.sum(e * (e - 2.0))
    return 0.5 * energy


def _morse_energy_gradient(pos, rho):
    """Compute gradient of Morse potential energy."""
    n = len(pos)
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(d ** 2, axis=2))
    np.fill_diagonal(r, 1e10)
    u = rho * (1.0 - r)
    e = np.exp(u)
    # dE/dr_ij = rho * e * (2*e - 2)
    de_dr = rho * e * (2.0 * e - 2.0)
    # Gradient with respect to positions
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if r[i, j] < 1e-9:
                continue
            factor = de_dr[i, j] / r[i, j]
            grad[i] -= factor * d[i, j]
            grad[j] += factor * d[i, j]
    return grad


def _fix_min_dist(pos, min_d=0.501):
    """Separate any atoms closer than min_d by pushing them apart."""
    n = len(pos)
    for _ in range(50):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-9:
                        # Random push
                        d = np.random.randn(3)
                        dist = np.linalg.norm(d)
                    push = (min_d - dist) / dist * d
                    pos[i] -= 0.5 * push
                    pos[j] += 0.5 * push
                    moved = True
        if not moved:
            break
    return pos


def _local_relax(pos, rho):
    """Relax positions using L-BFGS-B with bounds to maintain minimum distance."""
    pos = _fix_min_dist(pos)
    n = pos.shape[0]
    
    def obj_and_grad(x):
        p = x.reshape(n, 3)
        p = _fix_min_dist(p)
        e = _morse_energy(p, rho)
        g = _morse_energy_gradient(p, rho)
        return e, g.flatten()
    
    try:
        result = minimize(obj_and_grad, pos.flatten(), method='L-BFGS-B',
                          jac=True, options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10})
        if result.success or result.fun < _morse_energy(pos, rho):
            return result.x.reshape(n, 3), result.fun
    except Exception:
        pass
    
    # Fallback to chem_clusters.relax
    energy, positions = relax(pos, 'morse', rho)
    return positions, energy


def _make_icosahedral_start(n, rho):
    """Generate a structured starting configuration based on icosahedral shells."""
    # icosahedral radii and counts
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms at r=1
    # Shell 2: 42 atoms at r=sqrt(3)
    # Shell 3: 92 atoms at r=sqrt(5)
    
    # Icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Generate points on shells
    points = [np.zeros(3)]  # center
    
    # Shell 1: 12 icosahedral vertices at r=1
    if n > 1:
        shell1 = ico_verts * 1.0
        for v in shell1:
            if len(points) < n:
                points.append(v.copy())
    
    # Shell 2: 42 points at r=sqrt(3)
    # Use combinations of icosahedral directions
    if n > 13:
        r2 = np.sqrt(3.0)
        # Generate more points on sphere
        rng = np.random.default_rng(42)
        count = min(42, n - len(points))
        for _ in range(count):
            # Random point on sphere
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            points.append(v * r2)
    
    # Shell 3: 92 points at r=sqrt(5)
    if n > 55:
        r3 = np.sqrt(5.0)
        rng = np.random.default_rng(43)
        count = min(92, n - len(points))
        for _ in range(count):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            points.append(v * r3)
    
    # If still not enough, add more random points on outer shells
    if len(points) < n:
        rng = np.random.default_rng(44)
        while len(points) < n:
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            r = np.sqrt(5.0) + rng.uniform(0, 1)
            points.append(v * r)
    
    pos = np.array(points[:n])
    pos = _fix_min_dist(pos)
    return pos


def _random_start(n, rho):
    """Generate a random compact starting configuration."""
    rng = np.random.default_rng()
    radius = n ** (1.0 / 3.0) * 0.8
    
    points = []
    min_dist = 0.51
    max_attempts = n * 100
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    
    # If we couldn't place all atoms, add them with some overlap
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    pos = np.array(points)
    pos = _fix_min_dist(pos)
    return pos


def _basin_hopping_step(pos, rho, step_size):
    """Perform one basin hopping step: random perturbation + local relaxation."""
    n = pos.shape[0]
    rng = np.random.default_rng()
    
    # Random perturbation
    delta = rng.normal(0, step_size, size=(n, 3))
    new_pos = pos + delta
    new_pos = _fix_min_dist(new_pos)
    
    # Local relaxation
    relaxed_pos, energy = _local_relax(new_pos, rho)
    return relaxed_pos, energy


def construct(instance):
    """Find low-energy Morse cluster configuration using multi-start basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Initialize with icosahedral structure
    best_pos = _make_icosahedral_start(size, rho)
    best_energy, _ = _local_relax(best_pos, rho)
    
    # Multi-start basin hopping
    rng = np.random.default_rng(12345)
    step_size = 0.3
    max_iterations = 200
    
    for i in range(max_iterations):
        if _remaining_time() < 30:
            break
        
        # Adaptive step size: decrease if improving, increase if stuck
        if i > 0 and i % 20 == 0:
            step_size = max(0.1, step_size * 0.9)
        
        # Random perturbation from best
        delta = rng.normal(0, step_size, size=(size, 3))
        trial_pos = best_pos + delta
        trial_pos = _fix_min_dist(trial_pos)
        
        # Local relaxation
        trial_pos, trial_energy = _local_relax(trial_pos, rho)
        
        # Accept if better
        if trial_energy < best_energy:
            best_pos = trial_pos
            best_energy = trial_energy
    
    # Final relaxation with more iterations
    best_pos, best_energy = _local_relax(best_pos, rho)
    
    # Ensure minimum distance constraint
    best_pos = _fix_min_dist(best_pos, 0.501)
    
    return best_pos
# EVOLVE-BLOCK-END
