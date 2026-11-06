"""Morse cluster global minima search with multi-start local optimization."""
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


def _fix_min_dist(pos, min_d=0.501, max_iter=100):
    """Separate any atoms closer than min_d using iterative push."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_d and dist > 1e-10:
                    push = (min_d - dist) / dist * d
                    pos[i] += 0.5 * push
                    pos[j] -= 0.5 * push
                    moved = True
        if not moved:
            break
    return pos


def _make_icosahedral_start(n, rng):
    """Generate an icosahedral-like starting configuration."""
    # Start with a regular icosahedron (12 vertices) and add more shells
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    # Normalize to unit distance
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    base = base / norms
    
    points = [base[0].copy()]
    remaining = n - 1
    
    # Add remaining points on concentric shells
    shell_radii = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    shell_idx = 0
    while remaining > 0 and shell_idx < len(shell_radii):
        r = shell_radii[shell_idx]
        # Generate points on sphere of radius r
        max_on_shell = int(4 * np.pi * r * r / (0.8 ** 2))
        count = min(remaining, max_on_shell)
        for _ in range(count):
            # Random direction on sphere
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            xy = np.sqrt(1 - z * z)
            p = r * np.array([xy * np.cos(theta), xy * np.sin(theta), z])
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
                remaining -= 1
                if remaining == 0:
                    break
        shell_idx += 1
    
    # If we still need more points, add them randomly
    while remaining > 0:
        p = rng.uniform(-5, 5, 3)
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
            remaining -= 1
    
    return np.array(points[:n])


def _make_random_start(n, rng, radius=None):
    """Generate a random compact starting configuration."""
    if radius is None:
        radius = n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 50:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place enough, add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _optimize(pos, rho, time_budget=30.0):
    """Locally optimize a configuration using L-BFGS-B."""
    start = time.time()
    pos = _fix_min_dist(pos)
    
    def objective(x):
        p = x.reshape(-1, 3)
        p = _fix_min_dist(p)
        return _morse_energy(p, rho)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        # Use the provided gradient function
        try:
            _, grad = morse_energy_gradient(p, rho)
            return grad.flatten()
        except:
            # Numerical gradient fallback
            eps = 1e-6
            g = np.zeros_like(x)
            for i in range(len(x)):
                xp = x.copy()
                xp[i] += eps
                xm = x.copy()
                xm[i] -= eps
                g[i] = (objective(xp) - objective(xm)) / (2 * eps)
            return g
    
    result = minimize(
        objective,
        pos.flatten(),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-6}
    )
    
    elapsed = time.time() - start
    if result.success:
        return result.fun, result.x.reshape(-1, 3), elapsed
    else:
        # Return best found even if not converged
        return result.fun, result.x.reshape(-1, 3), elapsed


def construct(instance):
    """Multi-start local optimization for Morse cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Icosahedral-like start
    starts.append(_make_icosahedral_start(size, rng))
    
    # 2. Several random compact starts
    for i in range(5):
        radius = size ** (1.0 / 3.0) * (0.6 + 0.2 * i)
        starts.append(_make_random_start(size, rng, radius))
    
    # 3. A few more with different random seeds
    rng2 = np.random.default_rng(123)
    starts.append(_make_random_start(size, rng2, size ** (1.0 / 3.0) * 0.8))
    rng3 = np.random.default_rng(789)
    starts.append(_make_random_start(size, rng3, size ** (1.0 / 3.0) * 1.0))
    
    # Optimize each start, tracking time
    for start_pos in starts:
        if _remaining_time() < 10.0:
            break
        
        # Allocate time budget based on remaining time and number of starts left
        starts_left = len(starts) - len([s for s in starts if s is start_pos or not hasattr(s, '_done')])
        time_budget = min(25.0, _remaining_time() / max(1, starts_left))
        
        try:
            energy, positions, elapsed = _optimize(start_pos, rho, time_budget)
            positions = _fix_min_dist(positions)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            pass
    
    # If we still have time, do a few more random restarts
    while _remaining_time() > 15.0:
        start_pos = _make_random_start(size, rng, size ** (1.0 / 3.0) * 0.7)
        try:
            energy, positions, elapsed = _optimize(start_pos, rho, time_budget=10.0)
            positions = _fix_min_dist(positions)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            pass
    
    if best_positions is None:
        # Fallback: return the first start
        best_positions = _fix_min_dist(starts[0])
    
    return best_positions
# EVOLVE-BLOCK-END
