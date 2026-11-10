"""Morse cluster global minima via basin hopping with structured seeds and vectorized energy/gradient."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

# Deadline: evaluator kills at 330s, keep safety margin
_DEADLINE = time.time() + 320.0  # 10s safety margin


def _morse_energy(positions, rho):
    """Compute total Morse potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist = np.maximum(dist, 1e-10)
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    pair_energy = exp_x * (exp_x - 2.0)
    return float(np.sum(pair_energy) / 2.0)


def _morse_energy_and_grad(positions, rho):
    """Compute total Morse potential energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(np.maximum(dist_sq, 1e-20))
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(dist_sq, np.inf)
    
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    # dE/dr = -rho * exp_x * (2*exp_x - 1)
    dE_dr = -rho * exp_x * (2.0 * exp_x - 1.0)
    
    # Gradient: dE/dpos_i = sum_j dE/dr_ij * (pos_i - pos_j) / r_ij
    # dE_dr / dist gives the scalar factor
    factor = dE_dr / dist  # shape (n, n)
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    energy = float(np.sum(exp_x * (exp_x - 2.0)) / 2.0)
    return energy, grad


def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are at least min_dist."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist)) >= min_dist


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    # Generate points on a 3D grid with FCC offsets
    points = []
    # FCC lattice vectors
    a = spacing
    # Generate candidates in a cube
    max_idx = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC: simple cubic + face-centered
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    # Sort by distance from center and take first n
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    points = []
    max_idx = int(np.ceil(n ** (1/3))) + 2
    a = spacing
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # BCC: simple cubic + body-centered
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _generate_icosahedral(n, spacing=1.0):
    """Generate an icosahedral-like cluster (Mackay icosahedron)."""
    # Mackay icosahedron shell structure
    points = []
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0]) * spacing * 0.5
    
    # Shell 0: center
    points.append(np.zeros(3))
    # Shell 1: icosahedron vertices (12 atoms)
    for v in verts:
        points.append(v * spacing * 0.5)
    # Shell 2: truncated icosahedron approx - use more points
    # For simplicity, add points on a sphere
    if n > 13:
        # Add points on a spherical shell
        remaining = n - len(points)
        # Use Fibonacci sphere for uniform distribution
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(remaining):
            z = 1 - 2 * (i + 0.5) / remaining
            r = np.sqrt(1 - z * z)
            theta = golden_angle * i
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * spacing * 1.0
            points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _local_minimize(positions, rho, max_iter=200):
    """Local minimization using L-BFGS-B."""
    def fun(x):
        return _morse_energy(x.reshape(-1, 3), rho)
    
    def jac(x):
        _, g = _morse_energy_and_grad(x.reshape(-1, 3), rho)
        return g.ravel()
    
    x0 = positions.ravel()
    try:
        result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return positions, _morse_energy(positions, rho)


def _basin_hop(positions, rho, n_steps, step_size, rng, max_iter_local=100):
    """Basin hopping: perturb, locally minimize, accept/reject."""
    current = positions.copy()
    current_energy = _morse_energy(current, rho)
    best = current.copy()
    best_energy = current_energy
    
    for step in range(n_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Perturb
        perturbation = rng.normal(0, step_size, size=current.shape)
        candidate = current + perturbation
        
        # Local minimize
        candidate, candidate_energy = _local_minimize(candidate, rho, max_iter=max_iter_local)
        
        if not _is_valid(candidate, 0.5):
            continue
        
        # Metropolis acceptance
        delta_e = candidate_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(1e-10, step_size * 0.1)):
            current = candidate
            current_energy = candidate_energy
        
        if current_energy < best_energy:
            best = current.copy()
            best_energy = current_energy
    
    return best, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations via basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    potential = instance["potential"]
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse initial seeds
    seeds = []
    seeds.append(_generate_fcc(n, spacing=1.0))
    seeds.append(_generate_bcc(n, spacing=1.0))
    seeds.append(_generate_icosahedral(n, spacing=1.0))
    seeds.append(_generate_random_compact(n, rng, spacing=1.0))
    seeds.append(_generate_random_compact(n, rng, spacing=1.0))
    
    best_positions = None
    best_energy = np.inf
    
    # First pass: local minimize each seed
    for seed in seeds:
        if time.time() > _DEADLINE - 30:
            break
        pos, energy = _local_minimize(seed, rho, max_iter=200)
        if _is_valid(pos, 0.5) and energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
    
    # Basin hopping from the best seed
    if best_positions is not None:
        # Multiple basin hopping runs with different step sizes
        step_sizes = [0.3, 0.5, 0.7, 1.0]
        for ss in step_sizes:
            if time.time() > _DEADLINE - 10:
                break
            # Run basin hopping
            bp, be = _basin_hop(best_positions, rho, n_steps=30, step_size=ss, rng=rng, max_iter_local=80)
            if be < best_energy:
                best_energy = be
                best_positions = bp.copy()
        
        # Also try basin hopping from other seeds
        for seed in seeds[1:]:
            if time.time() > _DEADLINE - 10:
                break
            pos, energy = _local_minimize(seed, rho, max_iter=100)
            if not _is_valid(pos, 0.5):
                continue
            bp, be = _basin_hop(pos, rho, n_steps=20, step_size=0.5, rng=rng, max_iter_local=60)
            if be < best_energy:
                best_energy = be
                best_positions = bp.copy()
    
    # Final polish: additional local minimization with tight tolerance
    if best_positions is not None and time.time() < _DEADLINE - 5:
        pos, energy = _local_minimize(best_positions, rho, max_iter=500)
        if _is_valid(pos, 0.5) and energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
    
    if best_positions is None:
        # Fallback: return a valid random configuration
        best_positions = _generate_random_compact(n, rng, spacing=1.0)
        best_positions, _ = _local_minimize(best_positions, rho, max_iter=50)
    
    return best_positions
# EVOLVE-BLOCK-END
