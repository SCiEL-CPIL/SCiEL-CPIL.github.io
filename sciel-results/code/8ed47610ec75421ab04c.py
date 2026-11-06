"""Morse cluster global minima: structured seeds + basin hopping with gradient relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Compute Morse energy for positions."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    exp_term = np.exp(rho * (1.0 - dist))
    e = exp_term * (exp_term - 2.0)
    return 0.5 * np.sum(e)


def _morse_grad(pos, rho):
    """Compute Morse energy gradient."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    exp_term = np.exp(rho * (1.0 - dist))
    # dE/dr_ij = 2*exp_term*(exp_term - 1)*rho (per pair, sign handled by diff)
    # gradient on atom i: sum_j dE/dr_ij * (r_i - r_j)/r_ij
    grad_factor = 2.0 * rho * exp_term * (exp_term - 1.0)
    np.fill_diagonal(grad_factor, 0.0)
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-8, dist, 1e8)
    unit = diff / dist_safe[:, :, np.newaxis]
    grad = np.sum(grad_factor[:, :, np.newaxis] * unit, axis=1)
    return grad


def _relax_grad(pos, rho, max_iter=200, tol=1e-8):
    """Gradient-based relaxation with adaptive step size."""
    pos = pos.copy()
    n = len(pos)
    if n < 2:
        return pos, _morse_energy(pos, rho)
    
    step = 0.01
    best_energy = _morse_energy(pos, rho)
    
    for it in range(max_iter):
        if _time_left() < 0.5:
            break
        grad = _morse_grad(pos, rho)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive step: try to reduce energy
        for _ in range(10):
            new_pos = pos - step * grad
            new_energy = _morse_energy(new_pos, rho)
            if new_energy < best_energy:
                pos = new_pos
                best_energy = new_energy
                break
            else:
                step *= 0.5
        
        if step < 1e-10:
            break
    
    return pos, best_energy


def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points."""
    points = []
    # FCC: face-centered cubic
    # Generate in a cube and keep nearest n
    max_coord = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    candidates.append((x + bx, y + by, z + bz))
    
    # Sort by distance from origin
    candidates = sorted(candidates, key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    for p in candidates[:n]:
        points.append([p[0] * spacing, p[1] * spacing, p[2] * spacing])
    
    return np.array(points[:n])


def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points."""
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                # BCC basis: (0,0,0), (0.5,0.5,0.5)
                for bx, by, bz in [(0,0,0), (0.5,0.5,0.5)]:
                    candidates.append((x + bx, y + by, z + bz))
    
    candidates = sorted(candidates, key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    for p in candidates[:n]:
        points.append([p[0] * spacing, p[1] * spacing, p[2] * spacing])
    
    return np.array(points[:n])


def _icosahedral_points(n, spacing=1.0):
    """Generate icosahedral shell points."""
    points = [[0.0, 0.0, 0.0]]
    
    # Icosahedron vertices (unit sphere)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # First shell
    for v in ico_verts:
        points.append(v * spacing)
    
    # Second shell (20 vertices)
    # Midpoints of edges scaled
    second_shell = []
    for i in range(12):
        for j in range(i+1, 12):
            mid = (ico_verts[i] + ico_verts[j]) / 2
            mid /= np.linalg.norm(mid)
            second_shell.append(mid * spacing * 1.618)
    # Deduplicate approximately
    for p in second_shell:
        if all(np.linalg.norm(np.array(p) - np.array(q)) > 0.1 * spacing for q in points):
            points.append(p)
    
    # Third shell if needed
    if len(points) < n:
        # Add more points on a sphere
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.normal(size=3)
            p /= np.linalg.norm(p)
            p = p * spacing * 2.5
            if all(np.linalg.norm(p - np.array(q)) >= 0.5 * spacing for q in points):
                points.append(p)
    
    return np.array(points[:n])


def _random_compact(n, spacing=1.0, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = n * 100
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # Fill remaining with small perturbations
    while len(points) < n:
        base = points[len(points) % len(points)] if points else [0,0,0]
        p = np.array(base) + rng.normal(0, 0.1 * spacing, 3)
        points.append(p)
    
    return np.array(points[:n])


def _basin_hop(pos, rho, temp, rng):
    """One basin hopping step: perturb, relax, accept/reject."""
    n = len(pos)
    # Perturbation magnitude
    kick = 0.15 * np.sqrt(temp)
    new_pos = pos + rng.normal(0, kick, pos.shape)
    
    # Relax
    relaxed_pos, energy = _relax_grad(new_pos, rho, max_iter=100)
    return relaxed_pos, energy


def construct(instance):
    """Find low-energy Morse cluster using structured seeds + basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    if _time_left() < 1.0:
        # Emergency: return a simple compact arrangement
        pos = _random_compact(size, 1.0, seed=size)
        return pos
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse seeds
    seeds = []
    
    # FCC
    try:
        seeds.append(_fcc_points(size, 1.0))
    except Exception:
        pass
    
    # BCC
    try:
        seeds.append(_bcc_points(size, 1.0))
    except Exception:
        pass
    
    # Icosahedral
    try:
        seeds.append(_icosahedral_points(size, 1.0))
    except Exception:
        pass
    
    # Random compact (multiple)
    for s in range(3):
        try:
            seeds.append(_random_compact(size, 1.0, seed=1000 + s))
        except Exception:
            pass
    
    # Relax all seeds
    best_pos = None
    best_energy = np.inf
    
    for i, seed in enumerate(seeds):
        if _time_left() < 2.0:
            break
        try:
            relaxed, energy = _relax_grad(seed, rho, max_iter=150)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
        except Exception:
            continue
    
    if best_pos is None:
        # Fallback
        best_pos = _random_compact(size, 1.0, seed=size)
        best_energy = _morse_energy(best_pos, rho)
    
    # Basin hopping from best position
    temp = 1.0
    current_pos = best_pos.copy()
    current_energy = best_energy
    iteration = 0
    
    while _time_left() > 3.0:
        iteration += 1
        
        # Generate a few perturbations per iteration
        for _ in range(3):
            if _time_left() < 2.0:
                break
            
            try:
                new_pos, new_energy = _basin_hop(current_pos, rho, temp, rng)
                
                # Metropolis acceptance
                delta_e = new_energy - current_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / (temp * 0.1)):
                    current_pos = new_pos
                    current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
            except Exception:
                continue
        
        # Cool down
        temp *= 0.95
    
    return best_pos
# EVOLVE-BLOCK-END
