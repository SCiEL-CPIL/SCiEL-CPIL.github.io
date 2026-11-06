"""Morse cluster global minima search with structured initialization and multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # Sum over i < j
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return energy


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # Energy
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    # Gradient
    dE_dr = -rho * ex * (2.0 * ex - 2.0)
    np.fill_diagonal(dE_dr, 0.0)
    inv_r = np.zeros_like(dists)
    mask = dists > 1e-10
    inv_r[mask] = 1.0 / dists[mask]
    w = dE_dr * inv_r
    grad = np.einsum('ij,ijk->ik', w, diff)
    return energy, grad


def _lattice_init(n, rng):
    """Generate a compact lattice-like initial configuration."""
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                p = np.array([i, j, k], dtype=float)
                if len(points) < n:
                    points.append(p)
    if len(points) < n:
        side2 = int(np.ceil(n ** (1/3))) + 4
        for i in range(-side2, side2 + 1):
            for j in range(-side2, side2 + 1):
                for k in range(-side2, side2 + 1):
                    p = np.array([i * 0.5, j * 0.5, k * 0.5], dtype=float)
                    if len(points) < n:
                        points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.05, points.shape)
    return points


def _random_packed_init(n, rng, spacing=0.9):
    """Generate a random packed initial configuration."""
    radius = spacing * n ** (1/3) * 0.9
    points = []
    attempts = 0
    max_attempts = 10000
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < spacing:
                    ok = False
                    break
            if ok:
                points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    points = np.array(points)
    return points


def _fcc_like_init(n, rng):
    """Generate an FCC-like initial configuration."""
    points = []
    a = 1.0
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                basis = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5, 0.5]),
                    np.array([0.5, 0, 0.5]),
                    np.array([0.5, 0.5, 0]),
                ]
                for b in basis:
                    p = a * (np.array([i, j, k]) + b)
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-3, 3, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.03, points.shape)
    return points


def _hcp_like_init(n, rng):
    """Generate an HCP-like initial configuration."""
    points = []
    a = 1.0
    c = 1.633 * a
    for layer in range(-5, 6):
        z = layer * c / 2
        for i in range(-5, 6):
            for j in range(-5, 6):
                offset = 0.0 if layer % 2 == 0 else 0.5
                p = np.array([a * (i + offset), a * j * np.sqrt(3)/2, z])
                if len(points) < n:
                    points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-3, 3, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.03, points.shape)
    return points


def _icosahedral_init(n, rng):
    """Generate an icosahedral shell initial configuration."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_vertices /= np.linalg.norm(ico_vertices[0])
    
    points = [np.zeros(3)]  # center
    # First shell: 12 vertices
    for v in ico_vertices:
        if len(points) < n:
            points.append(v * 1.0)
    # Second shell: 30 vertices (edges of icosahedron)
    if len(points) < n:
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_vertices[i] - ico_vertices[j])
                if 0.9 < d < 1.1:  # approximate edge length
                    mid = (ico_vertices[i] + ico_vertices[j]) / 2
                    mid /= np.linalg.norm(mid)
                    edges.append(mid)
        for v in edges:
            if len(points) < n:
                points.append(v * 1.7)
    # Third shell
    if len(points) < n:
        for v in ico_vertices:
            for s in [1.0, 2.0, 3.0]:
                if len(points) < n:
                    points.append(v * s * 1.8)
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-4, 4, 3)
            points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.03, points.shape)
    return points


def _optimize(pos0, rho, deadline, maxiter=300):
    """Optimize a single configuration."""
    if time.time() > deadline:
        return None, np.inf
    
    def fun(x):
        return _morse_energy(x.reshape(-1, 3), rho)
    
    def jac(x):
        return _morse_energy_grad(x.reshape(-1, 3), rho)[1].flatten()
    
    try:
        result = minimize(
            fun=fun,
            x0=pos0.flatten(),
            jac=jac,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-6, 'maxfun': maxiter * 10}
        )
        pos = result.x.reshape(-1, 3)
        energy = result.fun
        
        # Check minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist >= 0.5:
            return pos, energy
        else:
            return None, np.inf
    except Exception:
        return None, np.inf


def construct(instance):
    """Find global minima of Morse potential clusters."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin
    
    rng = np.random.default_rng(42 + n)
    
    best_energy = np.inf
    best_pos = None
    
    init_generators = [
        _lattice_init,
        _random_packed_init,
        _fcc_like_init,
        _hcp_like_init,
        _icosahedral_init,
    ]
    
    # Phase 1: Quick optimization with diverse initializations
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        gen_func = init_generators[restart % len(init_generators)]
        pos0 = gen_func(n, rng)
        
        pos, energy = _optimize(pos0, rho, deadline, maxiter=200)
        if pos is not None and energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Phase 2: More thorough optimization of best configurations
    if best_pos is not None:
        # Re-optimize the best with more iterations
        pos, energy = _optimize(best_pos, rho, deadline, maxiter=500)
        if pos is not None and energy < best_energy:
            best_energy = energy
            best_pos = pos
        
        # Try perturbed versions of the best
        for pert in range(3):
            if time.time() > deadline - 30:
                break
            pos_pert = best_pos + rng.normal(0, 0.1, best_pos.shape)
            pos, energy = _optimize(pos_pert, rho, deadline, maxiter=300)
            if pos is not None and energy < best_energy:
                best_energy = energy
                best_pos = pos
    
    # If no valid solution found, use a simple fallback
    if best_pos is None:
        pos0 = _random_packed_init(n, rng)
        pos, energy = _optimize(pos0, rho, deadline, maxiter=200)
        if pos is not None:
            best_pos = pos
        else:
            best_pos = pos0
    
    return best_pos
# EVOLVE-BLOCK-END
