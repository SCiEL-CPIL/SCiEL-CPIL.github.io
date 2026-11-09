"""
Morse cluster global minima finder using structured initializers,
vectorized energy/gradient, and basin hopping with time management.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Vectorized Morse energy computation."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask out self-interactions
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    # Morse potential: exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = np.exp(rho * (1.0 - dist))
    energy = np.sum(x * (x - 2.0))
    return energy


def _morse_gradient(positions, rho):
    """Vectorized Morse gradient computation."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Avoid division by zero
    dist = np.where(dist > 1e-10, dist, 1e-10)
    # d/dr of Morse: -rho * exp(rho*(1-r)) * (exp(rho*(1-r)) - 1)
    x = np.exp(rho * (1.0 - dist))
    dr = -rho * x * (x - 1.0)
    # Gradient: sum over j of dr_ij * (r_i - r_j) / |r_i - r_j|
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = diff[i, j]
            grad[i] += dr[i, j] * d / dist[i, j]
            grad[j] -= dr[i, j] * d / dist[i, j]
    return grad


def _local_minimize(positions, rho, max_iter=200, tol=1e-6):
    """
    Local minimization using L-BFGS-like approach with damping.
    Returns (energy, positions).
    """
    pos = positions.copy().astype(np.float64)
    n = len(pos)
    
    # Initial gradient
    grad = _morse_gradient(pos, rho)
    energy = _morse_energy(pos, rho)
    
    # Simple gradient descent with adaptive step size
    step = 0.01
    for it in range(max_iter):
        if np.max(np.abs(grad)) < tol:
            break
        
        # Try step
        new_pos = pos - step * grad
        new_energy = _morse_energy(new_pos, rho)
        
        if new_energy < energy:
            pos = new_pos
            energy = new_energy
            grad = _morse_gradient(pos, rho)
            step = min(step * 1.5, 0.1)
        else:
            step *= 0.5
            if step < 1e-8:
                break
    
    return energy, pos


def _generate_bcc_lattice(n, spacing=1.0):
    """Generate n atoms in BCC lattice structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    # BCC: simple cubic + body centers
    # Estimate how many shell layers we need
    max_coord = int(np.ceil((n / 2.0) ** (1.0/3.0))) + 2
    
    # Generate candidate points sorted by distance from origin
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Simple cubic lattice point
                p = np.array([i, j, k]) * spacing
                candidates.append((np.dot(p, p), p))
                # BCC body center
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * spacing
                candidates.append((np.dot(p2, p2), p2))
    
    # Sort by distance from origin
    candidates.sort(key=lambda x: x[0])
    
    # Select n points with minimum separation
    min_dist = 0.45 * spacing
    for dist_sq, p in candidates:
        if len(points) >= n:
            break
        # Check minimum distance
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    # If we didn't get enough, add more with relaxed constraint
    if len(points) < n:
        min_dist = 0.35 * spacing
        for dist_sq, p in candidates:
            if len(points) >= n:
                break
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    
    if len(points) < n:
        # Fallback: random
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-max_coord*spacing, max_coord*spacing, 3)
            points.append(p)
    
    return np.array(points[:n])


def _generate_fcc_lattice(n, spacing=1.0):
    """Generate n atoms in FCC-like shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 3
    
    # FCC lattice: points at (i,j,k) where i+j+k is even, scaled
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * spacing * 0.7071  # 1/sqrt(2)
                    candidates.append((np.dot(p, p), p))
    
    candidates.sort(key=lambda x: x[0])
    
    min_dist = 0.45 * spacing
    for dist_sq, p in candidates:
        if len(points) >= n:
            break
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    if len(points) < n:
        min_dist = 0.35 * spacing
        for dist_sq, p in candidates:
            if len(points) >= n:
                break
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-max_coord*spacing, max_coord*spacing, 3)
            points.append(p)
    
    return np.array(points[:n])


def _generate_icosahedral(n, spacing=1.0):
    """Generate n atoms in icosahedral shell structure (for small n)."""
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * spacing / (2 * phi)
    
    points = [np.zeros(3)]  # center
    points.extend([v.copy() for v in ico_verts])  # 12 vertices
    
    # Second shell: octahedron + additional
    if n > 13:
        # Octahedron vertices
        oct_verts = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ]) * spacing * 1.2
        for v in oct_verts:
            if len(points) < n:
                points.append(v.copy())
    
    # Add more points if needed
    if len(points) < n:
        rng = np.random.default_rng(42)
        radius = spacing * n ** (1.0/3.0) * 0.6
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in points):
                points.append(p)
    
    return np.array(points[:n])


def _basin_hop(positions, rho, n_hops=50, hop_scale=0.5, time_limit=None):
    """
    Basin hopping: perturb, minimize, accept if lower energy.
    """
    pos = positions.copy().astype(np.float64)
    energy = _morse_energy(pos, rho)
    best_energy = energy
    best_pos = pos.copy()
    
    start_time = time.time()
    rng = np.random.default_rng(hash(tuple(map(tuple, pos)))) % (2**31)
    
    for hop in range(n_hops):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Perturb
        noise = rng.normal(0, hop_scale, pos.shape)
        perturbed = pos + noise
        
        # Minimize
        new_energy, new_pos = _local_minimize(perturbed, rho, max_iter=100)
        
        # Accept if better
        if new_energy < energy:
            pos = new_pos
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy arrangement for Morse cluster.
    Uses multiple structured initializers, local minimization, and basin hopping.
    """
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Time management: 300s total for 6 instances, ~48s per instance
    # But we need to be safe, so use ~45s per instance
    instance_start = time.time()
    time_budget = 45.0
    
    # Generate multiple structured initializers
    initializers = []
    
    # BCC lattice
    initializers.append(_generate_bcc_lattice(n))
    
    # FCC lattice
    initializers.append(_generate_fcc_lattice(n))
    
    # Icosahedral (good for small n)
    if n <= 50:
        initializers.append(_generate_icosahedral(n))
    
    # Random compact
    rng = np.random.default_rng(n)
    radius = n ** (1.0/3.0) * 0.7
    for _ in range(3):
        pts = []
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                pts.append(p)
        initializers.append(np.array(pts))
    
    # Local minimize each initializer
    best_energy = np.inf
    best_pos = None
    
    for i, init in enumerate(initializers):
        elapsed = time.time() - instance_start
        if elapsed > time_budget * 0.5:
            break
        
        energy, pos = _local_minimize(init, rho, max_iter=150)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    # Basin hopping from best local minimum
    elapsed = time.time() - instance_start
    if elapsed < time_budget * 0.8 and best_pos is not None:
        hop_time_left = time_budget * 0.8 - elapsed
        bh_energy, bh_pos = _basin_hop(best_pos, rho, n_hops=30, hop_scale=0.4, time_limit=hop_time_left)
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Final check: ensure minimum distance constraint
    if best_pos is not None:
        # Verify no atoms are too close
        for i in range(len(best_pos)):
            for j in range(i+1, len(best_pos)):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < 0.5:
                    # Nudge apart
                    direction = best_pos[i] - best_pos[j]
                    if np.linalg.norm(direction) < 1e-10:
                        direction = np.array([1.0, 0, 0])
                    direction = direction / np.linalg.norm(direction)
                    shift = (0.5 - d + 0.01) / 2
                    best_pos[i] += shift * direction
                    best_pos[j] -= shift * direction
    
    return best_pos if best_pos is not None else np.zeros((n, 3))
# EVOLVE-BLOCK-END
