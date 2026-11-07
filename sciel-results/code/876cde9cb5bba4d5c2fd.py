"""Sutton-Chen transition-metal cluster global minima search.
Uses basin hopping with icosahedral/FCC/BCC seeds for diverse exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_positions(n):
    """Generate positions on icosahedral shells for transition metal clusters.
    
    Icosahedral shell sizes: 1, 12, 42, 92, 162, 252, ...
    For n atoms, fill shells from center outward.
    """
    # Icosahedral shell vertex positions
    # Shell 0: 1 atom at origin
    # Shell 1: 12 atoms (vertices of icosahedron)
    # Shell 2: 42 atoms (next icosahedral shell)
    # Shell 3: 92 atoms
    
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    ico_vertices /= np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    # Shell 2: 42 positions (icosahedral shell)
    # These are the face centers and edge midpoints of the icosahedron
    # For simplicity, generate a denser icosahedral-like shell
    # Use a Fibonacci sphere for approximate uniform distribution
    def fibonacci_sphere(n_points, radius=1.0):
        """Generate approximately uniformly distributed points on a sphere."""
        points = np.zeros((n_points, 3))
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            z = 1 - 2 * i / (n_points - 1) if n_points > 1 else 0
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i
            points[i] = [np.cos(theta) * r, np.sin(theta) * r, z] * radius
        return points
    
    # Build shells
    shell_sizes = [1, 12, 42, 92, 162, 252, 364]
    shell_radii = [0.0, 1.0, 1.9, 2.7, 3.4, 4.0, 4.6]
    
    positions = []
    remaining = n
    
    for shell_idx, (size, radius) in enumerate(zip(shell_sizes, shell_radii)):
        if remaining <= 0:
            break
        take = min(remaining, size)
        if shell_idx == 0:
            positions.append(np.zeros(3))
        elif shell_idx == 1:
            # Use actual icosahedron vertices
            ico_shell = ico_vertices[:take] * radius
            positions.extend(ico_shell.tolist())
        else:
            # Use Fibonacci sphere for higher shells
            if take == 1:
                positions.append([0, 0, radius])
            else:
                fib_pts = fibonacci_sphere(take, radius)
                positions.extend(fib_pts.tolist())
        remaining -= take
    
    # If we still need more atoms, add them on a larger shell
    if remaining > 0:
        next_radius = shell_radii[-1] + 0.8
        fib_pts = fibonacci_sphere(remaining, next_radius)
        positions.extend(fib_pts.tolist())
    
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _fcc_positions(n):
    """Generate n close-packed positions on an FCC lattice, then center."""
    a = 1.0
    max_k = int(np.ceil((2 * n) ** (1.0 / 3.0)))
    pts = []
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            for k in range(-max_k, max_k + 1):
                for f in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + f[0], j + f[1], k + f[2]]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    positions = pts[idx]
    positions -= np.mean(positions, axis=0)
    return positions


def _bcc_positions(n):
    """Generate n positions on a BCC lattice."""
    a = 1.0
    max_k = int(np.ceil((2 * n) ** (1.0 / 3.0)))
    pts = []
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            for k in range(-max_k, max_k + 1):
                for f in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = np.array([i + f[0], j + f[1], k + f[2]]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    positions = pts[idx]
    positions -= np.mean(positions, axis=0)
    return positions


def _random_compact(n, rng, spacing=0.72):
    """Random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _ensure_valid(positions, min_dist=0.35):
    """Check that all positions are finite and no two are closer than min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def _basin_hopping(positions, potential, kind, rng, max_iter=20, perturb_scale=0.3,
                   deadline=None):
    """Basin hopping: relax, perturb, relax. Returns best energy and positions found."""
    best_energy, best_pos = relax(positions, potential, kind)
    current_energy = best_energy
    current_pos = best_pos.copy()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        noise = rng.normal(0, perturb_scale, current_pos.shape)
        perturbed = current_pos + noise
        
        if not _ensure_valid(perturbed):
            continue
        
        # Relax
        try:
            energy, new_pos = relax(perturbed, potential, kind)
        except Exception:
            continue
        
        if np.isfinite(energy):
            if energy < current_energy:
                # Accept
                current_energy = energy
                current_pos = new_pos.copy()
            elif rng.random() < np.exp(-(energy - current_energy) / (0.1 + 0.01 * i)):
                # Metropolis accept
                current_energy = energy
                current_pos = new_pos.copy()
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Basin hopping with diverse icosahedral/FCC/BCC seeds."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = start_time + 315.0  # 15s safety margin
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial seeds
    seeds = []
    seed_types = []
    
    # Icosahedral seed (most important for transition metals)
    try:
        pos_ico = _icosahedron_positions(size)
        if _ensure_valid(pos_ico):
            seeds.append(pos_ico)
            seed_types.append('ico')
    except Exception:
        pass
    
    # FCC seed
    try:
        pos_fcc = _fcc_positions(size)
        if _ensure_valid(pos_fcc):
            seeds.append(pos_fcc)
            seed_types.append('fcc')
    except Exception:
        pass
    
    # BCC seed
    try:
        pos_bcc = _bcc_positions(size)
        if _ensure_valid(pos_bcc):
            seeds.append(pos_bcc)
            seed_types.append('bcc')
    except Exception:
        pass
    
    # Random compact seeds
    for i in range(4):
        rng_seed = np.random.default_rng(1000 + i * 777)
        pos_rand = _random_compact(size, rng_seed)
        if _ensure_valid(pos_rand):
            seeds.append(pos_rand)
            seed_types.append(f'rand{i}')
    
    # Run basin hopping on each seed
    for idx, (seed_pos, stype) in enumerate(zip(seeds, seed_types)):
        if time.time() > deadline:
            break
        
        # Allocate time per seed: more time for icosahedral
        if stype == 'ico':
            max_iter = 30
            perturb_scale = 0.25
        elif stype == 'fcc':
            max_iter = 25
            perturb_scale = 0.3
        elif stype == 'bcc':
            max_iter = 20
            perturb_scale = 0.3
        else:
            max_iter = 15
            perturb_scale = 0.35
        
        # Adjust for remaining time
        remaining_time = deadline - time.time()
        n_seeds_left = len(seeds) - idx
        time_per_seed = remaining_time / max(n_seeds_left, 1)
        
        # Estimate: each basin hopping iteration takes ~1-3 seconds for these sizes
        # Cap iterations based on available time
        max_iter = min(max_iter, max(3, int(time_per_seed / 1.5)))
        
        rng = np.random.default_rng(42 + size + idx * 1000)
        
        try:
            energy, positions = _basin_hopping(
                seed_pos, potential, kind, rng,
                max_iter=max_iter,
                perturb_scale=perturb_scale,
                deadline=deadline
            )
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # If time remains, do a final polish with small perturbations on best
    if best_positions is not None and time.time() < deadline - 10.0:
        rng = np.random.default_rng(999 + size)
        try:
            energy, positions = _basin_hopping(
                best_positions, potential, kind, rng,
                max_iter=10,
                perturb_scale=0.1,
                deadline=deadline
            )
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        best_positions = _fcc_positions(size)
    
    if not _ensure_valid(best_positions):
        best_positions = _fcc_positions(size)
    
    return best_positions
# EVOLVE-BLOCK-END
