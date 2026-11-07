"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_remaining(deadline, margin=5.0):
    return deadline - time.time() - margin


def _icosahedron_vertices():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _fib_sphere(n_pts, radius, rng):
    """Generate n_pts points on a sphere of given radius using Fibonacci lattice."""
    if n_pts <= 0:
        return np.empty((0, 3))
    if n_pts == 1:
        return np.array([[0.0, 0.0, radius]])
    
    # Golden angle
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y = np.linspace(1.0, -1.0, n_pts)
    radius_per_y = np.sqrt(1.0 - y**2)
    theta = golden_angle * np.arange(n_pts)
    
    x = np.cos(theta) * radius_per_y
    z = np.sin(theta) * radius_per_y
    
    # Add small random perturbation to break symmetry
    perturbation = rng.uniform(-0.05, 0.05, (n_pts, 3))
    points = np.column_stack([x, y, z]) * radius + perturbation
    
    return points


def _generate_icosahedral(n, spacing=0.72, seed=0):
    """Generate positions based on icosahedral shells."""
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (12 vertices)
    ico_verts = _icosahedron_vertices() * spacing
    
    # Build shells
    positions = []
    
    # Shell 0: center (1 atom)
    positions.append(np.array([0.0, 0.0, 0.0]))
    remaining = n - 1
    
    # Shell 1: 12 atoms at icosahedron vertices
    if remaining >= 12:
        positions.extend(ico_verts.tolist())
        remaining -= 12
    
    # Shell 2: 30 atoms (edges of icosahedron, at distance ~spacing*sqrt(2))
    if remaining >= 30:
        # Generate points on edges
        edge_midpoints = []
        # Icosahedron has 30 edges
        ico_idx = [
            (0,1),(0,3),(0,4),(0,8),(1,2),(1,4),(1,6),
            (2,3),(2,5),(2,8),(3,5),(3,9),(4,6),(4,9),
            (4,10),(5,7),(5,10),(5,11),(6,7),(6,10),
            (7,11),(8,9),(8,11),(9,10),(10,11),(0,2),(1,5),(2,7),(3,6),(7,9)
        ]
        for i, j in ico_idx:
            mid = (ico_verts[i] + ico_verts[j]) / 2.0
            # Push out slightly to be on shell 2
            mid = mid * (spacing * np.sqrt(2.0) / np.linalg.norm(mid))
            edge_midpoints.append(mid)
        # Only take as many as needed
        take = min(30, remaining)
        positions.extend(edge_midpoints[:take])
        remaining -= take
    
    # Remaining atoms: place on outer Fibonacci sphere shells
    while remaining > 0:
        shell_radius = spacing * (2.0 + 0.5 * len([p for p in positions if np.linalg.norm(p) > spacing * 1.5]))
        n_on_shell = min(remaining, 20 + len(positions))
        shell_pts = _fib_sphere(n_on_shell, shell_radius, rng)
        positions.extend(shell_pts.tolist())
        remaining -= n_on_shell
    
    return np.array(positions[:n])


def _generate_fcc(n, spacing=0.72, seed=0):
    """Generate positions based on FCC-like packing."""
    rng = np.random.default_rng(seed)
    
    positions = []
    a = spacing  # lattice constant
    
    # Generate FCC lattice points within a sphere
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * a
    
    # Generate lattice points
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 2
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                center = np.array([ix, iy, iz]) * a
                for b in basis:
                    p = center + b
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    
    # Sort by distance and take closest n
    order = np.argsort(dists)
    positions = candidates[order[:n]]
    
    # Add small random perturbation
    positions = positions + rng.uniform(-0.02, 0.02, positions.shape)
    
    return positions


def _generate_hcp(n, spacing=0.72, seed=0):
    """Generate positions based on HCP-like packing."""
    rng = np.random.default_rng(seed)
    
    positions = []
    a = spacing
    
    # HCP: ABAB stacking
    # Layer A: hexagonal lattice
    # Layer B: shifted hexagonal lattice
    
    sqrt3 = np.sqrt(3.0)
    c = a * np.sqrt(8.0 / 3.0)  # ideal c/a ratio
    
    # Generate hexagonal lattice points
    max_layer = int(np.ceil(n ** (1.0/3.0))) + 2
    candidates = []
    
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        # Hexagonal lattice in xy plane
        for ix in range(-max_layer, max_layer + 1):
            for iy in range(-max_layer, max_layer + 1):
                # Hexagonal coordinates
                x = a * (ix + 0.5 * (iy % 2))
                y = a * sqrt3 / 2.0 * iy
                
                # Alternate layers offset
                if layer % 2 != 0:
                    x += a / 2.0
                    y += a * sqrt3 / 6.0
                
                p = np.array([x, y, z])
                candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    
    order = np.argsort(dists)
    positions = candidates[order[:n]]
    
    positions = positions + rng.uniform(-0.02, 0.02, positions.shape)
    
    return positions


def _generate_random_compact(n, spacing=0.72, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points], default=np.inf)
            if min_dist >= 0.75 * spacing:
                points.append(p)
        attempts += 1
    
    # If we didn't get enough, add more randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _relax_with_timeout(positions, potential, kind, deadline, margin=5.0):
    """Relax positions, but check deadline before starting."""
    if _time_remaining(deadline, margin) <= 0:
        return None, None
    try:
        energy, pos = relax(positions, potential, kind)
        if _time_remaining(deadline, margin) <= 0:
            return None, None
        return energy, pos
    except Exception:
        return None, None


def construct(instance):
    """Multi-start with diverse structural templates, relaxed locally."""
    start_time = time.time()
    total_budget = 300.0
    deadline = start_time + total_budget
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    # Spacing based on potential
    spacing_map = {"sutton_chen": 0.72, "morse": 1.0, "dzugutov": 1.1}
    spacing = spacing_map.get(potential, 0.72)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting structures
    generators = [
        (lambda s: _generate_icosahedral(n, spacing, seed=s), "ico"),
        (lambda s: _generate_fcc(n, spacing, seed=s), "fcc"),
        (lambda s: _generate_hcp(n, spacing, seed=s), "hcp"),
        (lambda s: _generate_random_compact(n, spacing, seed=s), "rand"),
    ]
    
    # Try multiple seeds for each generator
    seeds_per_generator = 3
    
    for gen_func, gen_name in generators:
        for seed in range(seeds_per_generator):
            if _time_remaining(deadline, 10.0) <= 0:
                break
            
            try:
                positions = gen_func(seed * 1000 + 42)
                if positions.shape != (n, 3):
                    continue
                
                # Check for invalid configurations
                if not np.all(np.isfinite(positions)):
                    continue
                
                # Check minimum distance
                min_dist = np.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(positions[i] - positions[j])
                        if d < min_dist:
                            min_dist = d
                        if d < 0.35:
                            break
                    if min_dist < 0.35:
                        break
                
                if min_dist < 0.35:
                    continue
                
                energy, relaxed = _relax_with_timeout(positions, potential, kind, deadline)
                
                if energy is not None and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
            
            except Exception:
                continue
        
        if _time_remaining(deadline, 10.0) <= 0:
            break
    
    # If we still have time, try more random starts
    if best_positions is None or _time_remaining(deadline, 10.0) > 0:
        extra_seeds = range(100, 150)
        for seed in extra_seeds:
            if _time_remaining(deadline, 10.0) <= 0:
                break
            try:
                positions = _generate_random_compact(n, spacing, seed=seed)
                if positions.shape != (n, 3):
                    continue
                if not np.all(np.isfinite(positions)):
                    continue
                
                min_dist = np.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(positions[i] - positions[j])
                        if d < min_dist:
                            min_dist = d
                        if d < 0.35:
                            break
                    if min_dist < 0.35:
                        break
                
                if min_dist < 0.35:
                    continue
                
                energy, relaxed = _relax_with_timeout(positions, potential, kind, deadline)
                
                if energy is not None and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
                    
            except Exception:
                continue
    
    # Fallback: if no valid result, return a simple configuration
    if best_positions is None:
        # Return a linear chain as fallback
        best_positions = np.zeros((n, 3))
        best_positions[:, 0] = np.linspace(-n * spacing / 2, n * spacing / 2, n)
    
    return best_positions
# EVOLVE-BLOCK-END
