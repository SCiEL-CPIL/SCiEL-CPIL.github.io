"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_cluster(n, a=1.44):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites by distance from center."""
    if n <= 0:
        return np.empty((0, 3))
    # fcc basis vectors
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    # Generate candidate positions
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 3
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    candidates.append(p)
    candidates = np.array(candidates)
    # Sort by distance from origin
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    selected = candidates[order[:n]]
    # Center the cluster
    center = selected.mean(axis=0)
    selected = selected - center
    return selected


def _hcp_cluster(n, a=1.44):
    """Generate an n-atom hcp-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    # hcp: two close-packed layers with AB stacking
    # Generate hexagonal lattice points
    candidates = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 3
    for layer in range(-max_shell, max_shell + 1):
        z = layer * a * np.sqrt(2.0 / 3.0)
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * j)
                y = a * np.sqrt(3.0) / 2.0 * j
                if layer % 2 == 0:
                    candidates.append(np.array([x, y, z]))
                else:
                    candidates.append(np.array([x + a * 0.5, y + a * np.sqrt(3.0) / 6.0, z]))
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    selected = candidates[order[:n]]
    center = selected.mean(axis=0)
    selected = selected - center
    return selected


def _random_cluster(n, spacing=1.44, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 100000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we didn't get enough points, add random ones
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def construct(instance):
    """Search for low-energy Sutton-Chen cluster configurations using multiple starting geometries and local relaxation."""
    start_time = time.time()
    deadline = 280.0  # Safety margin before 300s limit
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Generate diverse starting configurations
    rng = np.random.default_rng(n)
    start_configs = []
    
    # FCC-like
    start_configs.append(_fcc_cluster(n))
    
    # HCP-like
    start_configs.append(_hcp_cluster(n))
    
    # Several random configurations
    for _ in range(6):
        start_configs.append(_random_cluster(n, rng=rng))
    
    best_energy = np.inf
    best_positions = None
    
    for i, config in enumerate(start_configs):
        if time.time() - start_time > deadline:
            break
        
        try:
            energy, positions = relax(config, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If we have time, try additional random restarts
    while time.time() - start_time < deadline - 30:
        config = _random_cluster(n, rng=rng)
        try:
            energy, positions = relax(config, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: return fcc cluster
        best_positions = _fcc_cluster(n)
    
    return best_positions
# EVOLVE-BLOCK-END
