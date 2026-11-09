"""Dzugutov cluster global minima search with self-assembly and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import math

# Constants
_TOTAL_BUDGET = 310.0  # 330s limit with 20s safety margin
_HARD_LIMIT = 325.0
_START_TIME = None
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_RESULTS = {}

A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16

# High-quality 13-atom icosahedron core (approximate relaxed coordinates)
# Standard icosahedron vertices scaled to optimal Dzugutov distance (~1.1-1.2)
_PHI = (1 + np.sqrt(5)) / 2
_ICO_VERTS = np.array([
    [0, 1, _PHI], [0, 1, -_PHI], [0, -1, _PHI], [0, -1, -_PHI],
    [1, _PHI, 0], [1, -_PHI, 0], [-1, _PHI, 0], [-1, -_PHI, 0],
    [_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, 1], [-_PHI, 0, -1]
])
_ICO_EDGE = 2 * _PHI
_ICO_VERTS = _ICO_VERTS / _ICO_EDGE * 1.15  # Scale to typical Dzugutov bond length
_CORE_13 = np.vstack([[0.0, 0.0, 0.0], _ICO_VERTS])


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return _TOTAL_BUDGET - _elapsed()


def _hard_remaining():
    return _HARD_LIMIT - _elapsed()


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        diffs = positions[i+1:] - positions[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        mask_a = dists < a_DZ
        mask_b = (dists >= a_DZ) & (dists < b_DZ)
        
        if np.any(mask_a):
            da = dists[mask_a]
            da = np.maximum(da, 1e-10)
            v_a = A_DZ * (da ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (da - a_DZ))
            total += np.sum(v_a)
        
        if np.any(mask_b):
            db = dists[mask_b]
            db = np.maximum(db, 1e-10)
            v_b = B_DZ * np.exp(d_DZ / (db - b_DZ))
            total += np.sum(v_b)
    
    return total


def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy."""
    try:
        return dzugutov_energy_gradient(positions)
    except Exception:
        # Finite difference fallback
        n = len(positions)
        grad = np.zeros_like(positions)
        eps = 1e-6
        E0 = _dzugutov_energy(positions)
        for i in range(n):
            for j in range(3):
                pos_p = positions.copy()
                pos_p[i, j] += eps
                E_p = _dzugutov_energy(pos_p)
                grad[i, j] = (E_p - E0) / eps
        return grad


def _fibonacci_sphere(n, radius, rng):
    """Generate n points on a sphere using Fibonacci spiral."""
    if n == 0:
        return np.empty((0, 3))
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        y = 1 - 2 * i / (n - 1) if n > 1 else 0
        r = np.sqrt(1 - y**2)
        theta = golden_angle * i + rng.uniform(0, 2 * np.pi)
        points[i] = np.array([np.cos(theta) * r, y, np.sin(theta) * r]) * radius
    return points

def _generate_core_plus_shell(n_atoms, rng, strategy='surface'):
    """
    Generate a cluster by starting with the 13-atom core and adding atoms.
    strategy: 'surface' (on a sphere), 'interstitial' (near bonds), 'random'
    """
    if n_atoms <= 13:
        return _CORE_13[:n_atoms].copy()
    
    remaining = n_atoms - 13
    new_points = []
    
    if strategy == 'surface':
        # Place remaining atoms on a sphere surrounding the core
        # Radius slightly larger than core radius (~1.15)
        radius = 1.15 * 1.6 
        fib_points = _fibonacci_sphere(remaining, radius, rng)
        new_points = fib_points
    elif strategy == 'interstitial':
        # Place atoms near the edges of the icosahedron (interstitial sites)
        # There are 30 edges in an icosahedron.
        # We will randomly select edges and place atoms near their midpoints, pushed out.
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                # Check if i and j are connected (distance is edge length)
                # In icosahedron, each vertex connects to 5 others.
                # Simplified: just use random pairs for now, or precomputed.
                # For efficiency, let's just use random points in the shell but biased towards high density regions.
                pass
        
        # Fallback for interstitial: just use a tighter shell
        radius = 1.15 * 1.3
        fib_points = _fibonacci_sphere(remaining, radius, rng)
        new_points = fib_points
    else:
        # Random compact
        radius = 1.15 * (n_atoms ** (1/3.0)) * 0.8
        for _ in range(remaining):
            p = rng.uniform(-radius, radius, 3)
            new_points.append(p)
        new_points = np.array(new_points)
    
    core = _CORE_13.copy()
    return np.vstack([core, new_points])


def _basin_hopping(positions, rng, max_iter=50, step_size=0.3, time_limit=10.0):
    """
    Simple basin hopping: perturb, relax, accept if lower energy.
    """
    start_time = time.time()
    current = positions.copy()
    current_energy = _dzugutov_energy(current)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb
        perturbed = current + rng.normal(0, step_size, current.shape)
        
        # Relax
        try:
            e_new, relaxed = relax(perturbed, 'dzugutov', None)
        except Exception:
            continue
        
        # Accept if lower (or with small probability for exploration)
        if e_new < current_energy or rng.random() < 0.1:
            current = relaxed
            current_energy = e_new
        
        # Reduce step size slowly
        step_size *= 0.98
    
    return current, current_energy


def _relax_and_score(positions, potential, parameter):
    """Relax positions and return energy, positions."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions

def _worker_generate_and_relax(size, seed, strategy, potential, parameter):
    """Worker function: generate a specific candidate and relax it."""
    rng = np.random.default_rng(seed)
    try:
        if strategy == 'surface':
            cand = _generate_core_plus_shell(size, rng, 'surface')
        elif strategy == 'interstitial':
            cand = _generate_core_plus_shell(size, rng, 'interstitial')
        elif strategy == 'random':
            # Generate a compact random cluster
            radius = 1.15 * size ** (1/3.0) * 0.8
            points = [np.array([0.0, 0.0, 0.0])]
            min_dist = 0.6
            attempts = 0
            while len(points) < size and attempts < 2000:
                attempts += 1
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius: continue
                if all(np.linalg.norm(p-q) >= min_dist for q in points):
                    points.append(p)
            while len(points) < size:
                points.append(rng.uniform(-radius, radius, 3))
            cand = np.array(points)
        else:
            return None, None
            
        # Perturb slightly to break symmetry
        cand += rng.normal(0, 0.05, cand.shape)
        
        energy, relaxed = _relax_and_score(cand, potential, parameter)
        return energy, relaxed
    except Exception:
        return None, None


def _process_instance(size, potential, parameter, time_budget):
    """Process a single instance with multiple strategies in parallel."""
    rng = np.random.default_rng(size * 1000 + 7)
    
    # Define diverse strategies
    strategies = [
        ('surface', size * 100 + 1),
        ('surface', size * 100 + 2),
        ('interstitial', size * 100 + 3),
        ('random', size * 100 + 4),
        ('random', size * 100 + 5),
        ('surface', size * 100 + 6),
    ]
    
    best_energy = np.inf
    best_positions = None
    results = []
    
    # Parallelize the generation and relaxation
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for strat, seed in strategies:
            if _hard_remaining() < 5.0:
                break
            futures.append(executor.submit(_worker_generate_and_relax, size, seed, strat, potential, parameter))
        
        for future in as_completed(futures):
            if _hard_remaining() < 5.0:
                break
            try:
                energy, positions = future.result()
                if positions is not None and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # If we still have time, do basin hopping on the best
    if best_positions is not None and _hard_remaining() > 10.0:
        bh_time = min(_hard_remaining() - 5.0, 20.0)
        try:
            bh_pos, bh_energy = _basin_hopping(
                best_positions, 
                np.random.default_rng(size * 9000 + 5),
                max_iter=50,
                step_size=0.3,
                time_limit=bh_time
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except Exception:
            pass
    
    # Final safety: ensure valid output
    if best_positions is None or len(best_positions) != size:
        radius = 1.1 * size ** (1.0 / 3.0)
        best_positions = rng.uniform(-radius, radius, (size, 3))
    
    return best_positions


def construct(instance):
    """Find low-energy Dzugutov cluster using core-based seeding and basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    remaining_instances = [s for s in _INSTANCE_ORDER if s >= size]
    time_per_instance = _remaining() / max(1, len(remaining_instances))
    time_per_instance = min(time_per_instance, 45.0)
    
    # Check if we already have a result
    if size in _INSTANCE_RESULTS:
        return _INSTANCE_RESULTS[size]
    
    # Process this instance
    result = _process_instance(size, potential, parameter, time_per_instance)
    
    # Cache the result
    _INSTANCE_RESULTS[size] = result
    
    return result
# EVOLVE-BLOCK-END
