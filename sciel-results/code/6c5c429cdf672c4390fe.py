"""Dzugutov cluster global minima search with physics-driven self-assembly."""
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


def _langevin_step(positions, velocities, dt, temperature, damping, rng):
    """
    Perform one Langevin dynamics step.
    positions: (n, 3) array
    velocities: (n, 3) array
    Returns updated positions, velocities, and energy
    """
    grad = _dzugutov_gradient(positions)
    # Langevin equation: m*dv/dt = -grad - gamma*v + sqrt(2*gamma*kT)*noise
    # With m=1: v_new = v + dt*(-grad - damping*v) + sqrt(2*damping*temperature)*dt*rng.normal
    noise = rng.normal(0, 1, positions.shape)
    velocities = velocities + dt * (-grad - damping * velocities) + np.sqrt(2 * damping * temperature) * dt * noise
    positions = positions + dt * velocities
    energy = _dzugutov_energy(positions)
    return positions, velocities, energy


def _self_assemble_cluster(n_atoms, rng, time_limit=15.0):
    """
    Physics-driven self-assembly: start with a small core and sequentially add atoms,
    allowing the system to relax via Langevin dynamics after each addition.
    This lets the structure naturally evolve into polytetrahedral order.
    """
    start_time = time.time()
    
    # Start with a small random cluster (3-5 atoms)
    initial_size = min(5, n_atoms)
    radius_init = 0.5
    positions = rng.uniform(-radius_init, radius_init, (initial_size, 3))
    velocities = np.zeros_like(positions)
    
    # Initial relaxation
    try:
        energy, positions = relax(positions, 'dzugutov', None)
    except Exception:
        pass
    
    # Parameters for Langevin dynamics
    dt = 0.005
    temperature = 0.3
    damping = 1.0
    
    # Sequentially add atoms
    while len(positions) < n_atoms:
        if time.time() - start_time > time_limit:
            break
        
        # Add one atom at a random position near the existing cluster
        current_radius = np.max(np.linalg.norm(positions, axis=1))
        add_radius = current_radius + 0.5
        new_atom = rng.uniform(-add_radius, add_radius, 3)
        
        # Ensure minimum distance from existing atoms
        for _ in range(20):
            dists = np.linalg.norm(positions - new_atom, axis=1)
            if np.min(dists) >= 0.5:
                break
            new_atom = rng.uniform(-add_radius, add_radius, 3)
        
        positions = np.vstack([positions, new_atom])
        velocities = np.vstack([velocities, np.zeros(3)])
        
        # Run a few Langevin steps to let the system relax
        n_steps = 20
        for _ in range(n_steps):
            if time.time() - start_time > time_limit:
                break
            positions, velocities, _ = _langevin_step(positions, velocities, dt, temperature, damping, rng)
        
        # Occasional full relaxation to remove accumulated errors
        if len(positions) % 5 == 0:
            try:
                energy, positions = relax(positions, 'dzugutov', None)
            except Exception:
                pass
    
    # Final relaxation
    try:
        energy, positions = relax(positions, 'dzugutov', None)
    except Exception:
        pass
    
    return positions, _dzugutov_energy(positions)


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
        if strategy == 'self_assemble':
            cand, energy = _self_assemble_cluster(size, rng, time_limit=12.0)
        elif strategy == 'random_compact':
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
            cand += rng.normal(0, 0.05, cand.shape)
            energy, cand = _relax_and_score(cand, potential, parameter)
        else:
            return None, None
        
        return energy, cand
    except Exception:
        return None, None


def _process_instance(size, potential, parameter, time_budget):
    """Process a single instance with physics-driven self-assembly and basin hopping."""
    rng = np.random.default_rng(size * 1000 + 7)
    
    # Define diverse strategies: primarily self-assembly, with some random compact as backup
    strategies = [
        ('self_assemble', size * 100 + 1),
        ('self_assemble', size * 100 + 2),
        ('self_assemble', size * 100 + 3),
        ('random_compact', size * 100 + 4),
        ('random_compact', size * 100 + 5),
    ]
    
    best_energy = np.inf
    best_positions = None
    
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
                if positions is not None and len(positions) == size and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # If we still have time, do basin hopping on the best
    if best_positions is not None and _hard_remaining() > 10.0:
        bh_time = min(_hard_remaining() - 5.0, 25.0)
        try:
            bh_pos, bh_energy = _basin_hopping(
                best_positions, 
                np.random.default_rng(size * 9000 + 5),
                max_iter=80,
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
        try:
            _, best_positions = relax(best_positions, 'dzugutov', None)
        except Exception:
            pass
    
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
