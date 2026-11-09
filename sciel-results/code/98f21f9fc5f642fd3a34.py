"""Physics-driven self-assembly for Dzugutov cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _enforce_min_distance(positions, min_dist=0.5, max_iter=20):
    """Enforce minimum distance between all atom pairs."""
    pos = positions.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-10:
                    push = (min_dist - dist) / dist * d
                    pos[i] += 0.5 * push
                    pos[j] -= 0.5 * push
                    moved = True
        if not moved:
            break
    return pos


def _growing_cluster_init(n, seed, time_budget=5):
    """Initialize by growing a cluster atom by atom with relaxation."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Start with a small cluster and grow
    positions = np.array([[0.0, 0.0, 0.0]])
    
    for size in range(2, n + 1):
        if time.time() - start_time > time_budget - 1:
            break
        
        # Add new atom at random position near existing cluster
        if size <= 4:
            # Initial tetrahedron-like structure
            if size == 2:
                new_pos = np.array([1.8, 0.0, 0.0])
            elif size == 3:
                new_pos = np.array([0.9, 1.5, 0.0])
            else:
                new_pos = np.array([0.9, 0.5, 1.4])
        else:
            # Add atom near surface of existing cluster
            # Pick a random existing atom and place new one at ~1.8 distance
            idx = rng.integers(0, len(positions))
            direction = rng.normal(0, 1, size=3)
            direction /= np.linalg.norm(direction)
            new_pos = positions[idx] + 1.8 * direction
        
        positions = np.vstack([positions, new_pos])
        positions = _enforce_min_distance(positions)
        
        # Relax periodically
        if size % 3 == 0 or size == n:
            if time.time() - start_time > time_budget - 0.5:
                break
            try:
                _, positions = relax(positions, 'dzugutov', None)
            except:
                pass
    
    return positions


def _random_langevin_anneal(n, seed, time_budget=15, init_method='random'):
    """Single random start with simulated annealing via Langevin dynamics."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    if init_method == 'growing':
        positions = _growing_cluster_init(n, seed, time_budget=min(3, time_budget * 0.2))
    else:
        # Random initial positions in a sphere
        radius = 1.1 * n ** (1.0 / 3.0)
        positions = rng.uniform(-radius, radius, size=(n, 3))
        positions = _enforce_min_distance(positions)
    
    velocities = rng.normal(0, 0.1, size=(n, 3))
    
    # Adaptive step size and temperature schedule
    dt = 0.004
    damping = 1.5
    t0_temp = 0.6
    t1_temp = 0.003
    total_steps = 800
    
    best_energy = np.inf
    best_positions = positions.copy()
    
    # Get initial energy
    try:
        init_energy, _ = dzugutov_energy_gradient(positions)
        best_energy = init_energy
    except:
        pass
    
    for step in range(total_steps):
        elapsed = time.time() - start_time
        if elapsed > time_budget - 2:
            break
        
        progress = step / max(1, total_steps)
        # Slower exponential cooling for better exploration
        temp = t1_temp + (t0_temp - t1_temp) * np.exp(-3.0 * progress)
        
        # Compute energy and gradient using the provided function
        try:
            energy, grad = dzugutov_energy_gradient(positions)
        except:
            break
        forces = -grad
        
        # Langevin step
        sigma = np.sqrt(2 * temp * damping)
        noise = rng.normal(0, 1, size=positions.shape)
        new_vel = velocities + (forces - damping * velocities) * dt + sigma * np.sqrt(dt) * noise
        new_pos = positions + new_vel * dt
        new_pos = _enforce_min_distance(new_pos)
        
        positions = new_pos
        velocities = new_vel
        
        # Periodic relaxation with scipy - more frequent early on
        relax_interval = 50 if progress < 0.5 else 100
        if step % relax_interval == relax_interval - 1 or step == total_steps - 1:
            try:
                energy, pos = relax(positions, 'dzugutov', None)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos.copy()
                    positions = pos.copy()
                    velocities = np.zeros_like(positions)
            except:
                pass
    
    # Final relaxation
    try:
        energy, pos = relax(positions, 'dzugutov', None)
        if energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
    except:
        pass
    
    return best_positions, best_energy


def _polish(positions, seed, time_budget=5):
    """Polish with small perturbations and relaxation."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    try:
        energy, _ = dzugutov_energy_gradient(positions)
    except:
        return positions
    
    best_energy = energy
    best_pos = positions.copy()
    
    n_attempts = max(3, min(15, int(time_budget * 3)))
    
    for attempt in range(n_attempts):
        if time.time() - start_time > time_budget - 1:
            break
        
        # Decreasing perturbation size
        sigma = 0.08 * (1.0 - attempt / n_attempts) + 0.02
        pos = positions + rng.normal(0, sigma, size=positions.shape)
        pos = _enforce_min_distance(pos)
        
        try:
            energy, pos = relax(pos, 'dzugutov', None)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
                positions = pos.copy()
        except:
            pass
    
    return best_pos


def construct(instance):
    """Physics-driven self-assembly for Dzugutov cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    
    if potential != 'dzugutov':
        spacing = {"morse": 1.0, "sutton_chen": 0.72}.get(potential, 1.0)
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, instance.get("rho", instance.get("kind")))
        return positions
    
    # 300s total / 6 instances = 50s per instance, use 48s
    total_budget = 48
    t0 = time.time()
    
    # Use multiple strategies: growing cluster + random
    strategies = [
        ('growing', size * 7919 + 1),
        ('random', size * 7919 + 2),
        ('growing', size * 7919 + 3),
        ('random', size * 7919 + 4),
        ('growing', size * 7919 + 5),
    ]
    
    best_energy = np.inf
    best_pos = None
    
    # Allocate time dynamically
    per_start_budget = (total_budget - 8) / len(strategies)  # Reserve 8s for polishing
    
    for i, (method, seed) in enumerate(strategies):
        elapsed = time.time() - t0
        remaining = total_budget - 8 - elapsed  # Leave 8s for polish
        if remaining < 2:
            break
        
        budget = min(per_start_budget, remaining)
        pos, energy = _random_langevin_anneal(size, seed, time_budget=budget, init_method=method)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Final polish
    remaining = total_budget - (time.time() - t0)
    if best_pos is not None and remaining > 2:
        seed3 = size * 7919 + 6
        best_pos = _polish(best_pos, seed3, time_budget=min(7, remaining - 1))
    
    if best_pos is None:
        # Fallback: simple random + relax
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0)
        positions = rng.uniform(-radius, radius, size=(size, 3))
        positions = _enforce_min_distance(positions)
        try:
            energy, best_pos = relax(positions, 'dzugutov', None)
        except:
            best_pos = positions
    
    return best_pos
# EVOLVE-BLOCK-END
