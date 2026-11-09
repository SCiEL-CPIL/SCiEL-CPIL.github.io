"""Physics-driven self-assembly for Dzugutov cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global time tracking
_total_start_time = None
_TIME_BUDGET_PER_INSTANCE = 45  # seconds per instance, 6 instances * 45 = 270s < 300s


def _get_remaining_time():
    """Get remaining time budget for current instance."""
    global _total_start_time
    if _total_start_time is None:
        return _TIME_BUDGET_PER_INSTANCE
    return max(0, _TIME_BUDGET_PER_INSTANCE - (time.time() - _total_start_time))


def _fast_dzugutov_energy_and_gradient(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute total energy and gradient for Dzugutov potential - vectorized."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    # Only compute for pairs within cutoff b
    mask = dist < b
    mask = np.triu(mask, k=1)
    
    if not np.any(mask):
        return 0.0, grad
    
    # Get indices
    i_idx, j_idx = np.where(mask)
    r = dist[i_idx, j_idx]
    
    # Compute energy and dv/dr for each pair
    v = np.zeros(len(r))
    dv_dr = np.zeros(len(r))
    
    # Region 1: r < a
    mask_a = r < a
    if np.any(mask_a):
        r_a = r[mask_a]
        exp_a = np.exp(c / (r_a - a))
        r_inv_m = r_a ** (-m)
        v[mask_a] = A * (r_inv_m - B) * exp_a
        dv_dr[mask_a] = A * (-m * r_a ** (-m - 1)) * exp_a + A * (r_inv_m - B) * exp_a * (-c / (r_a - a) ** 2)
    
    # Region 2: a <= r < b
    mask_b = (r >= a) & (r < b)
    if np.any(mask_b):
        r_b = r[mask_b]
        exp_b = np.exp(d / (r_b - b))
        v[mask_b] = B * exp_b
        dv_dr[mask_b] = B * exp_b * (-d / (r_b - b) ** 2)
    
    # Sum energy
    energy = np.sum(v)
    
    # Compute gradient
    for k in range(len(r)):
        i, j = i_idx[k], j_idx[k]
        r_k = r[k]
        if r_k > 1e-10:
            r_vec = positions[i] - positions[j]
            f = -dv_dr[k] / r_k * r_vec
            grad[i] += f
            grad[j] -= f
    
    return energy, grad


def _langevin_step_fast(positions, velocities, dt, temperature, damping, rng):
    """One Langevin dynamics step with vectorized operations."""
    n = len(positions)
    energy, grad = _fast_dzugutov_energy_and_gradient(positions)
    
    forces = -grad
    sigma = np.sqrt(2 * temperature * damping)
    noise = rng.normal(0, 1, size=positions.shape)
    
    new_velocities = velocities + (forces - damping * velocities) * dt + sigma * np.sqrt(dt) * noise
    new_positions = positions + new_velocities * dt
    
    # Enforce minimum distance constraint
    diff = new_positions[:, np.newaxis, :] - new_positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    mask = dist < 0.5
    i_idx, j_idx = np.where(np.triu(mask, k=1))
    
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        d = new_positions[i] - new_positions[j]
        dist_k = np.linalg.norm(d)
        if dist_k > 1e-10:
            push = (0.5 - dist_k) / dist_k * d
            new_positions[i] += 0.5 * push
            new_positions[j] -= 0.5 * push
    
    return new_positions, new_velocities


def _place_atom_near_cluster(positions, rng, min_dist=0.55):
    """Place a new atom near the existing cluster, preferring positions with many neighbors."""
    n = len(positions)
    if n == 0:
        return np.zeros(3)
    
    # Try to place near existing atoms
    for attempt in range(50):
        # Pick a random existing atom
        i = rng.integers(0, n)
        # Place at random distance and direction from that atom
        dist = rng.uniform(1.0, 1.5)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(-np.pi / 2, np.pi / 2)
        new_atom = positions[i] + dist * np.array([
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi)
        ])
        
        # Check minimum distance
        if all(np.linalg.norm(new_atom - p) >= min_dist for p in positions):
            return new_atom
    
    # Fallback: place far away
    center = np.mean(positions, axis=0)
    radius = 1.5 * n ** (1.0 / 3.0)
    while True:
        new_atom = center + rng.uniform(-radius, radius, 3)
        if all(np.linalg.norm(new_atom - p) >= min_dist for p in positions):
            return new_atom


def _self_assemble_cluster(n, seed, time_budget=None):
    """Build cluster by sequential addition with Langevin relaxation."""
    if time_budget is None:
        time_budget = _get_remaining_time()
    
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Start with 2 atoms at optimal distance
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    velocities = np.zeros((2, 3))
    
    # Adaptive parameters
    base_steps_per_atom = 80
    temperature_start = 0.8
    temperature_end = 0.02
    damping = 0.9
    dt = 0.008
    
    for target_n in range(3, n + 1):
        # Check time
        if time.time() - start_time > time_budget * 0.7:
            break
        
        # Place new atom near cluster
        new_atom = _place_atom_near_cluster(positions, rng)
        positions = np.vstack([positions, new_atom])
        velocities = np.vstack([velocities, np.zeros(3)])
        
        # Relax with Langevin dynamics
        progress = (target_n - 2) / max(1, n - 2)
        temp = temperature_start * (1 - progress) ** 1.5 + temperature_end * progress
        
        # Scale steps based on cluster size and remaining time
        steps_this = max(20, int(base_steps_per_atom * (1 - 0.3 * progress)))
        
        for step in range(steps_this):
            if time.time() - start_time > time_budget * 0.8:
                break
            positions, velocities = _langevin_step_fast(positions, velocities, dt, temp, damping, rng)
        
        # Local relaxation after each addition
        if target_n % 3 == 0 or target_n == n:
            energy, pos = relax(positions, 'dzugutov', None)
            positions = pos
            velocities = np.zeros_like(positions)
    
    return positions


def _multi_start_langevin(n, seed, n_starts, time_budget):
    """Multi-start Langevin dynamics for full cluster."""
    start_time = time.time()
    rng = np.random.default_rng(seed)
    best_energy = np.inf
    best_positions = None
    
    steps_per_start = max(100, int(time_budget * 80 / max(1, n_starts)))
    
    for start in range(n_starts):
        if time.time() - start_time > time_budget * 0.85:
            break
        
        # Random compact initial configuration
        radius = 1.1 * n ** (1.0 / 3.0) * 0.65
        positions = rng.uniform(-radius, radius, size=(n, 3))
        
        # Enforce minimum distance with more iterations
        for _ in range(30):
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            mask = dist < 0.5
            i_idx, j_idx = np.where(np.triu(mask, k=1))
            
            if len(i_idx) == 0:
                break
            
            for k in range(len(i_idx)):
                i, j = i_idx[k], j_idx[k]
                d = positions[i] - positions[j]
                dist_k = np.linalg.norm(d)
                if dist_k > 1e-10:
                    push = (0.5 - dist_k) / dist_k * d
                    positions[i] += 0.5 * push
                    positions[j] -= 0.5 * push
        
        velocities = rng.normal(0, 0.08, size=(n, 3))
        
        # Langevin dynamics with cooling
        temperature_start = 0.5
        temperature_end = 0.01
        damping = 0.85
        dt = 0.006
        
        for step in range(steps_per_start):
            if time.time() - start_time > time_budget * 0.9:
                break
            
            progress = step / steps_per_start
            temp = temperature_start * (1 - progress) ** 2 + temperature_end * progress
            
            positions, velocities = _langevin_step_fast(positions, velocities, dt, temp, damping, rng)
        
        # Final local relaxation
        energy, pos = relax(positions, 'dzugutov', None)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    return best_positions


def _polish_positions(positions, seed, time_budget):
    """Polish positions with small perturbations and local relaxation."""
    start_time = time.time()
    best_energy, best_pos = relax(positions, 'dzugutov', None)
    rng = np.random.default_rng(seed)
    
    # Try multiple perturbation scales
    for scale in [0.03, 0.05, 0.08, 0.12]:
        if time.time() - start_time > time_budget * 0.7:
            break
        
        for _ in range(3):
            if time.time() - start_time > time_budget * 0.85:
                break
            
            pos = best_pos + rng.normal(0, scale, size=best_pos.shape)
            
            # Enforce min distance
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            mask = dist < 0.5
            i_idx, j_idx = np.where(np.triu(mask, k=1))
            for k in range(len(i_idx)):
                i, j = i_idx[k], j_idx[k]
                d = pos[i] - pos[j]
                dist_k = np.linalg.norm(d)
                if dist_k > 1e-10:
                    push = (0.5 - dist_k) / dist_k * d
                    pos[i] += 0.5 * push
                    pos[j] -= 0.5 * push
            
            energy, pos = relax(pos, 'dzugutov', None)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
    
    return best_pos


def _combined_search(n, seed, time_budget):
    """Combine self-assembly and multi-start Langevin with time management."""
    start_time = time.time()
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Self-assembly (fast, good for moderate sizes)
    sa_time = min(time_budget * 0.35, 15)
    if time_budget > 10:
        pos = _self_assemble_cluster(n, seed, time_budget=sa_time)
        if pos is not None and len(pos) == n:
            energy, pos = relax(pos, 'dzugutov', None)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
    
    # Strategy 2: Multi-start Langevin
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 8:
        n_starts = max(2, int(remaining_time / 12))
        pos = _multi_start_langevin(n, seed + 1000, n_starts=n_starts, 
                                     time_budget=remaining_time * 0.6)
        if pos is not None:
            energy, pos = relax(pos, 'dzugutov', None)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
    
    # Final polish
    if best_positions is not None:
        polish_time = min(5, time_budget - (time.time() - start_time))
        if polish_time > 1:
            best_positions = _polish_positions(best_positions, seed + 9999, polish_time)
    
    return best_positions


def construct(instance):
    """Physics-driven self-assembly for Dzugutov cluster global minima."""
    global _total_start_time
    if _total_start_time is None:
        _total_start_time = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    if potential != 'dzugutov':
        # Fallback for other potentials
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
    
    # For Dzugutov, use physics-driven self-assembly
    seed = size * 7919  # Deterministic seed per cluster size
    positions = _combined_search(size, seed, time_budget=_TIME_BUDGET_PER_INSTANCE)
    
    if positions is None or len(positions) != size:
        # Fallback: simple random placement with relaxation
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.7
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), 'dzugutov', None)
    
    return positions
# EVOLVE-BLOCK-END
