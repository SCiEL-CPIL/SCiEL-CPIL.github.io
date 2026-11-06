"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _generate_dense_packing(n, rng):
    """Generate n points in a dense random packing."""
    points = []
    min_dist = 0.8
    max_attempts = 5000
    added = 0
    
    # Start with a central point
    points.append(np.array([0.0, 0.0, 0.0]))
    added += 1
    
    while added < n and max_attempts > 0:
        # Sample in a sphere of radius proportional to n^(1/3)
        radius = 1.5 * n ** (1.0/3.0)
        r = rng.uniform(0, radius)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        p = np.array([x, y, z])
        
        # Check distance to existing points
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < min_dist:
                max_attempts -= 1
                continue
        
        points.append(p)
        added += 1
        max_attempts -= 1
        
    # If we didn't get enough, force add random points (should be rare)
    while added < n:
        r = rng.uniform(0, 2.0)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points.append(np.array([x, y, z]))
        added += 1
        
    return np.array(points)

def _basin_hopping_improved(pos, potential, parameter, n_iter=100, kick_scale=0.5, rng=None, time_limit=48.0):
    """
    Improved Basin Hopping with Metropolis Acceptance and Adaptive Kick Size.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    start_time = time.time()
    
    # Initial relaxation
    try:
        energy, pos = relax(pos, potential, parameter)
    except Exception:
        return float('inf'), pos
        
    best_energy = energy
    best_pos = pos.copy()
    
    # Adaptive kick scale
    current_kick = kick_scale
    acceptance_count = 0
    window_size = 10
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Propose kick
        kick = rng.normal(0, current_kick, pos.shape)
        new_pos = pos + kick
        
        # Relax
        try:
            new_energy, new_pos_relaxed = relax(new_pos, potential, parameter)
        except Exception:
            continue
            
        # Metropolis Acceptance
        delta_e = new_energy - energy
        if delta_e < 0:
            # Accept
            pos = new_pos_relaxed
            energy = new_energy
            acceptance_count += 1
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        else:
            # Metropolis criterion
            # Temperature schedule: start high, cool down
            T = 0.5 * np.exp(-0.05 * i)
            if T > 1e-6:
                prob = np.exp(-delta_e / T)
                if rng.random() < prob:
                    pos = new_pos_relaxed
                    energy = new_energy
                    acceptance_count += 1
        
        # Adaptive Kick Size (every 10 steps)
        if (i + 1) % window_size == 0:
            accept_rate = acceptance_count / window_size
            if accept_rate > 0.7:
                current_kick *= 1.1 # Increase kick
            elif accept_rate < 0.3:
                current_kick *= 0.9 # Decrease kick
            acceptance_count = 0
            
        # Ensure kick doesn't get too small or too large
        current_kick = np.clip(current_kick, 0.1, 2.0)

    return best_energy, best_pos

def construct(instance):
    """
    Finds the lowest-energy structure for a Dzugutov cluster of size n.
    Uses Improved Basin Hopping with Adaptive Kick Size and Multiple Initializations.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 48.0 # 6 instances * 50s = 300s, leave buffer
    
    rng = np.random.default_rng(42)
    
    best_energy = float('inf')
    best_positions = None
    
    # Strategy 1: Dense Random Packing Initialization
    # Run multiple basin hopping instances from different dense packings
    
    n_runs = 15 # Increase number of runs
    
    for i in range(n_runs):
        if time.time() - start_time > time_limit:
            break
            
        # Generate dense random packing
        pos = _generate_dense_packing(size, rng)
        
        # Run Basin Hopping
        try:
            e, p = _basin_hopping_improved(pos, potential, parameter, n_iter=100, kick_scale=0.5, rng=rng, time_limit=time_limit)
            if e < best_energy:
                best_energy = e
                best_positions = p
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(size)
        radius = 2.0
        points = rng.uniform(-radius, radius, (size, 3))
        try:
            e, best_positions = relax(points, potential, parameter)
        except Exception:
            best_positions = points

    return best_positions

# EVOLVE-BLOCK-END
