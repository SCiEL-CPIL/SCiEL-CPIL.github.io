"""New program using Basin Hopping for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax


def _random_compact_cluster(n, seed):
    """Generate a random compact cluster with minimum distance 0.5."""
    rng = np.random.default_rng(seed)
    # Estimate radius to fit n atoms with min dist 0.5
    # Volume ~ n * (0.5)^3 * packing_factor
    # R ~ (n * 0.125 * 0.64)^(1/3)
    # Let's just use a heuristic radius
    radius = 1.5 * n ** (1.0 / 3.0)
    
    points = []
    max_attempts = n * 200
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) > 0:
                # Check distance to existing points
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.5:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    
    # If we failed to place all, force place remaining (might be close, relax will fix)
    # But we must ensure > 0.5 for validity. 
    # If we can't place, we expand radius slightly and retry or just place far away?
    # Better: if stuck, place on a larger shell
    while len(points) < n:
        p = rng.uniform(-radius*1.5, radius*1.5, 3)
        if len(points) > 0:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) >= 0.5:
                points.append(p)
            else:
                # Try again
                continue
        else:
            points.append(p)
            
    return np.array(points)


def construct(instance):
    """Find low-energy Morse cluster configuration using Basin Hopping."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Initial configuration
    # Use a few different random starts to be safe, or just one good BH run?
    # Given 50s budget, one long BH run is better than many short ones.
    # Let's do a single Basin Hopping run with adaptive step size.
    
    # Initial seed
    seed = 42
    current_pos = _random_compact_cluster(n, seed)
    
    # Relax initial configuration
    try:
        current_energy, current_pos = relax(current_pos, potential, rho)
    except:
        # Fallback if initial relax fails
        current_pos = _random_compact_cluster(n, 123)
        current_energy = 0.0 # Will be updated
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # Basin Hopping parameters
    # Number of iterations
    # 300s / 6 instances = 50s per instance.
    # Relax is expensive. Let's aim for ~200-500 relaxations.
    # If relax takes ~5ms, 500 relaxations = 2.5s. We can afford more.
    # Let's try 1000 iterations if time permits, but we don't have a timer inside.
    # Let's estimate: n=80 is worst case. 
    # Let's use 500 iterations.
    n_iter = 500
    
    # Temperature schedule
    # Initial temp should be high enough to accept bad moves
    # E_pub is roughly -n * 0.5 * rho? No, well depth is 1. 
    # Total energy for n atoms is roughly -n * coordination / 2.
    # Let's set T0 based on a fraction of the expected energy scale.
    # For n=80, E ~ -100 to -200. 
    # Let's use T0 = 1.0 and cool down.
    
    T = 1.0
    T_min = 1e-4
    cooling = (T_min / T) ** (1.0 / n_iter)
    
    # Step size (kick magnitude)
    # Start with a moderate step size
    step_size = 0.5
    
    # Acceptance rate tracking for adaptive step size
    n_accept = 0
    n_try = 0
    window = 50
    
    for i in range(n_iter):
        # Generate new position by perturbing current
        # Gaussian noise
        noise = np.random.normal(0, step_size, (n, 3))
        new_pos = current_pos + noise
        
        # Relax new position
        try:
            new_energy, new_pos_relaxed = relax(new_pos, potential, rho)
        except:
            # If relax fails, reject
            new_energy = current_energy + 10.0
            new_pos_relaxed = current_pos
        
        # Metropolis acceptance
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            # Accept
            current_pos = new_pos_relaxed
            current_energy = new_energy
            n_accept += 1
        else:
            # Reject
            pass
            
        n_try += 1
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Adapt step size based on acceptance rate in last 'window'
        if n_try >= window:
            acc_rate = n_accept / n_try
            # Target acceptance rate ~ 0.5
            if acc_rate < 0.2:
                step_size *= 0.8
            elif acc_rate > 0.8:
                step_size *= 1.2
            n_accept = 0
            n_try = 0
            
        # Cool down
        T *= cooling
        
        # Optional: if T is very low, we can stop or just continue with small kicks
        if T < 1e-5:
            T = 1e-5
            
    return best_pos
# EVOLVE-BLOCK-END
