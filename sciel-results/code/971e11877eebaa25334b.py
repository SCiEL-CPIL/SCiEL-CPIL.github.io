import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def _run_bh_restart(n, kind, rng_seed, max_bh_steps=150, perturbation_scale=0.3):
    """
    Runs a single Basin Hopping restart.
    """
    rng = np.random.default_rng(rng_seed)
    
    # 1. Initialization
    # Estimate radius
    radius = 1.5 * n ** (1.0 / 3.0)
    
    points = []
    attempts = 0
    # Try to pack points with min dist 0.4
    while len(points) < n and attempts < 2000:
        attempts += 1
        # Sample from sphere
        vec = rng.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm == 0: continue
        r = radius * rng.random() ** (1.0/3.0)
        p = vec / norm * r
        
        if len(points) == 0:
            points.append(p)
        else:
            # Check distance
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) > 0.4:
                points.append(p)
                
    if len(points) < n:
        return np.inf, None
        
    points = np.array(points)
    
    # 2. Initial Relaxation
    try:
        current_energy, current_points = relax(points, 'sutton_chen', kind)
    except Exception:
        return np.inf, None
        
    if not np.isfinite(current_energy):
        return np.inf, None
        
    # Center the cluster
    current_points = current_points - np.mean(current_points, axis=0)
    
    best_energy = current_energy
    best_points = current_points.copy()
    
    # 3. Basin Hopping Loop
    T = 0.5
    cooling_rate = 0.95
    
    for step in range(max_bh_steps):
        # Perturb
        # Scale perturbation slightly with T to explore more at high T
        scale = perturbation_scale * (T / 0.5)
        perturbation = rng.normal(scale=scale, size=(n, 3))
        perturbed_points = current_points + perturbation
        
        # Relax
        try:
            new_energy, new_points = relax(perturbed_points, 'sutton_chen', kind)
        except Exception:
            continue
            
        if not np.isfinite(new_energy):
            continue
            
        # Center
        new_points = new_points - np.mean(new_points, axis=0)
        
        # Accept/Reject
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_energy = new_energy
            current_points = new_points
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points.copy()
                
        T *= cooling_rate
        if T < 0.05:
            break
            
    return best_energy, best_points

def construct(instance):
    n = instance['n']
    kind = instance['kind']
    
    start_time = time.time()
    time_limit = 45.0 # Seconds
    
    # Number of restarts
    # 8 restarts with 4 threads = 2 batches.
    # This should fit within time limit for all N.
    n_restarts = 8
    
    seeds = [instance['n'] * 1000 + i * 12345 for i in range(n_restarts)]
    
    best_energy = np.inf
    best_positions = None
    
    # Use ThreadPoolExecutor
    # max_workers=4 as per constraint
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for seed in seeds:
            futures.append(executor.submit(_run_bh_restart, n, kind, seed))
            
        for future in as_completed(futures):
            try:
                e, p = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = p
            except Exception:
                pass
                
    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(n)
        points = rng.normal(size=(n, 3)) * 0.5
        return points
        
    return best_positions
