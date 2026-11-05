"""New program using structured seeds + diverse perturbations for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from concurrent.futures import ThreadPoolExecutor, as_completed

def _initialize_random_cluster(n, seed):
    """
    Generate a random cluster of n atoms with minimum distance 0.5.
    """
    rng = np.random.default_rng(seed)
    R = n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        v = rng.normal(size=3)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            continue
        v /= norm
        r = R * rng.random() ** (1.0/3.0)
        p = r * v
        if len(points) > 0:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) >= 0.5:
                points.append(p)
        else:
            points.append(p)
        attempts += 1
    
    if len(points) < n:
        for i in range(len(points), n):
            v = rng.normal(size=3)
            norm = np.linalg.norm(v)
            if norm < 1e-10:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v /= norm
            p = (R + 0.1 * (i - len(points))) * v
            points.append(p)
            
    return np.array(points)

def _md_anneal(pos, rho, seed, steps=1500, dt=0.05, T_start=1.0, T_end=0.01):
    """
    Perform MD with annealing.
    """
    rng = np.random.default_rng(seed)
    pos = pos.copy()
    vel = np.zeros_like(pos)
    m = 1.0
    gamma = 1.0
    
    best_energy = np.inf
    best_pos = pos.copy()
    
    for i in range(steps):
        T = T_start + (T_end - T_start) * (i / steps)
        
        try:
            energy, grad = morse_energy_gradient(pos, rho)
        except:
            break
            
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
        force = -grad
        
        vel += force / m * (dt / 2.0)
        pos += vel * dt
        
        try:
            energy_new, grad_new = morse_energy_gradient(pos, rho)
        except:
            break
        force_new = -grad_new
        
        vel += force_new / m * (dt / 2.0)
        
        vel *= np.exp(-gamma * dt)
        vel += np.sqrt(2.0 * T * gamma * dt / m) * rng.normal(size=pos.shape)
        
    return best_pos, best_energy

def _worker(args):
    n, rho, seed, potential, steps = args
    init_pos = _initialize_random_cluster(n, seed)
    try:
        final_pos, final_energy = _md_anneal(init_pos, rho, seed, steps=steps)
        try:
            energy, positions = relax(final_pos, potential, rho)
            return energy, positions
        except:
            return final_energy, final_pos
    except:
        return np.inf, None

def construct(instance):
    """
    Find low-energy Morse cluster configuration using MD with Simulated Annealing.
    """
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    n_restarts = 4
    steps_per_restart = 1500
    
    args_list = [(n, rho, 1000 + r, potential, steps_per_restart) for r in range(n_restarts)]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, args) for args in args_list]
        for future in as_completed(futures):
            try:
                energy, positions = future.result()
                if positions is not None and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except:
                pass
                
    if best_positions is None:
        best_positions = _initialize_random_cluster(n, 0)
        try:
            _, best_positions = relax(best_positions, potential, rho)
        except:
            pass
            
    return best_positions
# EVOLVE-BLOCK-END
