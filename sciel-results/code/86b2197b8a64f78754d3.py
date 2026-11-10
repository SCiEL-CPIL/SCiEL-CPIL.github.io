"""
Strategy: Parallel Simulated Annealing
- Exploits the 4-thread limit to run 4 independent SA chains in parallel.
- Uses a cooling schedule to balance exploration and exploitation.
- Leverages the provided relax() for local minimization after perturbations.
- Deadline-guarded to ensure valid output within 330s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # Safety margin below 330s

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_vertices():
    """Generate vertices of a regular icosahedron normalized to unit radius."""
    phi = (1 + 5**0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _generate_seed(n, rng, seed_type):
    """
    Generate a diverse seed based on type.
    Types: 'ico', 'fcc', 'random'
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    if seed_type == 'ico':
        # Icosahedral seed
        v = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])]
        pts.extend([v[i] for i in range(12)])
        # Add more if needed
        while len(pts) < n:
            # Add on surface of icosahedron
            # Pick a face, interpolate
            # Simple: random on sphere radius 1.5
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts.append(dir * (1.5 + 0.1*rng.random()))
        return np.array(pts[:n])
        
    elif seed_type == 'fcc':
        # FCC lattice seed
        a = 1.0
        fcc_points = []
        # Generate a large enough grid
        L = int(np.ceil(n**(1/3))) + 2
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    fcc_points.append(np.array([i*a, j*a, k*a]))
                    fcc_points.append(np.array([(i+0.5)*a, (j+0.5)*a, k*a]))
                    fcc_points.append(np.array([(i+0.5)*a, j*a, (k+0.5)*a]))
                    fcc_points.append(np.array([i*a, (j+0.5)*a, (k+0.5)*a]))
        
        centroid = np.mean(fcc_points, axis=0)
        fcc_points = [p - centroid for p in fcc_points]
        fcc_points.sort(key=lambda p: np.linalg.norm(p))
        
        pts = [fcc_points[i] for i in range(min(n, len(fcc_points)))]
        while len(pts) < n:
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts.append(dir * (2.0 + 0.2 * rng.random()))
        return np.array(pts[:n])
        
    else: # random
        # Random compact
        points = []
        radius = 1.5 * n ** (1.0 / 3.0)
        max_attempts = 2000
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < 0.5:
                    attempts += 1
                    continue
            points.append(p)
            attempts += 1
        
        if len(points) < n:
            for i in range(len(points), n):
                p = np.array([i * 1.0, 0, 0])
                points.append(p)
                
        return np.array(points)

def _simulated_annealing(positions, potential, kind, rng, max_iters, t_start, t_end, deadline_check):
    """
    Run Simulated Annealing.
    Returns best energy and positions found.
    """
    n = len(positions)
    current_energy, current_pos = relax(positions, potential, kind)
    best_energy, best_pos = current_energy, current_pos.copy()
    
    t = t_start
    
    for i in range(max_iters):
        if not deadline_check():
            break
            
        # Cooling
        # Linear or geometric? Geometric is common.
        # t = t_start * (t_end/t_start) ** (i/max_iters)
        # Let's use a simple linear cooling for stability in early stages?
        # No, geometric is better for SA.
        t = t_start * (t_end / t_start) ** (i / max_iters)
        
        # Perturbation
        # Step size should scale with temperature?
        # Usually step is fixed or slowly decreasing.
        # For cluster minimization, a step of ~0.2-0.5 is typical.
        step_size = 0.3 
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        try:
            new_energy, new_pos = relax(perturbed, potential, kind)
        except:
            continue
            
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / t):
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                
    return best_energy, best_pos

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    best_energy = float('inf')
    best_pos = None
    
    # Define 4 parallel chains with different seeds and RNG seeds
    chains = [
        {'seed_type': 'ico', 'rng_seed': 101},
        {'seed_type': 'fcc', 'rng_seed': 102},
        {'seed_type': 'random', 'rng_seed': 103},
        {'seed_type': 'random', 'rng_seed': 104},
    ]
    
    # We will run these in parallel using ThreadPoolExecutor
    # Each thread will run SA until deadline or max_iters
    # We need a shared state or just collect results at the end.
    # Since threads can't easily update a shared 'best' atomically without locks in a simple way,
    # and we just want the global best, we can run them to completion (or deadline) 
    # and then compare.
    
    # However, if one thread finds a very good solution early, we might want to stop others?
    # The deadline guard inside SA handles stopping.
    
    # Estimate max_iters:
    # Each relax call takes ~0.1s (guess).
    # 300s / 0.1s = 3000 iters per thread? 
    # But overhead of Python loops. Let's be conservative.
    # Let's say 500 iters per thread.
    # 4 threads * 500 iters = 2000 relax calls total.
    # If each takes 0.1s, that's 200s. Plus overhead. Seems safe.
    
    max_iters = 800
    t_start = 5.0
    t_end = 0.01
    
    results = []
    
    def run_chain(chain_cfg):
        rng = np.random.default_rng(chain_cfg['rng_seed'])
        seed_pos = _generate_seed(n, rng, chain_cfg['seed_type'])
        
        # Check if seed is valid
        if not np.all(np.isfinite(seed_pos)):
            return (float('inf'), None)
            
        # Validate min distance
        d = np.linalg.norm(seed_pos[:, None, :] - seed_pos[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 0.35:
             # Nudge apart
             for i in range(n):
                 for j in range(i+1, n):
                     if np.linalg.norm(seed_pos[i] - seed_pos[j]) < 0.35:
                         diff = seed_pos[i] - seed_pos[j]
                         diff /= np.linalg.norm(diff)
                         seed_pos[i] += diff * 0.1
                         seed_pos[j] -= diff * 0.1
             
        e, pos = _simulated_annealing(
            seed_pos, potential, kind, rng, 
            max_iters, t_start, t_end, 
            lambda: _time_left() > 5.0
        )
        return (e, pos)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_chain, cfg) for cfg in chains]
        
        for future in as_completed(futures):
            try:
                e, pos = future.result()
                if pos is not None and e < best_energy:
                    best_energy = e
                    best_pos = pos
            except Exception:
                pass

    # Fallback if no valid result
    if best_pos is None:
        # Simple linear chain
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    # Final validation
    if not np.all(np.isfinite(best_pos)):
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
        
    d = np.linalg.norm(best_pos[:, None, :] - best_pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.35:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    return best_pos
# EVOLVE-BLOCK-END
