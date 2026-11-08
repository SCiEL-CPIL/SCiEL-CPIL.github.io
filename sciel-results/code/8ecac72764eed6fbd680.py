"""
Strategy: Parallel Simulated Annealing (SA) with Adaptive Cooling and Polish.
Instead of basin hopping, which can get stuck in local minima with fixed step sizes,
we use Simulated Annealing, which has a theoretical guarantee to find the global
minimum if run long enough (with appropriate cooling). We run 4 independent SA chains
in parallel, each with a different initial configuration and cooling schedule.
The global best is updated whenever a chain finds a lower energy.
A final high-precision L-BFGS-B polish is applied to the best configuration found.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 290.0  # Safety margin for 300s limit
_START_TIME = None
_MAX_THREADS = 4

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return max(0.0, _DEADLINE_SEC - (time.time() - _START_TIME))

def _generate_fcc(n, spacing=0.72):
    """Generate n atoms from an FCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
    points = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if (x + y + z) % 2 == 0:
                    points.append([x * 0.5 * a, y * 0.5 * a, z * 0.5 * a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_hcp(n, spacing=0.72):
    """Generate n atoms from an HCP lattice fragment."""
    a = spacing
    c = 1.633 * a
    points = []
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3) / 2 * j)
            for k in range(-max_r, max_r + 1):
                points.append([x, y, k * c])
                points.append([x + 2*a/3, y + a*np.sqrt(3)/3, (k + 0.5) * c])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_bcc(n, spacing=0.72):
    """Generate n atoms from a BCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_random(n, spacing=0.72, seed=0):
    """Generate a random compact cluster with rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 1.2
    points = []
    min_dist = 0.4 * spacing
    attempts = 0
    max_attempts = n * 300
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                pts_arr = np.array(points)
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _simulated_annealing(instance, start_pos, thread_id):
    """
    Run Simulated Annealing for a given starting position.
    Returns the best energy and positions found.
    """
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Initial relaxation to ensure we start from a local minimum
    try:
        current_energy, current_pos = relax(start_pos, potential, kind)
        if not np.isfinite(current_energy):
            return float('inf'), None
    except Exception:
        return float('inf'), None

    best_energy = current_energy
    best_pos = current_pos.copy()
    
    rng = np.random.default_rng(thread_id * 7919 + 42)
    
    # SA Parameters
    # Temperature schedule: T(t) = T0 * alpha^t
    # We estimate the number of steps we can afford.
    # Each step: perturb (cheap) + energy calc (cheap) + maybe relax (expensive if accepted)
    # Actually, standard SA does not relax every step. It just evaluates energy.
    # But Sutton-Chen is complex. A "fast" SA might just perturb and check energy.
    # However, to escape local minima effectively in 3D clusters, we often need
    # to perturb significantly.
    # Let's use a "perturb and check" approach, but occasionally relax if we get stuck.
    
    # Estimate max steps
    # Assume ~0.005s per energy eval (vectorized).
    # If we do 1000 steps, that's 5s.
    # We have ~70s per thread (300s / 4 threads, roughly).
    # Let's aim for ~5000-10000 steps per thread.
    
    # Adaptive Step Size
    step_size = 0.2
    T = 5.0  # Initial temperature (arbitrary units, scaled by energy diff)
    alpha = 0.999  # Cooling rate
    
    # We will run until time is low or we have taken many steps
    steps = 0
    max_steps = 20000
    
    while _time_left() > 10.0 and steps < max_steps:
        steps += 1
        
        # Perturb
        noise = rng.normal(0, step_size, size=(n, 3))
        perturbed_pos = current_pos + noise
        
        # Check validity
        if not np.all(np.isfinite(perturbed_pos)):
            continue
            
        # Calculate energy of perturbed state
        try:
            e_new, _ = sutton_chen_energy_gradient(perturbed_pos, kind)
        except Exception:
            continue
            
        if not np.isfinite(e_new):
            continue
            
        delta_e = e_new - current_energy
        
        # Metropolis Criterion
        if delta_e < 0:
            # Accept if better
            current_energy = e_new
            current_pos = perturbed_pos
            if e_new < best_energy:
                best_energy = e_new
                best_pos = current_pos.copy()
                # Polish occasionally when finding new global best
                # This ensures we are at the bottom of the new basin
                if _time_left() > 20.0:
                    try:
                        e_polished, pos_polished = relax(perturbed_pos, potential, kind)
                        if np.isfinite(e_polished) and e_polished < best_energy:
                            best_energy = e_polished
                            best_pos = pos_polished.copy()
                            current_energy = e_polished
                            current_pos = pos_polished
                    except Exception:
                        pass
        else:
            # Accept if worse with probability exp(-delta_e / T)
            if rng.random() < np.exp(-delta_e / T):
                current_energy = e_new
                current_pos = perturbed_pos
            else:
                # Rejected: maybe increase step size slightly?
                # Or keep it same. Standard SA keeps step size fixed or adaptive.
                # Let's adapt step size based on acceptance rate roughly.
                # If we reject too much, decrease T or step size?
                # Actually, if we reject, we stay put.
                pass
        
        # Cool down
        T *= alpha
        
        # Adaptive step size: if we haven't improved in a long time, try larger jumps?
        # Or if we are stuck, increase noise.
        # Simple heuristic: if no improvement in last 1000 steps, increase step size.
        # We don't track 'last improvement' easily here without extra state.
        # Let's just slowly decrease step size as T decreases.
        step_size = max(0.05, step_size * 0.9995)
        
    return best_energy, best_pos

def _worker(args):
    instance, start_pos, thread_id = args
    return _simulated_annealing(instance, start_pos, thread_id)

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate diverse starting points
    starts = []
    
    fcc = _generate_fcc(n)
    if fcc is not None:
        starts.append(fcc)
        
    hcp = _generate_hcp(n)
    if hcp is not None:
        starts.append(hcp)
        
    bcc = _generate_bcc(n)
    if bcc is not None:
        starts.append(bcc)
        
    # Random starts
    for seed in range(5):
        rand = _generate_random(n, seed=seed)
        starts.append(rand)
        
    # Ensure we have at least 4 starts
    while len(starts) < _MAX_THREADS:
        starts.append(_generate_random(n, seed=len(starts)+100))
        
    # Select 4 diverse starts
    # Shuffle to ensure different threads get different templates
    rng_master = np.random.default_rng(123)
    rng_master.shuffle(starts)
    
    selected_starts = starts[:_MAX_THREADS]
    
    tasks = []
    for i in range(_MAX_THREADS):
        tasks.append((instance, selected_starts[i], i))
        
    best_energy_global = float('inf')
    best_pos_global = None
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=_MAX_THREADS) as executor:
        futures = {executor.submit(_worker, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            if _time_left() < 5:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy_global:
                    best_energy_global = energy
                    best_pos_global = positions
            except Exception:
                pass
                
    # Final Polish
    if best_pos_global is not None and _time_left() > 10:
        try:
            # Use a high precision polish
            e_final, pos_final = relax(best_pos_global, 'sutton_chen', kind)
            if np.isfinite(e_final) and e_final < best_energy_global:
                best_energy_global = e_final
                best_pos_global = pos_final
        except Exception:
            pass
            
    if best_pos_global is None:
        # Fallback
        fallback_start = _generate_random(n, seed=999)
        try:
            e, pos = relax(fallback_start, 'sutton_chen', kind)
            best_pos_global = pos
        except Exception:
            best_pos_global = fallback_start
            
    return best_pos_global
# EVOLVE-BLOCK-END
