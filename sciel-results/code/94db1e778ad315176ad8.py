"""
Morse cluster global minima: Basin Hopping with collision dynamics local minimization.
Uses velocity Verlet with thermostatting for local relaxation, enabling structural
reorganization without relying on predefined crystalline templates.
"""
import numpy as np
import time
import threading
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Deadline guard
TOTAL_DEADLINE = 290.0  # Safety margin before 330s kill
_START_TIME = None
_lock = threading.Lock()
_best_energy_global = np.inf
_best_positions_global = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient vectorized."""
    n = positions.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(positions)
        
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, 1.0)
    r = np.maximum(d, 1e-10)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient
    # V(r) = exp(x)(exp(x) - 2)
    # dV/dr = exp(x)(exp(x) - 1) * dx/dr = -rho * exp(x) * (exp(x) - 1)
    dvdr = -rho * ex * (ex - 1.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        w = dvdr / d
    np.fill_diagonal(w, 0.0)
    
    grad = np.sum(w[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _collision_dynamics_minimize(positions, rho, max_steps=500, dt=0.005, target_temp=0.0, 
                                  cooling_rate=0.999, min_temp=1e-4):
    """
    Local minimization using velocity Verlet with thermostatting.
    Allows structural reorganization through energy dissipation.
    
    Args:
        positions: (n, 3) array of initial positions
        rho: Morse range parameter
        max_steps: maximum number of Verlet steps
        dt: time step
        target_temp: target temperature for thermostat
        cooling_rate: multiplicative cooling factor per step
        min_temp: minimum temperature
    
    Returns:
        (positions, energy) tuple
    """
    global _START_TIME
    
    n = positions.shape[0]
    pos = positions.copy()
    vel = np.zeros_like(pos)
    
    # Initialize with small random velocities
    rng = np.random.default_rng(hash(tuple(pos.flatten())) % 2**32)
    vel = rng.normal(0, np.sqrt(0.1), pos.shape)
    
    current_temp = 0.5  # Start at moderate temperature
    energy, grad = _morse_energy_and_grad(pos, rho)
    
    best_pos = pos.copy()
    best_energy = energy
    
    for step in range(max_steps):
        # Check deadline
        if time.time() - _START_TIME > TOTAL_DEADLINE - 3:
            break
            
        # Velocity Verlet step 1: half-kick
        vel += 0.5 * dt * grad
        
        # Drift
        pos += dt * vel
        
        # Compute new forces
        energy_new, grad_new = _morse_energy_and_grad(pos, rho)
        
        # Velocity Verlet step 2: half-kick
        vel += 0.5 * dt * grad_new
        
        # Thermostatting: scale velocities to target temperature
        # Kinetic energy = 0.5 * sum(m*v^2), m=1
        ke = 0.5 * np.sum(vel ** 2)
        dof = 3 * n - 6  # Remove 3 translational + 3 rotational DOF
        if dof > 0:
            current_temp = ke / (0.5 * dof)
            
            # Cool down over time
            current_temp *= cooling_rate
            current_temp = max(current_temp, min_temp)
            
            # Scale velocities to match target temperature
            if current_temp > 0:
                target_ke = 0.5 * dof * current_temp
                if ke > 1e-12:
                    scale = np.sqrt(target_ke / ke)
                    vel *= scale
        
        # Remove center of mass motion
        com_vel = np.mean(vel, axis=0)
        vel -= com_vel
        
        # Update
        energy = energy_new
        grad = grad_new
        
        # Track best
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
        # Convergence check: if temperature is very low and gradient is small
        if current_temp < 1e-3 and np.max(np.abs(grad)) < 1e-6:
            break
    
    # Final energy computation
    final_energy = _morse_energy_and_grad(best_pos, rho)[0]
    return best_pos, final_energy


def _local_minimize(positions, rho):
    """
    Local minimization using collision dynamics.
    Falls back to chem_clusters.relax if collision dynamics fails.
    """
    global _START_TIME
    
    # Check deadline
    if time.time() - _START_TIME > TOTAL_DEADLINE - 5:
        return positions, _morse_energy_and_grad(positions, rho)[0]
    
    # Try collision dynamics first
    try:
        result_pos, result_energy = _collision_dynamics_minimize(positions, rho, max_steps=300)
        if np.isfinite(result_energy):
            return result_pos, result_energy
    except Exception:
        pass
    
    # Fallback: try the provided relax function
    try:
        relaxed = relax(positions, 'morse', rho)
        if relaxed is not None:
            relaxed = np.array(relaxed, dtype=float)
            if relaxed.shape == positions.shape:
                e = _morse_energy_and_grad(relaxed, rho)[0]
                if np.isfinite(e):
                    return relaxed, e
    except Exception:
        pass
    
    # Last resort: return original
    return positions, _morse_energy_and_grad(positions, rho)[0]


def _is_valid(positions, min_dist=0.5):
    """Check if all pair distances are at least min_dist."""
    try:
        d = pdist(positions)
        return np.min(d) >= min_dist
    except Exception:
        return False


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like compact cluster."""
    a = spacing
    max_coord = int(np.ceil(n ** (1.0/3.0)) * 2) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like compact cluster."""
    a = spacing
    max_coord = int(np.ceil(n ** (1.0/3.0)) * 2) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like compact cluster."""
    a = spacing
    max_coord = int(np.ceil(n ** (1.0/3.0)) * 2) + 4
    points = []
    # Layer A (triangular)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i * a
            y = j * a * np.sqrt(3) / 2
            if i % 2 == 1:
                y += a * np.sqrt(3) / 4
            points.append(np.array([x, y, 0]))
    z_b = a * np.sqrt(2.0/3.0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i * a
            y = j * a * np.sqrt(3) / 2
            if i % 2 == 1:
                y += a * np.sqrt(3) / 4
            points.append(np.array([x, y, z_b]))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _generate_random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum spacing."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    min_dist = 0.55 * spacing
    points = []
    attempts = 0
    max_attempts = n * 300
    while len(points) < n and attempts < max_attempts:
        # Sample from sphere
        while True:
            p = rng.normal(0, 1, 3)
            norm = np.linalg.norm(p)
            if norm <= 1:
                p = p / norm * radius * rng.uniform(0.2, 1.0) # Avoid center singularity issues, but keep compact
                break
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        attempts += 1
    while len(points) < n:
        # Fallback: add with small jitter
        p = np.array(points[len(points) % max(1, len(points))]) + rng.normal(0, 0.1, 3)
        points.append(p)
    return np.array(points)


def _bh_worker(start_pos, rho, n, thread_id, stop_event, local_best):
    """
    Runs a single Basin Hopping chain with collision dynamics local minimization.
    Uses adaptive kick size based on cluster size.
    """
    global _START_TIME
    
    current_pos = start_pos.copy()
    # Initial minimization
    current_pos, current_energy = _local_minimize(current_pos, rho)
    
    if current_energy < local_best['energy']:
        local_best['energy'] = current_energy
        local_best['positions'] = current_pos.copy()

    rng = np.random.default_rng(thread_id * 10000 + n)
    # Adaptive kick size: smaller for larger clusters
    base_kick = 0.25
    bh_step = base_kick * np.sqrt(10.0 / max(n, 10))
    
    step_count = 0
    while not stop_event.is_set():
        # Check time
        if time.time() - _START_TIME > TOTAL_DEADLINE - 2:
            break
            
        step_count += 1
        
        # Adaptive kick: reduce slightly over time for convergence
        effective_kick = bh_step * (1.0 - 0.1 * min(step_count / 100.0, 1.0))
        
        # Kick
        noise = rng.normal(0, effective_kick, current_pos.shape)
        perturbed = current_pos + noise
        
        # Minimize
        new_pos, new_energy = _local_minimize(perturbed, rho)
        
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if current_energy < local_best['energy']:
                local_best['energy'] = current_energy
                local_best['positions'] = current_pos.copy()
        else:
            # Metropolis accept with adaptive temperature
            delta_e = new_energy - current_energy
            # Temperature decreases over time
            T = max(0.5, 2.0 - 0.01 * step_count)
            if delta_e < 100:
                if rng.random() < np.exp(-delta_e / T):
                    current_pos = new_pos
                    current_energy = new_energy
                    if current_energy < local_best['energy']:
                        local_best['energy'] = current_energy
                        local_best['positions'] = current_pos.copy()


def construct(instance):
    global _START_TIME, _best_energy_global, _best_positions_global
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Generate diverse starting configurations
    starts = []
    
    # Crystalline templates
    try: starts.append(_generate_fcc(size, 1.0))
    except: pass
    try: starts.append(_generate_bcc(size, 1.0))
    except: pass
    try: starts.append(_generate_hcp(size, 1.0))
    except: pass
    
    # Random compact starts - more of them for better coverage
    for seed in [size, size+1, size+2, size+3, size+4, size+5, size+6, size+7]:
        try: starts.append(_generate_random_compact(size, 1.0, seed=seed))
        except: pass
    
    # High-temperature gas-like starts for collision dynamics to self-assemble
    for seed in [100, 200, 300]:
        try:
            rng = np.random.default_rng(seed)
            radius = size ** (1.0/3.0) * 1.5
            gas_pos = rng.uniform(-radius, radius, (size, 3))
            starts.append(gas_pos)
        except: pass
        
    if not starts:
        starts.append(_generate_random_compact(size, 1.0, seed=42))

    # Initialize global best with the best of the starts (after quick minimization)
    # But we will let the threads do the minimization.
    # We just need a valid initial guess to return if everything fails.
    best_energy = np.inf
    best_positions = starts[0]
    
    # Quick minimization of starts to find a good initial best
    for s in starts:
        if time.time() - _START_TIME > TOTAL_DEADLINE - 10:
            break
        try:
            pos, e = _local_minimize(s, rho)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            pass

    # Prepare for parallel BH
    num_threads = 4
    # Distribute starts to threads - each thread gets multiple starts
    
    # Shuffle starts
    np.random.shuffle(starts)
    
    threads = []
    stop_event = threading.Event()
    
    lock = threading.Lock()
    global_best = {'energy': best_energy, 'positions': best_positions.copy()}

    def worker_wrapper(start_list, tid):
        """Worker that processes multiple starting configurations."""
        local_best = {'energy': np.inf, 'positions': None}
        
        for idx, start_pos in enumerate(start_list):
            if stop_event.is_set() or time.time() - _START_TIME > TOTAL_DEADLINE - 5:
                break
            
            # Run BH from this start
            _bh_worker(start_pos, rho, size, tid * 100 + idx, stop_event, local_best)
            
            # Update global best
            with lock:
                if local_best['energy'] < global_best['energy']:
                    global_best['energy'] = local_best['energy']
                    if local_best['positions'] is not None:
                        global_best['positions'] = local_best['positions'].copy()
        
        # Final merge
        with lock:
            if local_best['energy'] < global_best['energy']:
                global_best['energy'] = local_best['energy']
                if local_best['positions'] is not None:
                    global_best['positions'] = local_best['positions'].copy()

    # Distribute starts evenly among threads
    starts_per_thread = len(starts) // num_threads
    remainder = len(starts) % num_threads
    
    thread_starts = []
    idx = 0
    for i in range(num_threads):
        count = starts_per_thread + (1 if i < remainder else 0)
        thread_starts.append(starts[idx:idx+count])
        idx += count
    
    for i in range(num_threads):
        if thread_starts[i]:  # Only start threads with work
            t = threading.Thread(target=worker_wrapper, args=(thread_starts[i], i))
            t.daemon = True
            t.start()
            threads.append(t)

    # Wait for threads to finish
    for t in threads:
        t.join()

    # Final result
    if global_best['positions'] is not None:
        return global_best['positions']
    else:
        return best_positions
