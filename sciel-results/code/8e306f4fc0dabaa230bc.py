"""New program: Parallel Simulated Annealing with Local Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    iu = np.triu_indices(n, k=1)
    xij = x[iu]
    e = np.exp(xij) * (np.exp(xij) - 2.0)
    return float(np.sum(e))


def _check_min_dist(pos, min_d=0.5):
    """Check if all pairs are at least min_d apart."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_d


def _sa_worker(n, rho, seed, time_budget, result_holder, lock):
    """Simulated Annealing worker thread."""
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Estimate radius: volume ~ n * (4/3 pi r_atom^3), but atoms are packed.
    # Heuristic: R ~ 1.5 * n^(1/3)
    radius = 1.5 * n ** (1.0 / 3.0)
    
    # Initialize positions: Random in sphere, ensure min dist
    pos = np.zeros((n, 3))
    points = []
    while len(points) < n:
        # Sample in sphere
        v = rng.normal(0, 1, 3)
        norm = np.linalg.norm(v)
        if norm > radius:
            continue
        p = v / norm * (radius * np.cbrt(rng.random()))
        # Check min dist
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    pos = np.array(points)
    
    current_energy = _morse_energy(pos, rho)
    best_energy = current_energy
    best_pos = pos.copy()
    
    T_start = 2.0
    T_end = 0.05
    
    step_count = 0
    last_refine_time = 0
    
    while time.time() - start_time < time_budget:
        # Linear cooling in time
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_budget)
        T = T_start * (1.0 - progress) + T_end * progress
        
        # Propose move
        idx = rng.integers(0, n)
        step_size = 0.1 * T + 0.02
        new_pos = pos.copy()
        new_pos[idx] += rng.normal(0, step_size, 3)
        
        # Check constraints
        if not _check_min_dist(new_pos, 0.5):
            continue
            
        new_energy = _morse_energy(new_pos, rho)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = pos.copy()
                
        step_count += 1
        
        # Refine every ~1000 steps or if 0.1s passed
        if step_count % 1000 == 0 or (time.time() - last_refine_time > 0.1 and step_count > 100):
            last_refine_time = time.time()
            try:
                # Use scipy minimize with gradient
                res = minimize(
                    lambda x: _morse_energy(x.reshape(-1, 3), rho),
                    pos.flatten(),
                    jac=lambda x: morse_energy_gradient(x.reshape(-1, 3), rho).flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 20, 'ftol': 1e-4}
                )
                new_pos_opt = res.x.reshape(-1, 3)
                if _check_min_dist(new_pos_opt, 0.5):
                    new_e_opt = _morse_energy(new_pos_opt, rho)
                    if new_e_opt < current_energy:
                        pos = new_pos_opt
                        current_energy = new_e_opt
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_pos = pos.copy()
            except Exception:
                pass

    # Final refinement
    try:
        res = minimize(
            lambda x: _morse_energy(x.reshape(-1, 3), rho),
            best_pos.flatten(),
            jac=lambda x: morse_energy_gradient(x.reshape(-1, 3), rho).flatten(),
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        final_pos = res.x.reshape(-1, 3)
        if _check_min_dist(final_pos, 0.5):
            final_energy = _morse_energy(final_pos, rho)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = final_pos
    except Exception:
        pass

    with lock:
        if result_holder['best_energy'] is None or best_energy < result_holder['best_energy']:
            result_holder['best_energy'] = best_energy
            result_holder['best_pos'] = best_pos


def construct(instance):
    """Parallel Simulated Annealing with Local Refinement."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    start_time = time.time()
    total_time_budget = 280.0 # Leave 20s margin
    
    # Use 4 threads
    n_threads = 4
    time_per_thread = total_time_budget / n_threads
    
    result_holder = {'best_energy': None, 'best_pos': None}
    lock = threading.Lock()
    
    threads = []
    for i in range(n_threads):
        t = threading.Thread(
            target=_sa_worker,
            args=(size, rho, 42 + i, time_per_thread, result_holder, lock)
        )
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    if result_holder['best_pos'] is not None:
        return result_holder['best_pos']
    else:
        # Fallback: return random valid config
        rng = np.random.default_rng(0)
        pos = np.zeros((size, 3))
        for i in range(size):
            pos[i] = rng.normal(0, 1, 3)
        return pos
# EVOLVE-BLOCK-END
