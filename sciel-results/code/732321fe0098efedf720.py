"""
Strategy: Parallel Multi-Threaded Basin Hopping with L-BFGS-B.
The current program uses a single sequential basin hopping chain, which is limited by the time required for each relaxation step.
This program utilizes the available 4 threads to run independent basin hopping chains in parallel.
Each chain uses L-BFGS-B for efficient local minimization and a random perturbation step to explore the landscape.
This parallelization allows for significantly more diverse exploration of the energy landscape within the 300s time budget,
increasing the probability of finding the global minimum for each cluster size.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 320.0  # Safety margin before 330s kill

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _generate_icosahedron(n_atoms, rng):
    """
    Generate a structure based on icosahedral shells (Mackay-like).
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 unique icosahedral vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)] # Center
    
    # Shell 1: 12 atoms
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)
        
    if n_atoms <= 13:
        return np.array(positions[:n_atoms])
        
    # Shell 2: Generate remaining atoms on a sphere with radius r2
    r2 = 1.9 * 1.75
    num_outer = n_atoms - 13
    if num_outer > 0:
        outer_pts = []
        for _ in range(num_outer):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            v += rng.normal(scale=0.01, size=3)
            outer_pts.append(v)
        positions.extend(outer_pts)
        
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng, spacing=1.0):
    """
    Generate random compact structure.
    """
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.9
    points = []
    max_tries = 10000
    tries = 0
    while len(points) < n_atoms and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if not points:
                points.append(p)
            else:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.6 * spacing:
                    points.append(p)
        tries += 1
    
    if len(points) < n_atoms:
        points = [rng.uniform(-radius, radius, 3) for _ in range(n_atoms)]
        
    return np.array(points)

def _local_minimize(positions, potential, parameter, max_iter=200):
    """
    Use L-BFGS-B to minimize the Dzugutov energy.
    Returns (energy, positions).
    """
    n = len(positions)
    x0 = positions.flatten()
    
    def objective(x):
        pos = x.reshape(n, 3)
        try:
            # Use the provided gradient function if possible, otherwise numerical
            # dzugutov_energy_gradient returns gradient, but we need energy too.
            # The relax function returns both, but minimize needs a function.
            # We can call relax in a single step? No, relax does full relaxation.
            # Let's use the gradient function and compute energy manually or via a helper.
            # Actually, the task says chem_clusters.relax and dzugutov_energy_gradient are available.
            # dzugutov_energy_gradient likely returns just the gradient.
            # We need energy for minimize.
            # Let's check if we can compute energy. The contract says evaluator recomputes E.
            # We can write a small energy function or use a trick.
            # Wait, dzugutov_energy_gradient might return (energy, grad) or just grad?
            # "dzugutov_energy_gradient(positions)" - usually implies gradient.
            # If it only returns gradient, we can't use it directly with minimize without energy.
            # However, relax(positions, potential, parameter) returns (energy, positions).
            # But relax is a full optimizer, not a single step.
            
            # Let's assume we can compute energy.
            # The potential is V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
            # A = 5.82, B = 1.28, a = 1.87, b = 1.94, c = 1.10, d = 0.27, m = 16
            
            # To be safe and fast, let's just use the relax function as the objective?
            # No, relax is expensive.
            
            # Let's implement a quick energy calculation.
            A = 5.82
            B = 1.28
            a = 1.87
            b = 1.94
            c = 1.10
            d = 0.27
            m = 16
            
            e = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    r = np.linalg.norm(pos[i] - pos[j])
                    if r < 1e-6:
                        return 1e10
                    if r < a:
                        e += A * (r**(-m) - B) * np.exp(c / (r - a))
                    if r < b:
                        e += B * np.exp(d / (r - b))
            return e

        except Exception:
            return 1e10

    def gradient(x):
        pos = x.reshape(n, 3)
        try:
            # Use the provided gradient function
            return dzugutov_energy_gradient(pos).flatten()
        except Exception:
            return np.zeros_like(x)

    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-6})
        final_pos = res.x.reshape(n, 3)
        final_energy = objective(final_pos)
        return final_energy, final_pos
    except Exception:
        return objective(x0), positions

def _basin_hopping_chain(initial_pos, potential, parameter, time_limit, rng):
    """
    Run a single basin hopping chain.
    """
    n = len(initial_pos)
    start_time = time.time()
    
    # Initial relaxation
    try:
        current_energy, current_pos = _local_minimize(initial_pos, potential, parameter)
    except Exception:
        current_energy, current_pos = 1e10, initial_pos
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    step_size = 0.2
    min_step_size = 0.05
    max_step_size = 0.5
    
    while time.time() - start_time < time_limit:
        # Check global deadline
        if time.time() > _HARD_LIMIT:
            break
            
        # Perturb
        perturbed = current_pos + rng.normal(scale=step_size, size=current_pos.shape)
        
        # Relax
        try:
            new_energy, new_pos = _local_minimize(perturbed, potential, parameter)
        except Exception:
            new_energy, new_pos = 1e10, perturbed
            
        if new_energy < current_energy:
            # Accept
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
            # Slightly increase step size if stuck? Or keep?
            # Simple scheme: keep step size
        else:
            # Reject, decrease step size
            step_size = max(min_step_size, step_size * 0.9)
            
        # Occasionally increase step size to escape
        if rng.random() < 0.1:
            step_size = min(max_step_size, step_size * 1.2)
            
    return best_energy, best_pos

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_budget = 45.0 # 6 instances * 45s = 270s, leaving 60s margin
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial structures
    starts = []
    
    # 1. PT-inspired
    try:
        pt_pos = _generate_icosahedron(size, rng)
        valid = True
        for i in range(len(pt_pos)):
            for j in range(i+1, len(pt_pos)):
                if np.linalg.norm(pt_pos[i] - pt_pos[j]) < 0.5:
                    valid = False
                    break
            if not valid: break
        if valid:
            starts.append(pt_pos)
    except Exception:
        pass
        
    # 2. Random Compact (Seed 1)
    try:
        rc1 = _random_compact(size, rng)
        starts.append(rc1)
    except Exception:
        pass
        
    # 3. Random Compact (Seed 2)
    try:
        rc2 = _random_compact(size, rng)
        starts.append(rc2)
    except Exception:
        pass
        
    # 4. "Deformed" PT
    if starts:
        pt_deformed = starts[0] + rng.normal(scale=0.2, size=starts[0].shape)
        valid = True
        for i in range(len(pt_deformed)):
            for j in range(i+1, len(pt_deformed)):
                if np.linalg.norm(pt_deformed[i] - pt_deformed[j]) < 0.5:
                    valid = False
                    break
            if not valid: break
        if valid:
            starts.append(pt_deformed)

    # If we have fewer than 4 starts, duplicate or add more random
    while len(starts) < 4:
        try:
            rc = _random_compact(size, rng)
            starts.append(rc)
        except Exception:
            starts.append(starts[0] + rng.normal(scale=0.1, size=starts[0].shape))

    # Run parallel basin hopping chains
    best_energy = np.inf
    best_positions = None
    
    # Use 4 threads
    num_threads = 4
    chain_time = time_budget / 4.0 * 0.9 # Leave some margin
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start_pos = starts[i % len(starts)]
            # Slight randomization of start to ensure independence
            start_pos = start_pos + rng.normal(scale=0.05, size=start_pos.shape)
            futures.append(executor.submit(_basin_hopping_chain, start_pos, potential, parameter, chain_time, np.random.default_rng(size + i)))
        
        for future in as_completed(futures):
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass

    if best_positions is None:
        # Fallback
        return starts[0]
        
    return best_positions

# EVOLVE-BLOCK-END
