# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _vectorized_morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient using vectorization for speed."""
    n = positions.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(positions)
        
    # Compute pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    # Mask for upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_pair = r[mask]
    diff_pair = diff[mask]
    
    if len(r_pair) == 0:
        return 0.0, np.zeros_like(positions)
        
    # Calculate energy terms
    rho_one_minus_r = rho * (1.0 - r_pair)
    exp_val = np.exp(rho_one_minus_r)
    e_pair = exp_val * (exp_val - 2.0)
    energy = np.sum(e_pair)
    
    # Calculate gradient terms
    # dE/dr = -rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 1)
    de_dr = -rho * exp_val * (2.0 * exp_val - 1.0)
    
    # Safe division
    safe_r = np.where(r_pair < 1e-12, 1e-12, r_pair)
    unit = diff_pair / safe_r[:, np.newaxis]
    forces = de_dr[:, np.newaxis] * unit
    
    # Accumulate gradients
    grad = np.zeros_like(positions)
    i_idx, j_idx = np.nonzero(mask)
    
    np.add.at(grad, i_idx, forces)
    np.add.at(grad, j_idx, -forces)
    
    return energy, grad


def _local_minimize(positions, rho, maxiter=100):
    """Perform local minimization using L-BFGS-B with vectorized gradient."""
    n = positions.shape[0]
    x0 = positions.flatten()
    
    def fun(x):
        return _vectorized_morse_energy_and_grad(x.reshape(-1, 3), rho)[0]
        
    def jac(x):
        return _vectorized_morse_energy_and_grad(x.reshape(-1, 3), rho)[1].flatten()
        
    try:
        result = minimize(
            fun=fun,
            x0=x0,
            jac=jac,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return positions, _vectorized_morse_energy_and_grad(positions, rho)[0]


def _random_compact_cluster(n, radius, min_dist, rng):
    """Generate a random cluster with minimum distance constraint."""
    points = []
    attempts = 0
    max_attempts = n * 1000
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        
        if valid:
            points.append(p)
            
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)


def _basin_hopping(positions, rho, time_limit, rng):
    """
    Perform Basin Hopping optimization.
    """
    start_time = time.time()
    
    # Initial relaxation
    x_best, e_best = _local_minimize(positions, rho)
    
    T = 10.0  # Initial temperature
    T_min = 1e-4
    cooling_rate = 0.95
    step_size = 0.5
    
    best_energy = e_best
    best_positions = x_best.copy()
    
    while (time.time() - start_time) < time_limit:
        # Perturb
        noise = rng.normal(0, step_size, size=(positions.shape[0], 3))
        x_trial = x_best + noise
        
        # Relax trial
        x_trial_relaxed, e_trial = _local_minimize(x_trial, rho)
        
        # Metropolis acceptance
        delta_e = e_trial - e_best
        if delta_e < 0:
            # Accept
            x_best = x_trial_relaxed
            e_best = e_trial
            if e_best < best_energy:
                best_energy = e_best
                best_positions = x_trial_relaxed.copy()
        else:
            # Accept with probability
            if T > T_min:
                prob = np.exp(-delta_e / T)
                if rng.random() < prob:
                    x_best = x_trial_relaxed
                    e_best = e_trial
        
        # Cool
        T *= cooling_rate
        
    return best_positions, best_energy


def construct(instance):
    """
    Find low-energy Morse clusters using Basin Hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    if potential != "morse":
        # Fallback
        spacing = 1.0
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.8
        points = _random_compact_cluster(size, radius, 0.55, rng)
        # Simple relax for non-morse
        try:
            from autoresearch_suite.chem_clusters import relax
            _, positions = relax(points, potential, rho)
            return positions
        except:
            return points
            
    # Time budget per instance
    # Total 300s for 6 instances.
    # Budget ~ 45s per instance.
    # We will run 2 independent BH chains of 22.5s each.
    
    time_budget = 45.0
    rng = np.random.default_rng()
    
    best_positions = None
    best_energy = np.inf
    
    num_chains = 2
    time_per_chain = time_budget / num_chains
    
    for i in range(num_chains):
        # Generate diverse start
        radius = 0.6 * size ** (1.0 / 3.0) + 0.5
        x0 = _random_compact_cluster(size, radius, 0.55, rng)
        
        pos, e = _basin_hopping(x0, rho, time_per_chain, rng)
        
        if e < best_energy:
            best_energy = e
            best_positions = pos
            
    if best_positions is None:
        radius = 0.6 * size ** (1.0 / 3.0) + 0.5
        best_positions = _random_compact_cluster(size, radius, 0.55, rng)
        best_positions, _ = _local_minimize(best_positions, rho)
        
    return best_positions
# EVOLVE-BLOCK-END
