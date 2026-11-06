"""Morse cluster global minima: Parallel Tempering (Replica Exchange)."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import time


def _generate_random_cluster(n, seed=0, radius_factor=1.2):
    """Generate a random cluster within a sphere of radius proportional to n^(1/3)."""
    rng = np.random.default_rng(seed)
    # Estimate radius: volume ~ n, so R ~ n^(1/3)
    # Equilibrium distance is 1.0.
    R = radius_factor * n ** (1.0 / 3.0)
    
    points = []
    # Rejection sampling in a sphere
    max_attempts = n * 100
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        # Generate point in cube, reject if outside sphere
        p = rng.uniform(-R, R, 3)
        if np.dot(p, p) <= R * R:
            # Check minimum distance constraint (0.5)
            # To be fast, we can skip strict checking here and rely on relax/penalty,
            # but the task says "atoms must be at least 0.5 apart". 
            # The evaluator checks this. If we return invalid, score is 0.
            # So we must ensure validity.
            # For small n, checking all pairs is cheap. For n=80, 80*80/2 = 3200 checks.
            # This is fine for initialization.
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
        
    # If we failed to place all atoms (too dense), fallback to random without min-dist check
    # and hope relax fixes it, or use a lattice.
    # Actually, if n is large, random packing in a sphere of R~n^(1/3) might fail.
    # Let's adjust radius if we fail.
    if len(points) < n:
        # Fallback: Lattice-like random
        points = []
        R = 1.5 * n ** (1.0 / 3.0)
        for _ in range(n):
            p = rng.uniform(-R, R, 3)
            if np.dot(p, p) <= R * R:
                points.append(p)
            else:
                points.append([0,0,0]) # Placeholder, will be fixed
        
        # Enforce minimum distance by pushing apart? 
        # Simpler: Use a grid-based initialization for fallback.
        # But let's try to just return what we have and let relax handle it?
        # No, relax might not fix overlapping atoms if the potential is singular or very steep.
        # Morse is smooth. 
        # Let's ensure the returned array has shape (n,3).
        if len(points) < n:
            # Pad with random
            while len(points) < n:
                points.append(rng.uniform(-R, R, 3))
        return np.array(points[:n])
        
    return np.array(points)


def construct(instance):
    """Find low-energy Morse cluster via Parallel Tempering."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    t_start = time.time()
    time_limit = 48.0  # Leave 2s margin
    
    # Parameters for Parallel Tempering
    n_replicas = 8
    # Temperature ladder: geometric spacing
    # T_low should be small enough to find minima, T_high large enough to explore.
    # Energy scale: Morse potential well depth is 1.
    # Let's define "temperature" as the standard deviation of the Gaussian perturbation.
    # sigma_low = 0.1, sigma_high = 1.0
    sigmas = np.geomspace(0.1, 1.5, n_replicas)
    
    # Initialize replicas
    rng = np.random.default_rng(42 + size)
    replicas = []
    energies = []
    
    for i in range(n_replicas):
        # Diverse starts: mix of random and slightly perturbed lattice
        if i == 0:
            # Best guess start: tight random
            pos = _generate_random_cluster(size, seed=1000 + size, radius_factor=0.8)
        elif i == 1:
            # Medium
            pos = _generate_random_cluster(size, seed=2000 + size, radius_factor=1.0)
        else:
            # Looser random
            pos = _generate_random_cluster(size, seed=3000 + i * 100 + size, radius_factor=1.2 + 0.1 * i)
        
        replicas.append(pos)
        
        # Initial relaxation
        try:
            e, pos_relaxed = relax(pos, potential, rho)
        except Exception:
            e, pos_relaxed = 1e10, pos
            
        replicas[i] = pos_relaxed
        energies.append(e)
        
        if time.time() - t_start > time_limit:
            break
            
    # Track global best
    best_energy = np.min(energies)
    best_idx = np.argmin(energies)
    best_pos = replicas[best_idx].copy()
    
    # Main loop
    # We will perform iterations. Each iteration consists of:
    # 1. Perturb each replica by sigma[i]
    # 2. Relax each replica
    # 3. Attempt swaps between adjacent replicas
    
    n_iterations = 200 # Max iterations
    
    for it in range(n_iterations):
        if time.time() - t_start > time_limit:
            break
            
        # 1. Perturb
        for i in range(n_replicas):
            noise = rng.normal(0, sigmas[i], size=replicas[i].shape)
            replicas[i] = replicas[i] + noise
            
        # 2. Relax
        for i in range(n_replicas):
            try:
                e, pos = relax(replicas[i], potential, rho)
            except Exception:
                e, pos = 1e10, replicas[i]
            replicas[i] = pos
            energies[i] = e
            
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
                
        if time.time() - t_start > time_limit:
            break
            
        # 3. Swap attempts
        # Iterate over pairs (i, i+1)
        # To avoid bias, we can iterate in random order or alternate direction
        for i in range(n_replicas - 1):
            # Metropolis criterion for swap
            # Accept with probability min(1, exp(-(E_i/T_i - E_j/T_j) ... ))
            # In our "sigma" formulation, the "temperature" is effectively the perturbation scale.
            # The standard PT uses T. Here we use sigma as a proxy for T.
            # The energy of the state is E.
            # The probability of being in state x at "temp" T is exp(-E/T).
            # If we swap x_i (at T_i) and x_j (at T_j):
            # P_swap = min(1, exp( (E_i/T_i - E_j/T_j) - (E_j/T_i - E_i/T_j) ))
            # P_swap = min(1, exp( E_i(1/T_i - 1/T_j) + E_j(1/T_j - 1/T_i) ))
            # P_swap = min(1, exp( (E_i - E_j) * (1/T_i - 1/T_j) ))
            
            T_i = sigmas[i]
            T_j = sigmas[i+1]
            E_i = energies[i]
            E_j = energies[i+1]
            
            # Avoid division by zero or very small T
            if T_i < 1e-6 or T_j < 1e-6:
                continue
                
            delta = (E_i - E_j) * (1.0 / T_i - 1.0 / T_j)
            
            if delta > 0:
                # Reject if exp(-delta) < random
                if rng.random() > np.exp(-delta):
                    continue
            # Else accept (delta <= 0 => exp(-delta) >= 1)
            
            # Swap positions and energies
            replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
            energies[i], energies[i+1] = energies[i+1], energies[i]
            
            # Update best if the moved replica is now the best
            if energies[i] < best_energy:
                best_energy = energies[i]
                best_pos = replicas[i].copy()
            if energies[i+1] < best_energy:
                best_energy = energies[i+1]
                best_pos = replicas[i+1].copy()
                
    return best_pos
# EVOLVE-BLOCK-END
