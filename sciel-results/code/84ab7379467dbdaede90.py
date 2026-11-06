# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _morse_energy_and_grad_vec(positions, rho):
    """
    Vectorized Morse energy and gradient calculation.
    positions: (N, 3) array
    rho: potential parameter
    Returns: energy (float), gradient (N, 3) array
    """
    n = positions.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute pairwise differences
    # diff[i, j] = positions[i] - positions[j]
    # We only need i < j, but broadcasting computes all.
    # To save memory and time, we can use triu indices or just compute full matrix and take upper triangle.
    # For N=80, N^2 = 6400, which is small.
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask to avoid division by zero and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = dists[mask]
    diffs = diff[mask]
    
    if len(dists) == 0:
        return 0.0, np.zeros_like(positions)
        
    # Morse potential: E = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let u = rho * (1 - r)
    u = rho * (1.0 - dists)
    exp_u = np.exp(u)
    e_pairs = exp_u * (exp_u - 2.0)
    
    energy = np.sum(e_pairs)
    
    # Gradient:
    # dE/dr = -rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 1)
    de_dr = -rho * exp_u * (2.0 * exp_u - 1.0)
    
    # grad_i = sum_{j != i} de_dr_ij * (r_i - r_j) / r_ij
    # grad_j = sum_{i != j} de_dr_ij * (r_j - r_i) / r_ij = - grad_i contribution
    
    # We need to scatter the gradients back to the positions.
    # de_dr is a vector of length M (number of pairs).
    # diffs is (M, 3) where diffs[k] = r_i - r_j for the k-th pair.
    # unit_vectors = diffs / dists[:, np.newaxis]
    # force_vectors = de_dr[:, np.newaxis] * unit_vectors
    
    # For each pair (i, j), force on i is F, force on j is -F.
    # We need indices of i and j for each pair.
    i_idx, j_idx = np.where(mask)
    
    # Calculate force vectors for each pair
    # Avoid division by zero (though mask excludes i==j, dists could be 0 if atoms overlap)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diffs / dists[:, np.newaxis]
        # Handle zero distance (overlap) by setting force to 0 or large? 
        # If dists is 0, the potential is infinite, but numerically we should avoid NaNs.
        unit_vecs = np.nan_to_num(unit_vecs, nan=0.0, posinf=0.0, neginf=0.0)
        
    force_mags = de_dr
    force_vecs = force_mags[:, np.newaxis] * unit_vecs
    
    grad = np.zeros_like(positions)
    np.add.at(grad, i_idx, force_vecs)
    np.add.at(grad, j_idx, -force_vecs)
    
    return energy, grad

def _build_fcc_cluster(n):
    """Generate an FCC-like cluster configuration."""
    points = []
    # FCC basis
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    max_shell = int(np.ceil(2.0 * n ** (1.0 / 3.0))) + 2
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    candidates.append(p)
    
    candidates = np.array(candidates)
    candidates -= np.mean(candidates, axis=0)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    points = candidates[order[:n]]
    return points

def _build_random_compact(n, seed):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 0.7 * n ** (1.0 / 3.0) + 0.5
    min_dist = 0.55
    
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

def construct(instance):
    """
    Find low-energy Morse clusters using Basin Hopping with vectorized optimization.
    """
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    if potential != "morse":
        # Fallback for other potentials (not expected in this task)
        spacing = 1.0
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
        return np.array(points)
        
    # Time management
    start_time = time.time()
    time_limit = 290.0  # Leave 10s buffer
    
    # Initialize with diverse structures
    initial_configs = [
        _build_fcc_cluster(size),
        _build_random_compact(size, 42),
        _build_random_compact(size, 123)
    ]
    
    best_energy = np.inf
    best_positions = None
    rng = np.random.default_rng(0)
    
    # Pre-compute kick size
    # Kick size should be small enough to stay in the basin after relaxation,
    # but large enough to jump to a new basin.
    # Typical interatomic distance is ~1.0. Kick of 0.1-0.5 is reasonable.
    kick_size = 0.25
    
    # Define the local minimizer function
    def local_minimize(x0_flat):
        x0 = x0_flat.reshape(-1, 3)
        result = minimize(
            fun=lambda x: _morse_energy_and_grad_vec(x.reshape(-1, 3), rho)[0],
            x0=x0_flat,
            jac=lambda x: _morse_energy_and_grad_vec(x.reshape(-1, 3), rho)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        return result.fun, result.x
        
    # Main Basin Hopping Loop
    for start_idx, x0 in enumerate(initial_configs):
        current_energy, current_pos_flat = local_minimize(x0.flatten())
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_pos_flat.copy()
            
        # Perform Basin Hopping iterations
        # Number of hops depends on time and cluster size
        # Smaller clusters can afford more hops, larger ones fewer
        max_hops = 200 if size < 50 else 100
        
        for hop in range(max_hops):
            if time.time() - start_time > time_limit:
                break
                
            # Generate kick
            kick = rng.normal(0, kick_size, current_pos_flat.shape)
            new_pos_flat = current_pos_flat + kick
            
            # Relax
            new_energy, new_pos_relaxed_flat = local_minimize(new_pos_flat)
            
            # Accept if better (Hill Climbing)
            # For global minima, simple acceptance is often robust if kicks are well-tuned.
            # Metropolis acceptance can help escape plateaus but might waste time.
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos_flat = new_pos_relaxed_flat
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_pos_flat.copy()
                    
            # Optional: Adapt kick size if stuck
            # If we haven't improved in last 20 hops, increase kick size slightly
            # For simplicity, we keep it constant, but could add logic here.
            
    # Final check and return
    if best_positions is None:
        # Fallback
        best_positions = _build_fcc_cluster(size).flatten()
        
    return best_positions.reshape(-1, 3)
# EVOLVE-BLOCK-END
