# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy(pos, rho):
    """
    Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij)) (exp(rho(1-r_ij)) - 2).
    Vectorized for speed.
    """
    n = pos.shape[0]
    # Compute pairwise distances
    # diff[i, j, k] = pos[i, k] - pos[j, k]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # We only need upper triangle for sum, but computing all is easier for vectorization
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    e = np.exp(rho * (1.0 - dists))
    pair_energy = e * (e - 2.0)
    
    # Sum over i < j
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    total_energy = 0.5 * np.sum(pair_energy[mask])
    return total_energy

def _morse_gradient(pos, rho):
    """
    Compute gradient of Morse energy.
    dE/dx_i = sum_{j!=i} dE/dr_ij * (x_i - x_j)/r_ij
    dE/dr = rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 2)
    """
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    e = np.exp(rho * (1.0 - dists))
    # dE/dr_ij
    dE_dr = rho * e * (2.0 * e - 2.0)
    
    # grad_i = sum_j dE_dr_ij * (x_i - x_j) / r_ij
    # factor[i, j] = dE_dr[i, j] / r_ij
    inv_dists = 1.0 / dists
    np.fill_diagonal(inv_dists, 0.0)
    
    factor = dE_dr * inv_dists
    # Zero out diagonal to be safe (though dE_dr is defined, distance is 0)
    np.fill_diagonal(factor, 0.0)
    
    # grad = sum_j factor[i, j] * diff[i, j]
    # diff[i, j] = pos[i] - pos[j]
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return grad

def _local_optimize(pos, rho, time_limit=5.0):
    """
    Run L-BFGS-B local optimization.
    Returns (energy, positions).
    """
    n = pos.shape[0]
    
    def fun(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)
        
    def jac(x):
        p = x.reshape(n, 3)
        return _morse_gradient(p, rho).ravel()

    x0 = pos.ravel()
    start_time = time.time()
    
    # Use a callback to stop if time limit is exceeded
    stop_flag = [False]
    def callback(xk):
        if time.time() - start_time > time_limit:
            stop_flag[0] = True
        return stop_flag[0]

    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20},
                       callback=callback)
        if res.success or res.fun < np.inf:
            return res.fun, res.x.reshape(n, 3)
        else:
            # If it failed, return the last known best? Or just current?
            # L-BFGS-B usually returns the best point found even if not converged.
            return res.fun, res.x.reshape(n, 3)
    except Exception:
        return np.inf, pos

def _simulate_annealing(pos, rho, steps=2000, t_start=1.0, t_end=0.01, time_limit=10.0):
    """
    Simple Simulated Annealing.
    """
    n = pos.shape[0]
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    start_time = time.time()
    
    for i in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (i / steps)
        
        # Propose move: random displacement
        # Scale step size with temperature?
        step_size = 0.1 + 0.5 * (t / t_start)
        displacement = np.random.normal(0, step_size, (n, 3))
        new_pos = current_pos + displacement
        
        # Check constraints (min distance 0.5)
        # If constraint violated, reject or penalize. 
        # Simple rejection is safer for "hard" constraints.
        # Computing full distance matrix is expensive. 
        # Approximation: check only if displacement is large? 
        # Or just accept and let energy handle it? 
        # The task says "atoms must be at least 0.5 apart". 
        # If we allow violations, the final result must be valid.
        # Better to use a soft constraint in energy or hard check.
        # Hard check is O(N^2). With N=80, 6400 pairs. 2000 steps * 6400 = 12.8M ops. Fast enough in Numpy.
        
        # Let's just calculate energy. If min dist < 0.5, add huge penalty.
        # Or simply reject if min dist < 0.5.
        # Let's try rejecting invalid moves to keep trajectory valid.
        
        # Efficient min distance check?
        # For now, calculate energy. If we want to be strict:
        # dists = np.linalg.norm(new_pos[:, np.newaxis] - new_pos[np.newaxis], axis=2)
        # np.fill_diagonal(dists, np.inf)
        # if np.min(dists) < 0.5: continue
        
        # Let's calculate energy first.
        new_energy = _morse_energy(new_pos, rho)
        
        # Check validity
        # To save time, we can skip the min-dist check in SA and rely on the fact that 
        # Morse potential goes to infinity as r->0, so energy will be huge.
        # But the evaluator checks min dist >= 0.5. 
        # If we end up with < 0.5, we fail.
        # So we must ensure final config is valid.
        # We can add a penalty term to energy if min dist < 0.5.
        # Let's add a penalty to the energy calculation used in SA.
        
        # Actually, _morse_energy is expensive to modify.
        # Let's just check min dist occasionally or at the end.
        # Given the time limit, let's assume standard SA moves are small enough 
        # or that the potential repulsion keeps them apart.
        # However, random moves can push atoms together.
        
        # Let's do a quick check:
        # If we are worried, we can just use the provided `relax` which might handle constraints?
        # No, `relax` is local.
        
        # Let's stick to standard SA with energy acceptance.
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / t):
            current_pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
    return best_energy, best_pos

def _generate_lattice_guess(n, lattice_type='fcc', spacing=1.0):
    """Generate a cluster of n atoms from a lattice cut."""
    if lattice_type == 'fcc':
        # FCC primitive vectors
        a = spacing * 2**0.5 # nearest neighbor distance = a/sqrt(2) = 1
        # Basis
        basis = np.array([
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0]
        ]) * a
        # Generate points in a cube
        L = int((n / 4)**(1/3)) + 2 # 4 atoms per unit cell
        points = []
        for i in range(-L, L+1):
            for j in range(-L, L+1):
                for k in range(-L, L+1):
                    cell_center = a * np.array([i, j, k])
                    for b in basis:
                        p = cell_center + b
                        points.append(p)
        points = np.array(points)
        # Center
        points -= np.mean(points, axis=0)
        # Sort by distance from center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
        
    elif lattice_type == 'bcc':
        # BCC
        a = spacing / (np.sqrt(3)/2) # nearest neighbor = a*sqrt(3)/2 = 1
        basis = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5]
        ]) * a
        L = int((n / 2)**(1/3)) + 2
        points = []
        for i in range(-L, L+1):
            for j in range(-L, L+1):
                for k in range(-L, L+1):
                    cell_center = a * np.array([i, j, k])
                    for b in basis:
                        p = cell_center + b
                        points.append(p)
        points = np.array(points)
        points -= np.mean(points, axis=0)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
        
    elif lattice_type == 'sc':
        # Simple Cubic
        a = spacing
        L = int(n**(1/3)) + 2
        points = []
        for i in range(-L, L+1):
            for j in range(-L, L+1):
                for k in range(-L, L+1):
                    p = a * np.array([i, j, k])
                    points.append(p)
        points = np.array(points)
        points -= np.mean(points, axis=0)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
        
    else:
        return np.random.uniform(-2, 2, (n, 3))

def construct(instance):
    """
    Find low-energy Morse cluster configurations.
    """
    start_time = time.time()
    deadline = 330.0
    safety_margin = 5.0
    effective_deadline = deadline - safety_margin
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42 + n)
    
    best_energy = np.inf
    best_pos = None
    
    # 1. Generate diverse initial guesses
    guesses = []
    
    # Lattice cuts
    guesses.append(_generate_lattice_guess(n, 'fcc'))
    guesses.append(_generate_lattice_guess(n, 'bcc'))
    guesses.append(_generate_lattice_guess(n, 'sc'))
    
    # Random compact (sphere)
    for _ in range(3):
        # Place atoms in a sphere of radius R ~ n^(1/3)
        r_max = 1.5 * n**(1/3)
        pts = []
        # Use rejection sampling to ensure min dist > 0.5
        # This might be slow for large n if not careful, but N=80 is small.
        # Actually, just random in sphere and let SA/Local fix it?
        # If we start with overlapping atoms, energy is huge, gradient pushes apart.
        # But L-BFGS might struggle.
        # Let's try to generate non-overlapping points.
        attempts = 0
        while len(pts) < n and attempts < 5000:
            p = rng.uniform(-r_max, r_max, 3)
            if np.linalg.norm(p) <= r_max:
                if all(np.linalg.norm(p - q) >= 0.51 for q in pts):
                    pts.append(p)
            attempts += 1
        if len(pts) == n:
            guesses.append(np.array(pts))
        else:
            # Fallback: random uniform
            guesses.append(rng.uniform(-r_max, r_max, (n, 3)))

    # 2. Phase 1: Local Optimization from all guesses
    # Allocate ~40% of time to this
    phase1_time_limit = 0.4 * (effective_deadline - (time.time() - start_time))
    
    for guess in guesses:
        if time.time() - start_time > effective_deadline:
            break
            
        # Run local optimization
        e, p = _local_optimize(guess, rho, time_limit=2.0)
        
        # Validate
        if p.shape[0] == n:
            # Check min distance
            dists = np.linalg.norm(p[:, np.newaxis] - p[np.newaxis], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            
            if min_d >= 0.5 - 1e-6:
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            else:
                # If invalid, try to push apart? Or just ignore.
                # Let's try a quick projection or just ignore.
                # Ignoring is safer for now.
                pass

    # 3. Phase 2: Simulated Annealing from best config and a few others
    # Allocate ~30% of time
    phase2_time_limit = 0.3 * (effective_deadline - (time.time() - start_time))
    
    sa_starts = [best_pos] if best_pos is not None else [guesses[0]]
    # Add a couple of random perturbations of best
    if best_pos is not None:
        for _ in range(2):
            perturbed = best_pos + rng.normal(0, 0.5, (n, 3))
            sa_starts.append(perturbed)
    
    for start_pos in sa_starts:
        if time.time() - start_time > effective_deadline:
            break
        
        e_sa, p_sa = _simulate_annealing(start_pos, rho, steps=1000, time_limit=phase2_time_limit/len(sa_starts))
        
        # Local optimize the SA result
        if time.time() - start_time < effective_deadline:
            e_loc, p_loc = _local_optimize(p_sa, rho, time_limit=2.0)
            
            dists = np.linalg.norm(p_loc[:, np.newaxis] - p_loc[np.newaxis], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            
            if min_d >= 0.5 - 1e-6:
                if e_loc < best_energy:
                    best_energy = e_loc
                    best_pos = p_loc.copy()

    # 4. Phase 3: Polish with provided relax and high-precision local opt
    # Allocate remaining time
    if best_pos is not None and time.time() - start_time < effective_deadline:
        # Try provided relax
        try:
            e_relax, p_relax = relax(best_pos, 'morse', rho)
            dists = np.linalg.norm(p_relax[:, np.newaxis] - p_relax[np.newaxis], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            if min_d >= 0.5 - 1e-6:
                if e_relax < best_energy:
                    best_energy = e_relax
                    best_pos = p_relax.copy()
        except Exception:
            pass
        
        # Final high precision local opt
        if time.time() - start_time < effective_deadline - 1.0:
            e_final, p_final = _local_optimize(best_pos, rho, time_limit=2.0)
            dists = np.linalg.norm(p_final[:, np.newaxis] - p_final[np.newaxis], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            if min_d >= 0.5 - 1e-6:
                if e_final < best_energy:
                    best_energy = e_final
                    best_pos = p_final.copy()

    # Fallback
    if best_pos is None:
        # Return a valid configuration
        # Line
        best_pos = np.zeros((n, 3))
        for i in range(n):
            best_pos[i, 0] = i * 1.0
        best_pos -= best_pos.mean(axis=0)
        
    return best_pos

# EVOLVE-BLOCK-END
