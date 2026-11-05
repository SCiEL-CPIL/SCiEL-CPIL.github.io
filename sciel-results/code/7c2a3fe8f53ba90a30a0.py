# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy(pos, rho):
    """
    Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij)) (exp(rho(1-r_ij)) - 2).
    Uses upper triangle only for efficiency.
    """
    n = pos.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = pos[iu[0]] - pos[iu[1]]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    e = np.exp(rho * (1.0 - dists))
    return np.sum(e * (e - 2.0))

def _morse_gradient(pos, rho):
    """
    Compute gradient of Morse energy.
    dE/dx_i = sum_{j!=i} dE/dr_ij * (x_i - x_j)/r_ij
    dE/dr = rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 2)
    """
    n = pos.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = pos[iu[0]] - pos[iu[1]]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    
    e = np.exp(rho * (1.0 - dists))
    dE_dr = rho * e * (2.0 * e - 2.0)
    factor = dE_dr / dists
    
    grad = np.zeros((n, 3))
    # For each pair (i,j) with i<j:
    # grad[i] += factor * diff
    # grad[j] -= factor * diff
    np.add.at(grad, iu[0], factor[:, np.newaxis] * diff)
    np.add.at(grad, iu[1], -factor[:, np.newaxis] * diff)
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
    
    stop_flag = [False]
    def callback(xk):
        if time.time() - start_time > time_limit:
            stop_flag[0] = True
        return stop_flag[0]

    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40},
                       callback=callback)
        return res.fun, res.x.reshape(n, 3)
    except Exception:
        return np.inf, pos

def _simulate_annealing(pos, rho, steps=5000, t_start=2.0, t_end=0.001, time_limit=10.0, rng=None):
    """
    Simulated Annealing with constraint handling.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = pos.shape[0]
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    start_time = time.time()
    
    for i in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        # Geometric cooling
        t = t_start * (t_end / t_start) ** (i / max(steps, 1))
        
        # Adaptive step size
        step_size = 0.05 + 0.3 * (t / t_start)
        displacement = rng.normal(0, step_size, (n, 3))
        new_pos = current_pos + displacement
        
        # Check minimum distance constraint
        iu = np.triu_indices(n, k=1)
        diff = new_pos[iu[0]] - new_pos[iu[1]]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        if np.min(dists) < 0.5:
            continue
        
        new_energy = _morse_energy(new_pos, rho)
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(t, 1e-10)):
            current_pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
    return best_energy, best_pos

def _generate_lattice_guess(n, lattice_type='fcc', spacing=1.0):
    """Generate a cluster of n atoms from a lattice cut."""
    if lattice_type == 'fcc':
        # FCC: nearest neighbor = a/sqrt(2), so a = sqrt(2) for nn=1
        a = np.sqrt(2.0)
        basis = np.array([
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0]
        ]) * a
        L = int((n / 4)**(1/3)) + 3
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
        
    elif lattice_type == 'bcc':
        # BCC: nearest neighbor = a*sqrt(3)/2, so a = 2/sqrt(3) for nn=1
        a = 2.0 / np.sqrt(3.0)
        basis = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5]
        ]) * a
        L = int((n / 2)**(1/3)) + 3
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
        a = 1.0
        L = int(n**(1/3)) + 3
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

def _generate_random_valid(n, rng, min_dist=0.51):
    """Generate random valid configuration with minimum distance constraint."""
    r_max = 2.0 * n**(1/3)
    pts = []
    attempts = 0
    max_attempts = 10000
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) <= r_max:
            if len(pts) == 0 or np.min(np.linalg.norm(np.array(pts) - p, axis=1)) >= min_dist:
                pts.append(p)
        attempts += 1
    if len(pts) == n:
        return np.array(pts)
    # Fallback: grid-based
    side = int(n**(1/3)) + 1
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(pts) < n:
                    pts.append([(i - side//2) * 1.0, (j - side//2) * 1.0, (k - side//2) * 1.0])
    return np.array(pts[:n])

def _check_valid(pos, min_dist=0.5):
    """Check if configuration is valid (min distance constraint)."""
    n = pos.shape[0]
    if n < 2:
        return True
    iu = np.triu_indices(n, k=1)
    diff = pos[iu[0]] - pos[iu[1]]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    return np.min(dists) >= min_dist - 1e-6

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
    
    # Random valid configurations
    for _ in range(3):
        guesses.append(_generate_random_valid(n, rng))

    # 2. Phase 1: Local Optimization from all guesses
    elapsed = time.time() - start_time
    remaining = effective_deadline - elapsed
    phase1_budget = 0.35 * remaining
    
    per_guess_time = min(3.0, phase1_budget / max(len(guesses), 1))
    
    for guess in guesses:
        if time.time() - start_time > effective_deadline:
            break
        time_left = effective_deadline - (time.time() - start_time)
        if time_left < 1.0:
            break
            
        tl = min(per_guess_time, time_left * 0.9)
        e, p = _local_optimize(guess, rho, time_limit=tl)
        
        if _check_valid(p):
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()

    # 3. Phase 2: Simulated Annealing with local polish
    elapsed = time.time() - start_time
    remaining = effective_deadline - elapsed
    phase2_budget = 0.45 * remaining
    
    if best_pos is None:
        best_pos = guesses[0]
        best_energy = _morse_energy(best_pos, rho)
    
    # Multiple SA runs from different starting points
    sa_starts = [best_pos.copy()]
    # Add perturbed versions
    for scale in [0.3, 0.5, 0.8]:
        perturbed = best_pos + rng.normal(0, scale, (n, 3))
        sa_starts.append(perturbed)
    # Add random valid starts
    for _ in range(2):
        sa_starts.append(_generate_random_valid(n, rng))
    
    per_sa_time = phase2_budget / max(len(sa_starts), 1)
    
    for start_pos in sa_starts:
        if time.time() - start_time > effective_deadline:
            break
        time_left = effective_deadline - (time.time() - start_time)
        if time_left < 2.0:
            break
        
        tl_sa = min(per_sa_time * 0.7, time_left * 0.6)
        tl_loc = min(per_sa_time * 0.3, time_left * 0.3)
        
        e_sa, p_sa = _simulate_annealing(start_pos, rho, steps=3000, time_limit=tl_sa, rng=rng)
        
        if _check_valid(p_sa) and e_sa < best_energy:
            best_energy = e_sa
            best_pos = p_sa.copy()
        
        # Polish with local optimization
        if time.time() - start_time < effective_deadline - 1.0:
            e_loc, p_loc = _local_optimize(p_sa, rho, time_limit=tl_loc)
            if _check_valid(p_loc) and e_loc < best_energy:
                best_energy = e_loc
                best_pos = p_loc.copy()

    # 4. Phase 3: Final polish
    if best_pos is not None and time.time() - start_time < effective_deadline - 3.0:
        # Try provided relax
        try:
            e_relax, p_relax = relax(best_pos, 'morse', rho)
            if _check_valid(p_relax) and e_relax < best_energy:
                best_energy = e_relax
                best_pos = p_relax.copy()
        except Exception:
            pass
        
        # Final high-precision local opt
        if time.time() - start_time < effective_deadline - 1.0:
            e_final, p_final = _local_optimize(best_pos, rho, time_limit=2.0)
            if _check_valid(p_final) and e_final < best_energy:
                best_energy = e_final
                best_pos = p_final.copy()

    # Fallback
    if best_pos is None or not _check_valid(best_pos):
        best_pos = _generate_random_valid(n, rng)
        if not _check_valid(best_pos):
            # Absolute fallback: linear arrangement
            best_pos = np.zeros((n, 3))
            for i in range(n):
                best_pos[i, 0] = i * 1.0
            best_pos -= best_pos.mean(axis=0)
        
    return best_pos

# EVOLVE-BLOCK-END
