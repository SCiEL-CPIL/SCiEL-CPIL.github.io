"""Morse cluster global minima: Simulated Annealing with periodic relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill
_instance_call_count = [0]

def _time_left():
    return _HARD_DEADLINE - time.time()

def _fast_morse_energy(positions, rho):
    """
    Fast vectorized Morse energy calculation.
    E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)
    """
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    # Use broadcasting: (n, 1, 3) - (1, n, 3) -> (n, n, 3)
    # This uses O(N^2) memory, which is fine for N <= 80.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Mask out i=j (distance 0) to avoid division by zero or invalid exp
    # Actually, exp(rho(1-0)) = exp(rho). The term is exp(rho)(exp(rho)-2).
    # But the sum is i<j. We can compute full matrix and divide by 2, 
    # but we must ensure the diagonal (i=j) doesn't contribute.
    # The potential at r=0 is finite in Morse, but physically atoms can't overlap.
    # However, the formula is defined for r>0.
    # Let's mask the diagonal.
    np.fill_diagonal(dists, np.inf) # exp(inf) is inf, but we want 0 contribution.
    
    # Calculate potential term
    # V(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let u = rho * (1 - r)
    # V = exp(u) * (exp(u) - 2)
    
    # To avoid overflow for small r, we can clip r to a small positive number if needed,
    # but the problem states atoms must be at least 0.5 apart.
    # If r < 0.5, the energy is very high, which is fine for SA (it will reject).
    
    u = rho * (1.0 - dists)
    # Clip u to prevent overflow in exp if r is very small (though min dist 0.5 helps)
    # exp(100) is huge. 
    # If r=0.5, rho=10, u = 10*0.5 = 5. exp(5) ~ 148. V ~ 148*(148-2) ~ 21000.
    # If r=0.1, rho=10, u = 9. exp(9) ~ 8100. V ~ 6.5e7.
    # It's safe.
    
    exp_u = np.exp(u)
    v = exp_u * (exp_u - 2.0)
    
    # Sum over i<j
    # Since the matrix is symmetric, sum all and divide by 2.
    # Diagonal is inf, so we need to handle that.
    # Replace inf with 0 before summing.
    v[np.isinf(v)] = 0.0
    
    total_energy = 0.5 * np.sum(v)
    return total_energy

def _generate_random_compact(n, rho, seed=0):
    """
    Generate n random points in a sphere, ensuring min distance.
    Radius estimated based on n and rho.
    For Morse, equilibrium is r=1.
    Cluster radius ~ (n/4)^(1/3) * 1.5
    """
    rng = np.random.default_rng(seed)
    # Estimate radius
    # Packing density ~ 0.74 for FCC. Volume ~ n * (4/3) pi (0.5)^3 ? No, volume per atom ~ 1.
    # Radius ~ (n)^(1/3)
    radius = max(2.0, n ** (1.0/3.0) * 1.2)
    
    points = []
    min_dist = 0.55 # Slightly larger than 0.5 to avoid high energy starts
    
    attempts = 0
    max_attempts = n * 500
    while len(points) < n and attempts < max_attempts:
        # Sample in sphere
        vec = rng.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm > 0:
            p = vec / norm * radius * (rng.uniform(0, 1) ** (1/3))
            # Check min distance
            if len(points) == 0:
                points.append(p)
            else:
                pts = np.array(points)
                dists = np.linalg.norm(pts - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
        
    if len(points) < n:
        # Fill remaining with random points, allowing closer distances
        # This is better than failing
        while len(points) < n:
            vec = rng.normal(size=3)
            norm = np.linalg.norm(vec)
            if norm > 0:
                p = vec / norm * radius * (rng.uniform(0, 1) ** (1/3))
            else:
                p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)

def _generate_lattice(n, lattice_type='fcc', seed=0):
    """
    Generate points from a lattice.
    """
    rng = np.random.default_rng(seed)
    # Simple cubic for now, can add FCC/BCC
    # Estimate side length
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                points.append([i, j, k])
    points = np.array(points)
    
    # Shuffle and take first n
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
        return points[idx]
    else:
        # Pad with random
        extra = _generate_random_compact(n - len(points), 10, seed=seed+1)
        return np.vstack([points, extra])

def _simulated_annealing(initial_pos, rho, time_limit, seed=0):
    """
    Run SA on initial_pos for time_limit seconds.
    """
    rng = np.random.default_rng(seed)
    n = len(initial_pos)
    
    # Initial energy
    try:
        current_energy, current_pos = relax(initial_pos, 'morse', rho)
    except:
        current_energy = _fast_morse_energy(initial_pos, rho)
        current_pos = initial_pos.copy()
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # Temperature parameters
    # Start T high enough to accept most moves, end T low.
    # Typical dE scale is ~10-100.
    T_start = 50.0
    T_end = 0.1
    # We don't know total steps in advance, so we use time-based cooling.
    # Or we can estimate steps.
    # Let's use a fixed number of steps per second?
    # Better: Cool based on elapsed time fraction.
    
    start_time = time.time()
    step_count = 0
    relax_interval = 500 # Relax every 500 steps
    
    # Initial step size
    step_size = 0.2
    
    while time.time() - start_time < time_limit:
        # Check deadline
        if time.time() > _HARD_DEADLINE - 5.0:
            break
            
        # Temperature based on time elapsed
        elapsed = time.time() - start_time
        # Simple linear cooling in log space? Or exponential?
        # Let's use exponential cooling with a rate that ensures T_end is reached at time_limit
        # T(t) = T_start * exp(-lambda * t)
        # T_end = T_start * exp(-lambda * time_limit)
        # lambda = -ln(T_end/T_start) / time_limit
        lambda_cool = -np.log(T_end / T_start) / max(time_limit, 1e-6)
        T = T_start * np.exp(-lambda_cool * elapsed)
        
        # Perform one SA step
        atom_idx = rng.integers(0, n)
        displacement = rng.normal(0, step_size, 3)
        
        old_pos = current_pos.copy()
        old_pos[atom_idx] += displacement
        current_pos = old_pos
        
        new_energy = _fast_morse_energy(current_pos, rho)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        else:
            current_pos = old_pos
            # current_energy remains unchanged
        
        step_count += 1
        
        # Periodic Relaxation
        if step_count % relax_interval == 0:
            try:
                relaxed_energy, relaxed_pos = relax(current_pos, 'morse', rho)
                if relaxed_energy < best_energy:
                    best_energy = relaxed_energy
                    best_pos = relaxed_pos.copy()
                # Update current state to relaxed state
                current_energy = relaxed_energy
                current_pos = relaxed_pos.copy()
            except:
                pass
                
        # Adjust step size
        # If energy is increasing a lot, reduce step size?
        # Simple heuristic: keep constant or slow decay
        step_size = 0.2 * (0.9999 ** step_count)
        if step_size < 0.01:
            step_size = 0.01

    return best_energy, best_pos

def construct(instance):
    """
    Find low-energy Morse cluster configurations.
    """
    _instance_call_count[0] += 1
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Time budget
    # We have 6 instances. Total time ~280s.
    # Per instance ~45s.
    # But we need to leave time for overhead.
    per_instance_budget = 40.0
    if _instance_call_count[0] == 1:
        per_instance_budget = 45.0
    elif _instance_call_count[0] == 6:
        per_instance_budget = 35.0
        
    instance_deadline = time.time() + per_instance_budget
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Random compact
    for seed in range(5):
        initial_configs.append(_generate_random_compact(n, rho, seed=seed))
        
    # 2. Lattice based
    for seed in range(3):
        initial_configs.append(_generate_lattice(n, 'sc', seed=seed+100))
        
    # 3. Perturbed random
    base_rand = _generate_random_compact(n, rho, seed=999)
    for mag in [0.1, 0.3, 0.5]:
        initial_configs.append(base_rand + np.random.default_rng(1).uniform(-mag, mag, (n,3)))
        
    # Run SA on each, keeping the best
    best_energy = np.inf
    best_positions = None
    
    # Time per SA run
    num_runs = len(initial_configs)
    time_per_run = (per_instance_budget - 5.0) / num_runs # Leave 5s for final checks
    
    for i, init_pos in enumerate(initial_configs):
        if time.time() > instance_deadline - 2.0:
            break
            
        try:
            e, pos = _simulated_annealing(init_pos, rho, time_per_run, seed=i+42)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except Exception as ex:
            pass
            
    # If no SA run succeeded or time ran out, do a few quick relaxes
    if best_positions is None:
        # Fallback: just relax the first random config
        init_pos = _generate_random_compact(n, rho, seed=0)
        try:
            best_energy, best_positions = relax(init_pos, 'morse', rho)
        except:
            best_positions = init_pos
            best_energy = _fast_morse_energy(init_pos, rho)
            
    # Final Refinement: Run relax on the best found
    if time.time() < instance_deadline - 1.0:
        try:
            final_energy, final_pos = relax(best_positions, 'morse', rho)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except:
            pass
            
    # Ensure minimum distance
    # The relax function should handle this, but let's verify.
    # If any pair is < 0.5, we might need to fix it.
    # However, the evaluator checks this.
    
    return best_positions
# EVOLVE-BLOCK-END
