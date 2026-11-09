"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16


def _compute_energy(positions):
    """Compute Dzugutov potential energy for a set of positions."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a_POT:
                energy += A_POT * (r ** (-m_POT) - B_POT) * np.exp(c_POT / (r - a_POT))
            elif r < b_POT:
                energy += B_POT * np.exp(d_POT / (r - b_POT))
            else:
                energy += 0.0
    return energy


def _compute_forces(positions):
    """Compute forces on all atoms using the Dzugutov potential."""
    n = len(positions)
    forces = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-12:
                continue
            
            if r < a_POT:
                # V(r) = A * (r^(-m) - B) * exp(c/(r-a))
                # dV/dr = A * [(-m)*r^(-m-1) * exp(c/(r-a)) + (r^(-m) - B) * exp(c/(r-a)) * (-c/(r-a)^2)]
                exp_term = np.exp(c_POT / (r - a_POT))
                r_pow = r ** (-m_POT)
                dV_dr = A_POT * [(-m_POT) * r ** (-m_POT - 1) * exp_term + 
                                  (r_pow - B_POT) * exp_term * (-c_POT / (r - a_POT)**2)]
            elif r < b_POT:
                # V(r) = B * exp(d/(r-b))
                # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
                exp_term = np.exp(d_POT / (r - b_POT))
                dV_dr = B_POT * exp_term * (-d_POT / (r - b_POT)**2)
            else:
                continue
            
            # Force on i is -dV/dr * (r_vec / r), force on j is +dV/dr * (r_vec / r)
            # where r_vec = positions[i] - positions[j]
            force_mag = -dV_dr / r
            f_vec = force_mag * diff
            forces[i] += f_vec
            forces[j] -= f_vec
    
    return forces


def _langevin_step(positions, velocities, dt, temperature, damping, rng):
    """Perform one Langevin dynamics step."""
    n = len(positions)
    forces = _compute_forces(positions)
    
    # Velocity Verlet with Langevin noise
    # v_new = v + (F/m)*dt - gamma*v*dt + sqrt(2*gamma*kT*dt)*noise
    mass = 1.0
    gamma = damping
    kT = temperature
    
    noise = rng.normal(0, np.sqrt(2 * gamma * kT * dt), size=positions.shape)
    velocities = velocities + (forces / mass) * dt - gamma * velocities * dt + noise
    positions = positions + velocities * dt
    
    return positions, velocities


def _self_assemble_cluster(n_target, seed=42, max_time=30.0):
    """
    Physics-driven self-assembly: grow cluster by sequentially adding atoms
    and relaxing via Langevin dynamics.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Start with a small core
    positions = np.array([[0.0, 0.0, 0.0]])
    velocities = np.zeros_like(positions)
    
    # Initial temperature for annealing
    T_start = 0.5
    T_end = 0.05
    damping = 1.0
    dt = 0.005
    
    for n_current in range(1, n_target):
        if time.time() - start_time > max_time:
            break
        
        # Add a new atom at a random position near the existing cluster
        # Place it at a distance that allows bonding but not overlap
        center = np.mean(positions, axis=0)
        radius = np.max(np.linalg.norm(positions - center, axis=1))
        
        # New atom position: random direction, radius + offset
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        offset = rng.uniform(1.0, 1.8)
        new_pos = center + direction * (radius + offset)
        
        # Check minimum distance constraint
        min_dist_ok = True
        for p in positions:
            if np.linalg.norm(new_pos - p) < 0.55:
                min_dist_ok = False
                break
        
        if not min_dist_ok:
            # Try a few more positions
            for _ in range(10):
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                offset = rng.uniform(1.0, 2.5)
                new_pos = center + direction * (radius + offset)
                min_dist_ok = True
                for p in positions:
                    if np.linalg.norm(new_pos - p) < 0.55:
                        min_dist_ok = False
                        break
                if min_dist_ok:
                    break
        
        if min_dist_ok:
            positions = np.vstack([positions, new_pos])
            velocities = np.vstack([velocities, np.zeros(3)])
        
        # Relax the current cluster via Langevin dynamics
        # Temperature decreases as cluster grows
        progress = n_current / n_target
        T = T_start * (1 - progress) + T_end * progress
        
        n_steps = min(50, max(10, int(200 / (n_current + 1))))
        
        for step in range(n_steps):
            if time.time() - start_time > max_time:
                break
            
            positions, velocities = _langevin_step(
                positions, velocities, dt, T, damping, rng
            )
            
            # Keep velocities bounded
            v_max = np.max(np.linalg.norm(velocities, axis=1))
            if v_max > 5.0:
                velocities *= 5.0 / v_max
        
        # After relaxation, do a quick gradient-based polish
        try:
            def energy_func(x):
                return _compute_energy(x.reshape(-1, 3))
            
            def gradient_func(x):
                pos_reshaped = x.reshape(-1, 3)
                grad, _ = dzugutov_energy_gradient(pos_reshaped)
                return grad.flatten()
            
            x0 = positions.flatten()
            result = minimize(
                energy_func, x0,
                jac=gradient_func,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if result.fun < _compute_energy(positions):
                positions = result.x.reshape(-1, 3)
                velocities = np.zeros_like(positions)
        except Exception:
            pass
    
    # Final relaxation with lower temperature
    if time.time() - start_time < max_time:
        T_final = 0.02
        n_steps_final = min(100, int((max_time - (time.time() - start_time)) / 0.05))
        
        for step in range(n_steps_final):
            if time.time() - start_time > max_time - 2:
                break
            positions, velocities = _langevin_step(
                positions, velocities, dt, T_final, damping, rng
            )
            v_max = np.max(np.linalg.norm(velocities, axis=1))
            if v_max > 5.0:
                velocities *= 5.0 / v_max
        
        # Final gradient polish
        try:
            def energy_func(x):
                return _compute_energy(x.reshape(-1, 3))
            
            def gradient_func(x):
                pos_reshaped = x.reshape(-1, 3)
                grad, _ = dzugutov_energy_gradient(pos_reshaped)
                return grad.flatten()
            
            x0 = positions.flatten()
            result = minimize(
                energy_func, x0,
                jac=gradient_func,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result.fun < _compute_energy(positions):
                positions = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    # Ensure minimum distance constraint
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.5:
                diff = positions[i] - positions[j]
                if np.linalg.norm(diff) > 1e-10:
                    diff = diff / np.linalg.norm(diff) * 0.02
                else:
                    diff = np.array([0.02, 0, 0])
                positions[i] += diff
                positions[j] -= diff
    
    return positions


def _self_assemble_from_seed(n_target, seed_positions, seed=42, max_time=25.0):
    """
    Self-assembly starting from a given seed structure.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    positions = seed_positions.copy()
    velocities = np.zeros_like(positions)
    
    damping = 1.0
    dt = 0.005
    T_start = 0.3
    T_end = 0.03
    
    n_steps = min(200, int((max_time - (time.time() - start_time)) / 0.02))
    
    for step in range(n_steps):
        if time.time() - start_time > max_time - 5:
            break
        
        progress = step / max(1, n_steps)
        T = T_start * (1 - progress) + T_end * progress
        
        positions, velocities = _langevin_step(
            positions, velocities, dt, T, damping, rng
        )
        
        v_max = np.max(np.linalg.norm(velocities, axis=1))
        if v_max > 5.0:
            velocities *= 5.0 / v_max
    
    # Final gradient polish
    try:
        def energy_func(x):
            return _compute_energy(x.reshape(-1, 3))
        
        def gradient_func(x):
            pos_reshaped = x.reshape(-1, 3)
            grad, _ = dzugutov_energy_gradient(pos_reshaped)
            return grad.flatten()
        
        x0 = positions.flatten()
        result = minimize(
            energy_func, x0,
            jac=gradient_func,
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-10}
        )
        
        if result.fun < _compute_energy(positions):
            positions = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Ensure minimum distance constraint
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.5:
                diff = positions[i] - positions[j]
                if np.linalg.norm(diff) > 1e-10:
                    diff = diff / np.linalg.norm(diff) * 0.02
                else:
                    diff = np.array([0.02, 0, 0])
                positions[i] += diff
                positions[j] -= diff
    
    return positions


def _generate_random_seed(n, seed):
    """Generate a random initial configuration."""
    rng = np.random.default_rng(seed)
    radius = 1.5 * n ** (1.0 / 3.0)
    points = []
    min_dist = 0.8
    
    attempts = 0
    max_attempts = n * 200
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _relax_and_evaluate(pos, potential, parameter, timeout_per_relax=30.0):
    """Relax a structure using gradient-based optimization and return (energy, positions)."""
    try:
        start = time.time()
        
        # First try the built-in relax
        energy, relaxed_pos = relax(pos, potential, parameter)
        elapsed = time.time() - start
        
        # If time allows, try gradient-based refinement
        if time.time() - start < timeout_per_relax - 5:
            try:
                # Use L-BFGS-B for further refinement
                def energy_func(x):
                    positions_reshaped = x.reshape(-1, 3)
                    return _compute_energy(positions_reshaped)
                
                def gradient_func(x):
                    positions_reshaped = x.reshape(-1, 3)
                    grad, _ = dzugutov_energy_gradient(positions_reshaped)
                    return grad.flatten()
                
                x0 = relaxed_pos.flatten()
                result = minimize(
                    energy_func, x0,
                    jac=gradient_func,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                
                if result.fun < energy:
                    energy = result.fun
                    relaxed_pos = result.x.reshape(-1, 3)
            except Exception:
                pass
        
        return energy, relaxed_pos, time.time() - start
    except Exception as e:
        return np.inf, pos, 0.0


def _parallel_relax(candidates, potential, parameter, max_workers=4):
    """Relax multiple candidates in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_relax_and_evaluate, pos, potential, parameter): i
            for i, pos in enumerate(candidates)
        }
        for future in as_completed(futures):
            energy, pos, elapsed = future.result()
            results.append((energy, pos, elapsed))
    
    # Sort by energy
    results.sort(key=lambda x: x[0])
    return results


def _basin_hopping(pos, energy, potential, parameter, max_steps=10, timeout=60.0):
    """Perform basin hopping refinement with gradient-based local minimization."""
    rng = np.random.default_rng(int(time.time() * 1000) % 10000)
    current_energy = energy
    current_pos = pos.copy()
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > timeout:
            break
        
        sigma = 0.3 * np.exp(-step * 0.1)
        noise = rng.normal(0, sigma, size=current_pos.shape)
        perturbed_pos = current_pos + noise
        
        try:
            # Use gradient-based relaxation for the perturbed structure
            def energy_func(x):
                positions_reshaped = x.reshape(-1, 3)
                return _compute_energy(positions_reshaped)
            
            def gradient_func(x):
                positions_reshaped = x.reshape(-1, 3)
                grad, _ = dzugutov_energy_gradient(positions_reshaped)
                return grad.flatten()
            
            x0 = perturbed_pos.flatten()
            result = minimize(
                energy_func, x0,
                jac=gradient_func,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            
            if result.fun < current_energy:
                current_energy = result.fun
                current_pos = result.x.reshape(-1, 3)
        except Exception:
            continue
    
    return current_energy, current_pos


def construct(instance):
    """
    Find low-energy structures of Dzugutov clusters using physics-driven self-assembly
    with Langevin dynamics and multi-start optimization.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    # Strategy: Use physics-driven self-assembly as primary method
    # with multiple independent runs and parallel relaxation
    
    candidates = []
    
    # 1. Primary: Self-assembly from scratch with different seeds
    # Use 2 self-assembly runs with different seeds (inspired by finding that
    # multiple physics-driven seeds improve exploration)
    for seed in [42, 123]:
        try:
            remaining = deadline - time.time()
            if remaining < 20:
                break
            p_self = _self_assemble_cluster(size, seed=seed, max_time=min(40, remaining - 10))
            if len(p_self) == size:
                candidates.append(p_self)
        except Exception:
            pass
    
    # 2. Secondary: Self-assembly from random seeds
    for seed in [7, 99, 200]:
        try:
            remaining = deadline - time.time()
            if remaining < 15:
                break
            p_rand_seed = _generate_random_seed(size, seed)
            p_self = _self_assemble_from_seed(size, p_rand_seed, seed=seed + 1000, max_time=min(25, remaining - 10))
            if len(p_self) == size:
                candidates.append(p_self)
        except Exception:
            pass
    
    # 3. Tertiary: Pure random packings (fallback)
    for seed in [1, 7, 42]:
        try:
            remaining = deadline - time.time()
            if remaining < 10:
                break
            p_rand = _generate_random_seed(size, seed + 500)
            candidates.append(p_rand)
        except Exception:
            pass
    
    # Remove duplicates
    unique_candidates = []
    for c in candidates:
        if len(c) != size:
            continue
        is_dup = False
        for uc in unique_candidates:
            if len(c) == len(uc):
                diff = np.linalg.norm(c - uc, axis=1)
                if np.mean(diff) < 0.3:
                    is_dup = True
                    break
        if not is_dup:
            unique_candidates.append(c)
    
    # Parallel relaxation with gradient-based refinement
    if unique_candidates:
        results = _parallel_relax(unique_candidates, potential, parameter, max_workers=4)
    else:
        # Fallback: single random structure
        p_fallback = _generate_random_seed(size, 0)
        energy, p_fallback, _ = _relax_and_evaluate(p_fallback, potential, parameter)
        results = [(energy, p_fallback, 0)]
    
    best_energy = np.inf
    best_pos = None
    
    for energy, pos, elapsed in results:
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Basin hopping refinement on the best structure
    if best_pos is not None and time.time() < deadline - 30:
        remaining_time = deadline - time.time()
        max_bh_steps = max(10, int(remaining_time / 6))
        best_energy, best_pos = _basin_hopping(
            best_pos, best_energy, potential, parameter,
            max_steps=max_bh_steps, timeout=min(60, remaining_time)
        )
    
    # If we still have time, try basin hopping on the second best
    if len(results) > 1 and time.time() < deadline - 15:
        second_energy, second_pos, _ = results[1]
        if second_energy < np.inf:
            remaining = deadline - time.time()
            e2, p2 = _basin_hopping(
                second_pos, second_energy, potential, parameter,
                max_steps=8, timeout=min(30, remaining)
            )
            if e2 < best_energy:
                best_energy = e2
                best_pos = p2
    
    # Final check: ensure minimum distance constraint
    if best_pos is not None:
        for i in range(len(best_pos)):
            for j in range(i + 1, len(best_pos)):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < 0.5:
                    diff = best_pos[i] - best_pos[j]
                    if np.linalg.norm(diff) > 1e-10:
                        diff = diff / np.linalg.norm(diff) * 0.02
                    else:
                        diff = np.array([0.02, 0, 0])
                    best_pos[i] += diff
                    best_pos[j] -= diff
    
    if best_pos is None:
        # Fallback
        return np.array([np.array([i, 0, 0]) for i in range(size)])
    
    return best_pos
# EVOLVE-BLOCK-END
