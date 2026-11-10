"""Morse cluster global minima: stochastic collision dynamics + gradient refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# Deadline guard: evaluator kills at 330s, use 300s as hard limit with safety margin
TOTAL_DEADLINE = 295.0
START_TIME = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient for a cluster.
    
    V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    """
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 1e-10, d, 1.0)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = -rho * ex * (ex - 1)
    dvdr = -rho * ex * (ex - 1.0)
    np.fill_diagonal(dvdr, 0.0)
    
    # grad_i = sum_j dvdr_ij * (r_i - r_j) / r_ij
    # r_ij vector = positions[i] - positions[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_safe = np.where(r > 1e-10, r, 1.0)
    r_safe[np.diag_indices(n)] = 1.0
    
    # Multiply dvdr by unit vector (r_i - r_j)/r_ij
    grad = np.einsum('ij,ijk->ik', dvdr / r_safe, diff)
    
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute Morse energy only (for validation)."""
    e, _ = _morse_energy_and_grad(positions, rho)
    return e


def _is_valid(positions, min_dist=0.5):
    """Check if all pair distances are at least min_dist."""
    d = pdist(positions)
    return np.all(d >= min_dist - 1e-6)


def _generate_random_cluster(n, rng, scale=3.0):
    """Generate a random cluster with minimum distance constraint."""
    positions = np.zeros((n, 3))
    for i in range(n):
        for _ in range(100):
            pos = rng.uniform(-scale, scale, 3)
            if i == 0:
                positions[i] = pos
                break
            # Check minimum distance
            dists = np.linalg.norm(positions[:i] - pos, axis=1)
            if np.all(dists >= 0.5):
                positions[i] = pos
                break
        else:
            # Fallback: place on a spiral
            t = i * 0.5
            positions[i] = [t * np.cos(t), t * np.sin(t), t * 0.1]
    # Center
    positions -= positions.mean(axis=0)
    return positions


def _generate_crystalline(n, lattice='fcc'):
    """Generate crystalline cluster."""
    if lattice == 'fcc':
        a = 1.0
        basis = [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ]
    elif lattice == 'bcc':
        a = 1.0
        basis = [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
        ]
    else:  # hcp
        a = 1.0
        c = a * np.sqrt(8.0 / 3.0)
        n_layers = int(np.ceil((n / 6.0) ** (1.0 / 3.0))) + 2
        positions = []
        for layer in range(-n_layers, n_layers + 1):
            z = layer * c / 2.0
            if layer % 2 == 0:
                for i in range(-n_layers, n_layers + 1):
                    for j in range(-n_layers, n_layers + 1):
                        x = i * a + (j % 2) * a / 2.0
                        y = j * a * np.sqrt(3.0) / 2.0
                        positions.append([x, y, z])
            else:
                for i in range(-n_layers, n_layers + 1):
                    for j in range(-n_layers, n_layers + 1):
                        x = i * a + a / 2.0
                        y = j * a * np.sqrt(3.0) / 2.0 + a * np.sqrt(3.0) / 6.0
                        positions.append([x, y, z])
        positions = np.array(positions)
        positions -= positions.mean(axis=0)
        dists = np.linalg.norm(positions, axis=1)
        order = np.argsort(dists)
        return positions[order][:n]
    
    L = int(np.ceil((n / 4.0) ** (1.0 / 3.0))) + 1 if lattice == 'fcc' else int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 1
    candidates = []
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + np.array(b) * a
                    candidates.append(p)
    candidates = np.array(candidates)
    candidates -= candidates.mean(axis=0)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    return candidates[order][:n]


def _collision_dynamics(positions, rho, n_steps=500, dt=0.01, temperature=0.5, rng=None):
    """
    Stochastic collision dynamics: Velocity Verlet with thermostatting and random kicks.
    This allows the cluster to escape local minima and find lower-energy configurations.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = positions.shape[0]
    pos = positions.copy()
    vel = np.zeros_like(pos)
    
    # Initial energy
    e0, _ = _morse_energy_and_grad(pos, rho)
    
    # Velocity Verlet integration with thermostatting
    for step in range(n_steps):
        # Compute forces (negative gradient)
        _, grad = _morse_energy_and_grad(pos, rho)
        forces = -grad
        
        # Velocity Verlet: update velocity half-step
        vel += 0.5 * dt * forces
        # Random kicks (Langevin thermostatting)
        vel += rng.normal(0, np.sqrt(temperature * dt), pos.shape)
        # Damping
        vel *= 0.99
        # Update positions
        pos += dt * vel
        # Second half-step velocity update
        _, grad2 = _morse_energy_and_grad(pos, rho)
        forces2 = -grad2
        vel += 0.5 * dt * forces2
        vel *= 0.99
        
        # Check validity and repair if needed
        if not _is_valid(pos, min_dist=0.45):
            # Push apart overlapping atoms
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < 0.5:
                        if d < 1e-10:
                            direction = rng.normal(size=3)
                            direction /= np.linalg.norm(direction)
                        else:
                            direction = (pos[i] - pos[j]) / d
                        push = 0.01 * direction
                        pos[i] += push
                        pos[j] -= push
    
    return pos


def _relax_with_gradient(positions, rho, time_limit=10.0):
    """Relax positions using L-BFGS-B with exact gradients."""
    start = time.time()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g
    
    try:
        result = minimize(
            fun,
            positions.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 10000, 'gtol': 1e-8}
        )
        if result.success or result.status in [1, 2, 3]:
            new_pos = result.x.reshape(-1, 3)
            if _is_valid(new_pos):
                return _morse_energy_only(new_pos, rho), new_pos
    except Exception:
        pass
    
    # Fallback: try with fewer iterations if time is running out
    elapsed = time.time() - start
    if elapsed < time_limit:
        try:
            result = minimize(
                fun,
                positions.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000}
            )
            new_pos = result.x.reshape(-1, 3)
            if _is_valid(new_pos):
                return _morse_energy_only(new_pos, rho), new_pos
        except Exception:
            pass
    
    return _morse_energy_only(positions, rho), positions


def construct(instance):
    """Find global minimum of Morse cluster using stochastic collision dynamics + gradient refinement."""
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    # Check remaining time
    elapsed = time.time() - START_TIME
    remaining = TOTAL_DEADLINE - elapsed
    if remaining < 5.0:
        # Not enough time, return best quick result
        pos = _generate_crystalline(size, 'bcc')
        return pos
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Per-instance time budget
    per_instance_budget = 40.0
    instance_start = time.time()
    
    # Strategy 1: Start from crystalline templates + collision dynamics + gradient refinement
    for lattice in ['fcc', 'bcc', 'hcp']:
        if time.time() - instance_start > per_instance_budget:
            break
        
        try:
            pos = _generate_crystalline(size, lattice)
            if len(pos) != size:
                continue
            
            # Apply collision dynamics to escape local minima
            time_left = per_instance_budget - (time.time() - instance_start)
            if time_left < 5.0:
                break
            
            n_steps = min(300, int(time_left * 50))  # Scale steps with available time
            pos_cd = _collision_dynamics(pos, rho, n_steps=n_steps, dt=0.01, temperature=0.3, rng=rng)
            
            # Gradient refinement
            energy, relaxed_pos = _relax_with_gradient(pos_cd, rho, time_limit=min(time_left - 2.0, 15.0))
            
            if energy < best_energy and _is_valid(relaxed_pos):
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # Strategy 2: Random starts + collision dynamics + gradient refinement
    n_random_starts = 5
    for i in range(n_random_starts):
        if time.time() - instance_start > per_instance_budget:
            break
        
        try:
            pos = _generate_random_cluster(size, rng, scale=4.0)
            if not _is_valid(pos):
                continue
            
            time_left = per_instance_budget - (time.time() - instance_start)
            if time_left < 5.0:
                break
            
            n_steps = min(200, int(time_left * 40))
            pos_cd = _collision_dynamics(pos, rho, n_steps=n_steps, dt=0.01, temperature=0.5, rng=rng)
            
            energy, relaxed_pos = _relax_with_gradient(pos_cd, rho, time_limit=min(time_left - 2.0, 10.0))
            
            if energy < best_energy and _is_valid(relaxed_pos):
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # Strategy 3: If time remains, perturb best solution and re-relax
    instance_elapsed = time.time() - instance_start
    if best_positions is not None and instance_elapsed < per_instance_budget * 0.6:
        for _ in range(3):
            if time.time() - instance_start > per_instance_budget:
                break
            perturbed = best_positions + rng.normal(0, 0.15, best_positions.shape)
            if _is_valid(perturbed):
                time_left = per_instance_budget - (time.time() - instance_start)
                if time_left < 3.0:
                    break
                n_steps = min(100, int(time_left * 30))
                pos_cd = _collision_dynamics(perturbed, rho, n_steps=n_steps, dt=0.01, temperature=0.2, rng=rng)
                energy, relaxed_pos = _relax_with_gradient(pos_cd, rho, time_limit=min(time_left - 1.0, 8.0))
                if energy < best_energy and _is_valid(relaxed_pos):
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
    
    if best_positions is None:
        # Fallback: return a valid configuration
        best_positions = _generate_crystalline(size, 'bcc')
        if not _is_valid(best_positions):
            best_positions = _generate_random_cluster(size, rng, scale=5.0)
    
    return best_positions
