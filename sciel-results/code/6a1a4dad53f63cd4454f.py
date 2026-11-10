"""Morse cluster global minima: hybrid crystalline + stochastic dynamics + L-BFGS-B refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# Deadline guard
TOTAL_DEADLINE = 290.0
START_TIME = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient for a cluster.
    
    V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    """
    n = positions.shape[0]
    # Vectorized pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 1e-10, d, 1.0)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = -rho * ex * (ex - 1)
    dvdr = -rho * ex * (ex - 1.0)
    np.fill_diagonal(dvdr, 0.0)
    
    r_safe = np.where(r > 1e-10, r, 1.0)
    r_safe[np.diag_indices(n)] = 1.0
    
    # grad_i = sum_j dvdr_ij * (r_i - r_j) / r_ij
    # Use broadcasting: dvdr[i,j] / r[i,j] * diff[i,j,:]
    coeff = dvdr / r_safe  # n x n
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute Morse energy only (for validation)."""
    e, _ = _morse_energy_and_grad(positions, rho)
    return e


def _is_valid(positions, min_dist=0.5):
    """Check if all pair distances are at least min_dist."""
    d = pdist(positions)
    return np.all(d >= min_dist - 1e-6)


def _repair_min_dist(positions, min_dist=0.5, rng=None, max_iter=50):
    """Repair positions that violate minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng()
    n = positions.shape[0]
    pos = positions.copy().astype(float)
    for _ in range(max_iter):
        d = squareform(pdist(pos))
        np.fill_diagonal(d, 1.0)
        # Find all violating pairs
        violating = np.argwhere(d < min_dist - 1e-6)
        if len(violating) == 0:
            break
        for i, j in violating:
            if i >= j:
                continue
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                diff = rng.normal(0, 0.01, 3)
                dist = np.linalg.norm(diff)
            direction = diff / dist
            push = (min_dist - d[i, j] + 0.01) / 2.0
            pos[i] += push * direction
            pos[j] -= push * direction
    return pos


def _generate_fcc(n):
    """Generate FCC-like cluster of n atoms."""
    a = 1.0
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    L = int(np.ceil((n / 4.0) ** (1.0 / 3.0))) + 1
    candidates = []
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + np.array(b) * a
                    candidates.append(p)
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    candidates -= center
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    positions = candidates[:n]
    return positions


def _generate_bcc(n):
    """Generate BCC-like cluster of n atoms."""
    a = 1.0
    basis = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ]
    L = int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 1
    candidates = []
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + np.array(b) * a
                    candidates.append(p)
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    candidates -= center
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    positions = candidates[:n]
    return positions


def _generate_hcp(n):
    """Generate HCP-like cluster of n atoms."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    positions = []
    n_layers = int(np.ceil((n / 6.0) ** (1.0 / 3.0))) + 2
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
    center = positions.mean(axis=0)
    positions -= center
    dists = np.linalg.norm(positions, axis=1)
    order = np.argsort(dists)
    positions = positions[order]
    return positions[:n]


def _generate_shell(n):
    """Generate shell-like cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    positions = [np.zeros(3)]
    remaining = n - 1
    
    shell_idx = 0
    while remaining > 0:
        shell_idx += 1
        if shell_idx == 1:
            count = min(12, remaining)
        elif shell_idx == 2:
            count = min(42, remaining)
        else:
            count = min(int(4 * shell_idx * shell_idx), remaining)
        
        r = shell_idx * 0.55
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(count):
            t = i / count
            z = 1.0 - 2.0 * t
            radius_xy = np.sqrt(1.0 - z * z)
            theta_i = golden_angle * i
            x = radius_xy * np.cos(theta_i)
            y = radius_xy * np.sin(theta_i)
            positions.append(r * np.array([x, y, z]))
        
        remaining -= count
    
    return np.array(positions[:n])


def _generate_icosahedral(n):
    """Generate icosahedral-like cluster (targets short-range ground state symmetry)."""
    if n <= 1:
        return np.zeros((n, 3))
    
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize to unit sphere
    ico_vertices /= np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    positions = [np.zeros(3)]  # center
    remaining = n - 1
    
    # First shell: icosahedron (12 atoms)
    r1 = 1.0
    count1 = min(12, remaining)
    for i in range(count1):
        positions.append(r1 * ico_vertices[i])
    remaining -= count1
    
    # Second shell: truncated icosahedron / fullerene-like (60 positions, use 42 or fewer)
    if remaining > 0:
        r2 = 1.8
        # Generate points on sphere using Fibonacci spiral
        count2 = min(60, remaining)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(count2):
            t = (i + 0.5) / count2
            z = 1.0 - 2.0 * t
            radius_xy = np.sqrt(max(0, 1.0 - z * z))
            theta = golden_angle * i
            x = radius_xy * np.cos(theta)
            y = radius_xy * np.sin(theta)
            positions.append(r2 * np.array([x, y, z]))
        remaining -= count2
    
    # Third shell if needed
    if remaining > 0:
        r3 = 2.6
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(remaining):
            t = (i + 0.5) / remaining
            z = 1.0 - 2.0 * t
            radius_xy = np.sqrt(max(0, 1.0 - z * z))
            theta = golden_angle * i
            x = radius_xy * np.cos(theta)
            y = radius_xy * np.sin(theta)
            positions.append(r3 * np.array([x, y, z]))
    
    return np.array(positions[:n])


def _generate_random_compact(n, rng, min_dist=0.6):
    """Generate a random compact cluster with minimum distance constraint."""
    radius = 1.0 * n ** (1.0 / 3.0) * 0.8
    positions = []
    max_attempts = 10000
    attempts = 0
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        valid = True
        for q in positions:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            positions.append(p)
    
    if len(positions) < n:
        # Fallback: use grid-based placement
        positions = []
        step = min_dist * 1.1
        for i in range(-10, 11):
            for j in range(-10, 11):
                for k in range(-10, 11):
                    p = np.array([i, j, k]) * step
                    if len(positions) < n:
                        positions.append(p)
    
    return np.array(positions[:n])


def _velocity_verlet_anneal(positions, rho, n_steps=500, dt=0.01, 
                              T_start=2.0, T_end=0.05, rng=None, time_limit=15.0):
    """Velocity Verlet with linear annealing for self-assembly."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = positions.shape[0]
    pos = positions.copy().astype(float)
    vel = rng.normal(0, 0.3, pos.shape)
    
    start_time = time.time()
    
    for step in range(n_steps):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Current temperature (geometric cooling for better exploration)
        frac = step / max(n_steps - 1, 1)
        T = T_start * (T_end / T_start) ** frac
        
        # Compute forces (negative gradient)
        e, grad = _morse_energy_and_grad(pos, rho)
        forces = -grad
        
        # Velocity Verlet half-step
        vel += 0.5 * dt * forces
        pos += dt * vel
        
        # Thermostat: scale velocities to target temperature
        target_ke = 1.5 * n * T
        current_ke = 0.5 * np.sum(vel * vel)
        if current_ke > 1e-10:
            scale = np.sqrt(target_ke / current_ke)
            vel *= scale
        
        # Second half-step
        e2, grad2 = _morse_energy_and_grad(pos, rho)
        forces2 = -grad2
        vel += 0.5 * dt * forces2
        
        # Ensure minimum distance constraint (vectorized repair)
        if not _is_valid(pos, min_dist=0.5):
            pos = _repair_min_dist(pos, min_dist=0.5, rng=rng, max_iter=10)
    
    return pos


def _relax_with_gradient(positions, rho, max_iter=500, time_limit=10.0):
    """Relax positions using L-BFGS-B with exact gradients."""
    start = time.time()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    try:
        result = minimize(
            fun,
            positions.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 20, 'gtol': 1e-8, 'factr': 1e7}
        )
        new_pos = result.x.reshape(-1, 3)
        if _is_valid(new_pos):
            return _morse_energy_only(new_pos, rho), new_pos
    except Exception:
        pass
    
    # Fallback: try with fewer iterations
    elapsed = time.time() - start
    if elapsed < time_limit:
        try:
            result = minimize(
                fun,
                positions.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000, 'gtol': 1e-6}
            )
            new_pos = result.x.reshape(-1, 3)
            if _is_valid(new_pos):
                return _morse_energy_only(new_pos, rho), new_pos
        except Exception:
            pass
    
    return _morse_energy_only(positions, rho), positions


def construct(instance):
    """Find global minimum of Morse cluster using hybrid approach."""
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
        pos = _generate_bcc(size)
        return pos
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Per-instance time budget: use remaining time divided by instances not yet done
    # We don't know how many instances remain, so use a fraction of remaining
    per_instance_budget = min(remaining * 0.95, 48.0)
    instance_start = time.time()
    
    # Phase 1: Try crystalline generators with gradient relaxation
    # Include icosahedral for short-range (small rho) clusters
    generators = [_generate_icosahedral, _generate_fcc, _generate_bcc, _generate_hcp, _generate_shell]
    
    # For small clusters (n <= 50), icosahedral is most promising
    # For large clusters, FCC/BCC are more relevant
    if size <= 50:
        generators = [_generate_icosahedral, _generate_shell, _generate_fcc, _generate_bcc]
    
    for gen in generators:
        if time.time() - instance_start > per_instance_budget * 0.45:
            break
        
        try:
            pos = gen(size)
            if len(pos) != size:
                continue
            if not _is_valid(pos):
                pos = _repair_min_dist(pos, min_dist=0.5, rng=rng)
                if not _is_valid(pos):
                    continue
            
            time_left = per_instance_budget * 0.45 - (time.time() - instance_start)
            if time_left < 2.0:
                break
            
            energy, relaxed_pos = _relax_with_gradient(pos, rho, max_iter=400, time_limit=min(time_left - 1.0, 12.0))
            
            if energy < best_energy and _is_valid(relaxed_pos):
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # Phase 2: Stochastic dynamics (velocity Verlet annealing) from multiple starts
    if time.time() - instance_start < per_instance_budget * 0.75:
        # Start from best found so far, or from random compact if none
        if best_positions is not None:
            starts_for_dynamics = [best_positions.copy()]
        else:
            starts_for_dynamics = [_generate_random_compact(size, rng)]
        
        # Add diverse random compact starts
        num_random_starts = 3 if size <= 50 else 2
        for _ in range(num_random_starts):
            if time.time() - instance_start > per_instance_budget * 0.75:
                break
            rand_pos = _generate_random_compact(size, rng)
            if _is_valid(rand_pos):
                starts_for_dynamics.append(rand_pos)
        
        # Also add perturbed versions of best
        if best_positions is not None:
            for _ in range(2):
                if time.time() - instance_start > per_instance_budget * 0.75:
                    break
                perturbed = best_positions + rng.normal(0, 0.2, best_positions.shape)
                if _is_valid(perturbed):
                    starts_for_dynamics.append(perturbed)
        
        for start_pos in starts_for_dynamics:
            if time.time() - instance_start > per_instance_budget * 0.75:
                break
            
            try:
                time_left = per_instance_budget * 0.75 - (time.time() - instance_start)
                if time_left < 3.0:
                    break
                
                # Run velocity Verlet annealing
                annealed_pos = _velocity_verlet_anneal(
                    start_pos, rho, 
                    n_steps=400, dt=0.008,
                    T_start=1.8, T_end=0.03,
                    rng=rng,
                    time_limit=min(time_left - 2.0, 10.0)
                )
                
                if not _is_valid(annealed_pos):
                    annealed_pos = _repair_min_dist(annealed_pos, min_dist=0.5, rng=rng)
                    if not _is_valid(annealed_pos):
                        continue
                
                # Refine with gradient-based relaxation
                energy, relaxed_pos = _relax_with_gradient(
                    annealed_pos, rho, 
                    max_iter=300, 
                    time_limit=min(time_left - 3.0, 10.0)
                )
                
                if energy < best_energy and _is_valid(relaxed_pos):
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
                    
            except Exception:
                continue
    
    # Phase 3: If time remains, try local perturbations and re-relaxation
    if best_positions is not None and time.time() - instance_start < per_instance_budget:
        num_perturbations = 4
        for _ in range(num_perturbations):
            if time.time() - instance_start > per_instance_budget:
                break
            
            try:
                # Varying perturbation scales
                scale = 0.1 + 0.1 * _
                perturbed = best_positions + rng.normal(0, scale, best_positions.shape)
                if not _is_valid(perturbed):
                    perturbed = _repair_min_dist(perturbed, min_dist=0.5, rng=rng)
                    if not _is_valid(perturbed):
                        continue
                
                time_left = per_instance_budget - (time.time() - instance_start)
                if time_left < 2.0:
                    break
                
                energy, relaxed_pos = _relax_with_gradient(
                    perturbed, rho,
                    max_iter=200,
                    time_limit=min(time_left - 1.0, 10.0)
                )
                
                if energy < best_energy and _is_valid(relaxed_pos):
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
                    
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: return a valid configuration
        best_positions = _generate_bcc(size)
        if not _is_valid(best_positions):
            best_positions = _generate_random_compact(size, rng)
        if not _is_valid(best_positions):
            best_positions = _repair_min_dist(best_positions, min_dist=0.5, rng=rng)
        if not _is_valid(best_positions):
            best_positions = np.zeros((size, 3))
            for i in range(size):
                best_positions[i] = rng.uniform(-2, 2, 3)
    
    return best_positions
