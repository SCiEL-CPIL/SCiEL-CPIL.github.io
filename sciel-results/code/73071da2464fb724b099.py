"""Morse cluster global minima: Parallel basin hopping with diverse crystalline and icosahedral starts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280  # Safety margin for 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Compute all pair differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(pos)
    
    # Avoid division by zero
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    # dE/dr = exp_term * (2*exp_term - 2) * (-rho)
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
    # Gradient accumulation
    grad = np.zeros_like(pos)
    i_idx, j_idx = np.where(mask)
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_masked)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_masked)
    
    return energy, grad

def _is_valid(pos, min_dist=0.5):
    """Check all pairwise distances >= min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(r, np.inf)
    return bool(np.min(r) >= min_dist)

def _generate_fcc(n, spacing=1.0):
    """Generate FCC lattice points, clipped to sphere."""
    points = []
    a = spacing
    max_coord = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append(np.array([i * a, j * a, k * a]))
                points.append(np.array([(i + 0.5) * a, (j + 0.5) * a, k * a]))
                points.append(np.array([(i + 0.5) * a, j * a, (k + 0.5) * a]))
                points.append(np.array([i * a, (j + 0.5) * a, (k + 0.5) * a]))
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points

def _generate_bcc(n, spacing=1.0):
    """Generate BCC lattice points, clipped to sphere."""
    points = []
    a = spacing
    max_coord = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append(np.array([i * a, j * a, k * a]))
                points.append(np.array([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a]))
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points

def _generate_hcp(n, spacing=1.0):
    """Generate HCP lattice points, clipped to sphere."""
    points = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_coord = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = c * k
                points.append(np.array([x, y, z]))
                x_b = a * (i + 0.5 * (j % 2)) + a / 2.0
                y_b = a * np.sqrt(3.0) / 2.0 * j + a * np.sqrt(3.0) / 6.0
                z_b = c * (k + 0.5)
                points.append(np.array([x_b, y_b, z_b]))
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points

def _generate_icosahedral_shell(n, spacing=1.0):
    """Generate an icosahedral-like shell structure with proper icosahedral symmetry."""
    points = []
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedral vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])  # Normalize
    
    # Add center
    points.append([0.0, 0.0, 0.0])
    
    # Add icosahedral vertices
    for v in ico_verts[:12]:
        points.append(v * spacing)
    
    # If we need more atoms, add them on concentric shells
    remaining = n - len(points)
    if remaining > 0:
        # Second shell: midpoints of edges
        edge_midpoints = []
        for i in range(12):
            for j in range(i+1, 12):
                v1, v2 = ico_verts[i], ico_verts[j]
                d = np.linalg.norm(v1 - v2)
                if d < 1.5:  # Only add actual edge midpoints
                    mid = (v1 + v2) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints.append(mid)
        
        # Remove duplicates
        unique_mids = []
        for m in edge_midpoints:
            if not any(np.linalg.norm(m - um) < 0.1 for um in unique_mids):
                unique_mids.append(m)
        
        for m in unique_mids[:remaining]:
            points.append(m * spacing * 1.2)
        
        # If still need more, add on outer shell
        still_remaining = n - len(points)
        if still_remaining > 0:
            golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
            for i in range(still_remaining):
                z = 1.0 - 2.0 * (i + 0.5) / still_remaining
                radius_xy = np.sqrt(1.0 - z * z)
                theta = 2.0 * np.pi * i / golden_ratio
                x = radius_xy * np.cos(theta)
                y = radius_xy * np.sin(theta)
                points.append([x, y, z] * spacing * 1.5)
    
    return np.array(points[:n], dtype=float)

def _generate_greedy_close_packing(n, spacing=1.0, rng=None):
    """Generate a random close packing using greedy insertion."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Target a density that allows close packing without immediate overlaps
    # Equilibrium distance is 1.0. Min allowed is 0.5.
    # We want to start in a "compact" but not "collapsed" state.
    min_dist = 0.7 * spacing
    
    radius = spacing * (n ** (1.0/3.0)) * 1.5
    points = []
    
    max_attempts = n * 200
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
            
        if len(points) == 0:
            points.append(p)
        else:
            # Check distance to all existing points
            dists = np.sqrt(np.sum((np.array(points) - p) ** 2, axis=1))
            if np.min(dists) >= min_dist:
                points.append(p)
            else:
                attempts += 1
                
    # If we didn't get enough, relax the constraint or fill randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points[:n])

def _local_minimize(pos, rho, max_iter=500):
    """Local minimization using L-BFGS-B."""
    try:
        result = minimize(
            lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[0],
            pos.flatten(),
            jac=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        new_pos = result.x.reshape(-1, 3)
        new_energy = _morse_energy_and_grad(new_pos, rho)[0]
        if _is_valid(new_pos):
            return new_energy, new_pos
    except Exception:
        pass
    return _morse_energy_and_grad(pos, rho)[0], pos

def _relax_worker(args):
    """Worker function for parallel relaxation."""
    pos, rho, max_iter = args
    if _check_deadline():
        return None
    try:
        e, p = _local_minimize(pos, rho, max_iter)
        if _is_valid(p):
            return e, p
    except Exception:
        pass
    return None

def _basin_hop_metropolis(pos, rho, rng, max_steps=80, perturb_scale=0.3, T_start=1.0, T_end=0.1):
    """Basin hopping with Metropolis acceptance and annealing schedule."""
    current_energy, current_pos = _local_minimize(pos, rho, max_iter=200)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    step_size = perturb_scale
    no_improve_count = 0
    
    for step in range(max_steps):
        if _check_deadline():
            break
        
        # Annealing schedule: decrease temperature over time
        progress = step / max_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Adaptive step size: increase if stuck, decrease if improving
        if no_improve_count > 15:
            step_size *= 1.3
            no_improve_count = 0
        elif no_improve_count == 0:
            step_size *= 0.95
            step_size = max(step_size, 0.05)

        perturbation = rng.normal(0, step_size, pos.shape)
        perturbed = current_pos + perturbation
        perturbed_energy, perturbed_pos = _local_minimize(perturbed, rho, max_iter=100)
        
        if _is_valid(perturbed_pos):
            delta_e = perturbed_energy - current_energy
            # Metropolis acceptance: always accept downhill, accept uphill with probability exp(-delta_e/T)
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                # Accept
                current_energy = perturbed_energy
                current_pos = perturbed_pos
                no_improve_count = 0
                
                if perturbed_energy < best_energy:
                    best_energy = perturbed_energy
                    best_pos = perturbed_pos.copy()
            else:
                # Reject
                no_improve_count += 1
                
    return best_energy, best_pos

def construct(instance):
    """Find global minimum of Morse cluster using parallel basin hopping."""
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Crystalline
    try:
        fcc = _generate_fcc(size)
        if _is_valid(fcc):
            initial_configs.append(fcc)
    except Exception:
        pass
    
    try:
        bcc = _generate_bcc(size)
        if _is_valid(bcc):
            initial_configs.append(bcc)
    except Exception:
        pass
        
    try:
        hcp = _generate_hcp(size)
        if _is_valid(hcp):
            initial_configs.append(hcp)
    except Exception:
        pass
        
    # Icosahedral/Shell (improved)
    try:
        shell = _generate_icosahedral_shell(size)
        if _is_valid(shell):
            initial_configs.append(shell)
    except Exception:
        pass
    
    # Multiple icosahedral variants with different scales
    for scale in [0.8, 1.2]:
        try:
            ico_var = _generate_icosahedral_shell(size, spacing=scale)
            if _is_valid(ico_var) and not any(np.allclose(ico_var, c, atol=1e-6) for c in initial_configs):
                initial_configs.append(ico_var)
        except Exception:
            pass
        
    # Greedy Close Packing (Structural change: better initial structures)
    for _ in range(6):
        try:
            rand = _generate_greedy_close_packing(size, rng=rng)
            if _is_valid(rand):
                initial_configs.append(rand)
        except Exception:
            pass
            
    if not initial_configs:
        pos = rng.uniform(-2, 2, (size, 3))
        initial_configs.append(pos)
        
    # Parallel relaxation
    best_energy = np.inf
    best_pos = None
    
    # Use 4 threads as allowed
    max_workers = min(4, len(initial_configs))
    
    if max_workers > 1 and len(initial_configs) > 1:
        tasks = [(config, rho, 300) for config in initial_configs]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(_relax_worker, tasks)
            for res in results:
                if res is not None:
                    e, p = res
                    if e < best_energy:
                        best_energy = e
                        best_pos = p.copy()
    else:
        # Fallback to sequential if parallelization fails or is not beneficial
        for config in initial_configs:
            if _check_deadline():
                break
            try:
                e, p = _local_minimize(config, rho, max_iter=300)
                if _is_valid(p) and e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            except Exception:
                pass

    if best_pos is None:
        best_pos = initial_configs[0]
        best_energy = _morse_energy_and_grad(best_pos, rho)[0]
        
    # Basin hopping on the best result if time permits (Structural change: Metropolis BH with annealing)
    if time.time() < _DEADLINE - 10:
        try:
            e_bh, pos_bh = _basin_hop_metropolis(best_pos, rho, rng, max_steps=80, perturb_scale=0.3, T_start=1.0, T_end=0.1)
            if e_bh < best_energy and _is_valid(pos_bh):
                best_energy = e_bh
                best_pos = pos_bh
        except Exception:
            pass
        
        # Additional basin hopping from second-best if time permits
        if time.time() < _DEADLINE - 15 and len(initial_configs) > 1:
            try:
                # Find second-best initial config
                second_best_pos = None
                second_best_e = np.inf
                for config in initial_configs:
                    e_cfg, _ = _local_minimize(config, rho, max_iter=100)
                    if e_cfg < second_best_e and not np.allclose(config, best_pos, atol=1e-6):
                        second_best_e = e_cfg
                        second_best_pos = config
                
                if second_best_pos is not None:
                    e_bh2, pos_bh2 = _basin_hop_metropolis(second_best_pos, rho, rng, max_steps=40, perturb_scale=0.4, T_start=1.5, T_end=0.1)
                    if e_bh2 < best_energy and _is_valid(pos_bh2):
                        best_energy = e_bh2
                        best_pos = pos_bh2
            except Exception:
                pass
            
    # Final refinement
    if time.time() < _DEADLINE - 2:
        try:
            e_final, pos_final = _local_minimize(best_pos, rho, max_iter=500)
            if e_final < best_energy and _is_valid(pos_final):
                best_energy = e_final
                best_pos = pos_final
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
