"""Structural-bias-guided basin hopping for SC cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=290.0):
    return deadline - (time.time() - start_time)


def _fcc_shells(n, spacing=1.0):
    """Generate n points from FCC lattice shells, centered."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 2)) + 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (ix + bx) * spacing
                    y = (iy + by) * spacing
                    z = (iz + bz) * spacing
                    if x*x + y*y + z*z <= (n ** (1/3) * spacing * 1.2) ** 2:
                        points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _hcp_shells(n, spacing=1.0):
    """Generate n points from HCP lattice shells."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    a = spacing
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3) * 2)) + 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                if iz % 2 == 0:
                    x = ix * a + 0.5 * iy * a * 0.5
                    y = iy * a * np.sqrt(3)/2
                else:
                    x = ix * a + 0.5 * a + 0.5 * iy * a * 0.5
                    y = iy * a * np.sqrt(3)/2
                z = iz * c / 2
                if x*x + y*y + z*z <= (n ** (1/3) * spacing * 1.2) ** 2:
                    points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _icosahedron_based(n, spacing=1.0):
    """Generate points based on icosahedral symmetry with nested shells."""
    phi = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices[0]) * spacing
    
    points = [[0.0, 0.0, 0.0]]
    points.extend([v.tolist() for v in vertices])
    
    if n > 13:
        dodec = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
            [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
            [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
        ])
        dodec = dodec / np.linalg.norm(dodec[0]) * spacing * 1.8
        for v in dodec:
            if len(points) >= n:
                break
            points.append(v.tolist())
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        base = points[rng.integers(0, len(points))]
        p = np.array(base) + rng.normal(0, 0.3 * spacing, 3)
        points.append(p.tolist())
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.5 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 200
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _perturb(positions, sigma=0.1, seed=None):
    """Add small random perturbations to positions."""
    rng = np.random.default_rng(seed)
    return positions + rng.normal(0, sigma, positions.shape)


def _check_validity(positions, min_dist=0.35):
    """Check if positions are valid (finite and no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def _fcc_local_score(positions):
    """
    Coarse-grained structural descriptor: penalize non-FCC-like local environments.
    
    For each atom, compute the coordination number and the variance of neighbor
    distances. FCC atoms have 12 nearest neighbors at roughly equal distances.
    A low variance and coordination near 12 indicates an FCC-like environment.
    """
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    total_penalty = 0.0
    for i in range(n):
        # Get distances from atom i to all others, sorted
        d = np.sort(dists[i])
        
        # Find the "first shell" - use a gap-based approach
        # The first shell ends where there's a significant jump
        if len(d) < 2:
            total_penalty += 1.0
            continue
        
        # Estimate first shell size: look for a gap > 1.5x the median of first few
        # Simple heuristic: take neighbors within 1.3 * median of first 12 (or all if fewer)
        ref = d[:min(12, len(d))]
        median_ref = np.median(ref) if len(ref) > 0 else 1.0
        cutoff = 1.4 * median_ref if median_ref > 0 else 1.0
        
        shell_mask = d < cutoff
        n_shell = np.sum(shell_mask)
        
        # Penalty for coordination number deviating from 12
        coord_penalty = abs(n_shell - 12) / 12.0
        
        # Penalty for variance in shell distances (FCC has uniform shell)
        if n_shell >= 2:
            shell_dists = d[shell_mask]
            var_penalty = np.var(shell_dists) / (np.mean(shell_dists)**2 + 1e-10)
        else:
            var_penalty = 1.0
        
        # Combine: want low coord_penalty and low var_penalty
        total_penalty += coord_penalty + var_penalty
    
    return total_penalty / n


def _combined_objective(x, kind, structural_weight=0.0):
    """
    Combined objective: SC energy + structural bias penalty.
    Returns (energy, gradient) for use with scipy minimize.
    """
    pos = x.reshape(-1, 3)
    e, g = sutton_chen_energy_gradient(pos, kind)
    
    if structural_weight > 0:
        # Approximate gradient of structural penalty via finite differences
        # This is expensive but only used in a limited number of iterations
        eps = 1e-4
        g_struct = np.zeros_like(x)
        for k in range(len(x)):
            x_plus = x.copy()
            x_plus[k] += eps
            _, g_plus = sutton_chen_energy_gradient(x_plus.reshape(-1, 3), kind)
            # We don't easily get structural gradient, so skip it for now
            # Instead, we'll use the structural score only for acceptance criteria
            break
        # For simplicity, just return the energy gradient and use structural
        # score separately in the basin hopping acceptance
        pass
    
    return e, g.flatten()


def _bfgs_refine(positions, potential, kind, start_time, deadline=290.0):
    """Refine positions using BFGS with the SC energy gradient."""
    if _time_left(start_time, deadline) < 5.0:
        return positions, sutton_chen_energy_gradient(positions, kind)[0]
    
    def objective_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = sutton_chen_energy_gradient(pos, kind)
        return e, g.flatten()
    
    try:
        res = minimize(objective_and_grad, positions.flatten(), jac=True, 
                      method='BFGS', options={'maxiter': 500, 'gtol': 1e-8})
        if res.success or res.fun < sutton_chen_energy_gradient(positions, kind)[0]:
            return res.x.reshape(-1, 3), res.fun
    except Exception:
        pass
    
    return positions, sutton_chen_energy_gradient(positions, kind)[0]


def _basin_hopping_structural(positions, potential, kind, start_time, deadline=290.0, 
                               n_iter=30, step_size=0.3, structural_weight=0.1):
    """
    Basin-hopping with structural bias in acceptance criterion.
    
    Accept a move if:
    1. Energy decreased, OR
    2. Energy increased but structural score improved significantly (FCC-like), OR
    3. Metropolis acceptance based on energy.
    
    The structural bias guides the search toward FCC-like configurations.
    """
    best_energy = sutton_chen_energy_gradient(positions, kind)[0]
    best_pos = positions.copy()
    best_struct = _fcc_local_score(positions)
    
    current_energy = best_energy
    current_pos = positions.copy()
    current_struct = best_struct
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if _time_left(start_time, deadline) < 10.0:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        try:
            energy, relaxed_pos = relax(perturbed, potential, kind)
        except Exception:
            continue
        
        if not np.isfinite(energy):
            continue
        
        # Compute structural score
        struct_score = _fcc_local_score(relaxed_pos)
        
        # Acceptance criteria with structural bias
        accept = False
        if energy < current_energy:
            accept = True
        elif struct_score < current_struct - structural_weight:
            # Structural improvement is significant enough to accept even with energy increase
            accept = True
        else:
            # Standard Metropolis
            T = max(0.1, 1.0 * (1 - i / n_iter))
            if rng.random() < np.exp(-(energy - current_energy) / T):
                accept = True
        
        if accept:
            current_energy = energy
            current_pos = relaxed_pos
            current_struct = struct_score
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
                best_struct = struct_score
    
    return best_energy, best_pos


def construct(instance):
    """Multi-start relaxation with structural-bias-guided basin hopping for SC clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = 290.0
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting structures
    starters = []
    
    # FCC-like (most important for SC clusters)
    starters.append(_fcc_shells(size, spacing=1.0))
    
    # HCP-like
    starters.append(_hcp_shells(size, spacing=1.0))
    
    # Icosahedron-based
    starters.append(_icosahedron_based(size, spacing=1.0))
    
    # Multiple random compact structures with different seeds
    for seed in [size, size + 1, size + 2, size + 3, size + 4, size + 5]:
        starters.append(_random_compact(size, spacing=1.0, seed=seed))
    
    # Relax each starter with L-BFGS-B, then refine with BFGS
    relaxed_results = []
    for i, starter in enumerate(starters):
        if _time_left(start_time, deadline) < 15.0:
            break
        
        try:
            # First pass: L-BFGS-B
            energy, positions = relax(starter, potential, kind)
            
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
            
            relaxed_results.append((energy, positions))
            
            # Second pass: BFGS refinement if we have time
            if _time_left(start_time, deadline) > 20.0:
                refined_pos, refined_e = _bfgs_refine(positions, potential, kind, start_time, deadline)
                if np.isfinite(refined_e) and refined_e < best_energy:
                    best_energy = refined_e
                    best_positions = refined_pos.copy()
                if np.isfinite(refined_e):
                    relaxed_results[-1] = (refined_e, refined_pos)
        except Exception:
            continue
    
    # Structural-bias-guided basin hopping from the best solution
    if best_positions is not None:
        # Basin-hopping from best solution with structural bias
        if _time_left(start_time, deadline) > 30.0:
            try:
                bh_energy, bh_positions = _basin_hopping_structural(
                    best_positions, potential, kind, start_time, deadline,
                    n_iter=30, step_size=0.25, structural_weight=0.1
                )
                if np.isfinite(bh_energy) and bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions.copy()
            except Exception:
                pass
        
        # Try perturbing the best solution and refining
        for attempt in range(4):
            if _time_left(start_time, deadline) < 20.0:
                break
            
            perturbed = _perturb(best_positions, sigma=0.05, seed=size + 100 + attempt)
            try:
                energy, positions = relax(perturbed, potential, kind)
                
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
                
                # BFGS refinement
                if _time_left(start_time, deadline) > 25.0:
                    refined_pos, refined_e = _bfgs_refine(positions, potential, kind, start_time, deadline)
                    if np.isfinite(refined_e) and refined_e < best_energy:
                        best_energy = refined_e
                        best_positions = refined_pos.copy()
            except Exception:
                continue
        
        # Basin-hopping from second-best starter if time remains
        if _time_left(start_time, deadline) > 40.0 and len(starters) > 1:
            try:
                # Use the second-best relaxed result if available
                second_best = None
                for e, p in relaxed_results:
                    if np.isfinite(e) and e < best_energy + 0.5:
                        second_best = p
                        break
                if second_best is None:
                    second_best = starters[1]
                
                bh_energy, bh_positions = _basin_hopping_structural(
                    second_best, potential, kind, start_time, deadline,
                    n_iter=25, step_size=0.3, structural_weight=0.1
                )
                if np.isfinite(bh_energy) and bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions.copy()
            except Exception:
                pass
        
        # Try additional random starts
        for attempt in range(5):
            if _time_left(start_time, deadline) < 15.0:
                break
            
            starter = _random_compact(size, spacing=1.0, seed=size + 200 + attempt)
            try:
                energy, positions = relax(starter, potential, kind)
                
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
                
                # BFGS refinement
                if _time_left(start_time, deadline) > 25.0:
                    refined_pos, refined_e = _bfgs_refine(positions, potential, kind, start_time, deadline)
                    if np.isfinite(refined_e) and refined_e < best_energy:
                        best_energy = refined_e
                        best_positions = refined_pos.copy()
            except Exception:
                continue
    
    # Fallback: return a simple valid configuration if nothing worked
    if best_positions is None:
        best_positions = _random_compact(size, spacing=1.0, seed=size)
    
    # Final validity check
    if not _check_validity(best_positions):
        # Try to fix by spreading out atoms
        rng = np.random.default_rng(size + 500)
        for _ in range(10):
            fixed = best_positions + rng.normal(0, 0.1, best_positions.shape)
            if _check_validity(fixed):
                best_positions = fixed
                break
    
    return best_positions
# EVOLVE-BLOCK-END
