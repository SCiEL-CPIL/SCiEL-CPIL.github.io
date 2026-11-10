# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 290s safety margin before 330s kill

def _remaining():
    return _DEADLINE - time.time()

def _get_params(kind):
    """Get Sutton-Chen parameters."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41

def _get_spacing(kind):
    """Calculate approximate equilibrium bond length."""
    n_exp, m_exp, c = _get_params(kind)
    return (2 * n_exp / (c * m_exp)) ** (1.0 / (n_exp - m_exp / 2.0))

def _sc_incremental_energy(new_pos, existing_pos, kind):
    """
    Compute the incremental energy contribution of adding new_pos to existing cluster.
    Returns (pair_energy, cohen_energy, total_energy) for the new atom.
    """
    n_exp, m_exp, c = _get_params(kind)
    if len(existing_pos) == 0:
        return 0.0, 0.0, 0.0
    
    dists = np.linalg.norm(existing_pos - new_pos, axis=1)
    # Avoid division by zero
    dists = np.maximum(dists, 1e-10)
    
    r_n = np.sum(dists ** (-n_exp))
    r_m = np.sum(dists ** (-m_exp))
    
    pair_e = 0.5 * r_n
    cohen_e = -c * np.sqrt(r_m)
    
    return pair_e, cohen_e, pair_e + cohen_e

def _coarse_to_fine_grid_insertion(n, spacing, kind, seed=0):
    """
    Coarse-to-fine grid search for cluster growth.
    Start with a small core, then insert atoms one by one using gradient-guided
    position search on a coarse grid, refining as we go.
    """
    rng = np.random.default_rng(seed)
    n_exp, m_exp, c = _get_params(kind)
    
    # Start with 2 atoms at equilibrium distance
    positions = np.array([[0.0, 0.0, 0.0], [spacing, 0.0, 0.0]])
    
    # For the first few atoms, use a coarse grid to find good positions
    for count in range(2, n):
        if _remaining() < 30.0:
            break
        
        # Determine grid resolution based on cluster size
        # Coarser for larger clusters (we're adding to a bigger structure)
        if count < 6:
            n_grid = 12
            grid_step = spacing * 0.3
        elif count < 12:
            n_grid = 10
            grid_step = spacing * 0.35
        else:
            n_grid = 8
            grid_step = spacing * 0.4
        
        # Center of mass of existing cluster
        com = np.mean(positions, axis=0)
        
        # Search radius: slightly larger than current cluster radius
        cluster_radius = np.max(np.linalg.norm(positions - com, axis=1))
        search_radius = cluster_radius + spacing * 1.5
        
        # Coarse grid search for best insertion position
        best_e = np.inf
        best_pos = None
        
        # Create grid points in a sphere
        for ix in range(-n_grid, n_grid + 1):
            for iy in range(-n_grid, n_grid + 1):
                for iz in range(-n_grid, n_grid + 1):
                    candidate = com + np.array([ix, iy, iz]) * grid_step
                    if np.linalg.norm(candidate - com) > search_radius:
                        continue
                    
                    # Quick check: minimum distance to existing atoms
                    min_d = np.min(np.linalg.norm(positions - candidate, axis=1))
                    if min_d < spacing * 0.4:
                        continue
                    
                    # Compute incremental energy
                    pe, ce, total = _sc_incremental_energy(candidate, positions, kind)
                    
                    # Add a small penalty for being too close to COM (favor compact clusters)
                    com_penalty = 0.01 * np.linalg.norm(candidate - com)
                    
                    if total + com_penalty < best_e:
                        best_e = total + com_penalty
                        best_pos = candidate
        
        if best_pos is None:
            # Fallback: random position
            angle1 = rng.uniform(0, 2 * np.pi)
            angle2 = rng.uniform(0, np.pi)
            r = spacing * rng.uniform(0.8, 1.5)
            best_pos = com + np.array([
                r * np.sin(angle2) * np.cos(angle1),
                r * np.sin(angle2) * np.sin(angle1),
                r * np.cos(angle2)
            ])
        
        # Refine with local gradient descent on the incremental energy
        def _obj_and_grad(x):
            pe, ce, total = _sc_incremental_energy(x, positions, kind)
            # Numerical gradient
            eps = 1e-5
            grad = np.zeros(3)
            for d in range(3):
                xp = x.copy()
                xp[d] += eps
                xn = x.copy()
                xn[d] -= eps
                grad[d] = (_sc_incremental_energy(xp, positions, kind)[2] - 
                          _sc_incremental_energy(xn, positions, kind)[2]) / (2 * eps)
            return total, grad
        
        # Local refinement using scipy
        try:
            res = minimize(_obj_and_grad, best_pos, method='L-BFGS-B', 
                         jac=True, options={'maxiter': 50, 'gtol': 1e-4})
            if res.success and np.all(np.isfinite(res.x)):
                # Check minimum distance
                min_d = np.min(np.linalg.norm(positions - res.x, axis=1))
                if min_d > spacing * 0.35:
                    best_pos = res.x
        except Exception:
            pass
        
        positions = np.vstack([positions, best_pos])
    
    return positions

def _gradient_guided_growth(n, spacing, kind, seed=0):
    """
    Alternative growth strategy: use full cluster energy gradient to guide insertion.
    """
    rng = np.random.default_rng(seed)
    positions = np.array([[0.0, 0.0, 0.0], [spacing, 0.0, 0.0]])
    
    for count in range(2, n):
        if _remaining() < 30.0:
            break
        
        com = np.mean(positions, axis=0)
        cluster_radius = np.max(np.linalg.norm(positions - com, axis=1))
        
        # Try multiple candidate positions and pick the one with lowest incremental energy
        candidates = []
        
        # Positions near existing atoms (face-centered, edge-centered, etc.)
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if j >= len(positions):
                    break
                mid = (positions[i] + positions[j]) / 2.0
                d = np.linalg.norm(positions[j] - positions[i])
                if d > spacing * 0.5 and d < spacing * 2.0:
                    # Face center candidate
                    normal = np.cross(positions[j] - positions[i], positions[0] - positions[i])
                    if np.linalg.norm(normal) > 1e-10:
                        normal = normal / np.linalg.norm(normal)
                        candidates.append(mid + normal * spacing * 0.3)
                        candidates.append(mid - normal * spacing * 0.3)
        
        # Random candidates
        for _ in range(8):
            r = rng.uniform(spacing * 0.7, cluster_radius + spacing)
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            candidates.append(com + np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ]))
        
        # Pick best by incremental energy
        best_e = np.inf
        best_pos = None
        for cand in candidates:
            min_d = np.min(np.linalg.norm(positions - cand, axis=1))
            if min_d < spacing * 0.4:
                continue
            pe, ce, total = _sc_incremental_energy(cand, positions, kind)
            if total < best_e:
                best_e = total
                best_pos = cand
        
        if best_pos is None:
            r = spacing * rng.uniform(0.8, 1.2)
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            best_pos = com + np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
        
        positions = np.vstack([positions, best_pos])
    
    return positions

def _fcc_positions(n, a=1.0):
    """Generate n atoms in FCC lattice positions (truncated)."""
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    dist = np.linalg.norm(pos)
                    sites.append((dist, pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _hcp_positions(n, a=1.0):
    """Generate n atoms in HCP lattice positions (truncated)."""
    c = a * np.sqrt(8/3)
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for layer in range(-max_idx, max_idx + 1):
        z = layer * c / 2
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([i * a, j * a * np.sqrt(3)/2, z])
                sites.append((np.linalg.norm(pos), pos))
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([(i + 0.5) * a, (j + 1/6) * a * np.sqrt(3), z + c/2])
                sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _random_cluster(n, spacing, seed, min_dist=None):
    """Random rejection-sampling cluster."""
    if min_dist is None:
        min_dist = spacing * 0.9
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    max_attempts = 5000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _molten_cluster(n, spacing, seed, temp_scale=1.0):
    """Generate a 'molten' cluster: random positions with soft repulsion."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 1.2
    points = []
    min_d = spacing * 0.5 * temp_scale
    max_attempts = 10000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_d for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _min_distance_check(pos):
    """Check minimum distance between any two atoms."""
    n = len(pos)
    if n < 2:
        return np.inf
    dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _valid_result(energy, pos):
    """Check if a relaxation result is valid."""
    if not np.all(np.isfinite(pos)):
        return False
    if _min_distance_check(pos) < 0.35:
        return False
    return True

def _relax_task(args):
    """Task for parallel execution: relax a single candidate."""
    pos, potential, kind = args
    try:
        energy, relaxed_pos = relax(pos, potential, kind)
        if _valid_result(energy, relaxed_pos):
            return energy, relaxed_pos
        else:
            return np.inf, relaxed_pos
    except Exception:
        return np.inf, pos

def construct(instance):
    global _START_TIME, _DEADLINE
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    spacing = _get_spacing(kind)
    
    # Generate diverse starting points
    candidates = []
    
    # 1. Coarse-to-fine grid growth (primary strategy)
    try:
        grown = _coarse_to_fine_grid_insertion(n, spacing, kind, seed=0)
        if len(grown) == n:
            candidates.append(grown)
    except Exception:
        pass
    
    # 2. Gradient-guided growth
    try:
        grown2 = _gradient_guided_growth(n, spacing, kind, seed=1)
        if len(grown2) == n:
            candidates.append(grown2)
    except Exception:
        pass
    
    # 3. FCC
    try:
        fcc = _fcc_positions(n, spacing)
        if len(fcc) == n:
            candidates.append(fcc)
    except Exception:
        pass
    
    # 4. HCP
    try:
        hcp = _hcp_positions(n, spacing)
        if len(hcp) == n:
            candidates.append(hcp)
    except Exception:
        pass
    
    # 5-7. Perturbed FCC/HCP
    for seed, scale in [(1, 0.08), (2, 0.15), (3, 0.25)]:
        try:
            base = _fcc_positions(n, spacing) if seed % 2 == 1 else _hcp_positions(n, spacing)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, spacing * scale, base.shape)
            candidates.append(base + noise)
        except Exception:
            pass
    
    # 8-9. Molten clusters
    for seed, ts in [(100, 1.5), (200, 0.8)]:
        try:
            molten = _molten_cluster(n, spacing, seed=seed, temp_scale=ts)
            candidates.append(molten)
        except Exception:
            pass
    
    # 10-12. Random clusters
    for seed in [42, 123, 456]:
        try:
            rand = _random_cluster(n, spacing, seed)
            candidates.append(rand)
        except Exception:
            pass
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Parallel Relaxation of Initial Candidates
    if _remaining() > 20.0:
        tasks = [(pos, potential, kind) for pos in candidates]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_task, task): task for task in tasks}
            for future in as_completed(futures):
                if _remaining() < 5.0:
                    break
                try:
                    energy, pos = future.result()
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = pos
                except Exception:
                    pass

    # If no valid result yet, try sequential fallback
    if best_positions is None:
        for pos in candidates:
            if _remaining() < 10.0:
                break
            try:
                energy, relaxed_pos = relax(pos, potential, kind)
                if _valid_result(energy, relaxed_pos) and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                pass

    # Phase 2: Basin-hopping with decaying kicks
    if best_positions is not None and _remaining() > 30.0:
        rng = np.random.default_rng(999)
        iteration = 0
        no_improve_count = 0
        while _remaining() > 40.0:
            iteration += 1
            base_kick = spacing * 0.6 * (0.75 ** iteration)
            kicks = [base_kick * 0.4, base_kick * 0.7, base_kick, 
                     base_kick * 1.3, base_kick * 1.6, base_kick * 2.0]
            
            tasks = []
            for kick in kicks:
                noise = rng.normal(0, kick, best_positions.shape)
                perturbed = best_positions + noise
                tasks.append((perturbed, potential, kind))
                
            improved_this_round = False
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_relax_task, task): task for task in tasks}
                for future in as_completed(futures):
                    if _remaining() < 10.0:
                        break
                    try:
                        energy, pos = future.result()
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = pos
                            improved_this_round = True
                    except Exception:
                        pass
            
            if not improved_this_round:
                no_improve_count += 1
                if no_improve_count >= 4:
                    break
            else:
                no_improve_count = 0
        
        # Swap-like moves
        if best_positions is not None and _remaining() > 30.0:
            rng2 = np.random.default_rng(555)
            for swap_round in range(3):
                if _remaining() < 20.0:
                    break
                tasks = []
                for _ in range(4):
                    perturbed = best_positions.copy()
                    idx1 = rng2.choice(n, size=min(4, n//2), replace=False)
                    idx2 = rng2.choice(n, size=min(4, n//2), replace=False)
                    for a1, a2 in zip(idx1, idx2):
                        mid = (perturbed[a1] + perturbed[a2]) / 2.0
                        delta = rng2.normal(0, spacing * 0.3, 3)
                        perturbed[a1] = mid + delta
                        perturbed[a2] = mid - delta
                    tasks.append((perturbed, potential, kind))
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(_relax_task, task): task for task in tasks}
                    for future in as_completed(futures):
                        if _remaining() < 10.0:
                            break
                        try:
                            energy, pos = future.result()
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = pos
                        except Exception:
                            pass
    
    # Phase 3: Additional diversification
    if best_positions is not None and _remaining() > 40.0:
        rng2 = np.random.default_rng(2024)
        for kick_scale in [0.3, 0.5, 0.8]:
            if _remaining() < 20.0:
                break
            noise = rng2.normal(0, spacing * kick_scale, best_positions.shape)
            perturbed = best_positions + noise
            try:
                energy, pos = _relax_task((perturbed, potential, kind))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
        
        # Try additional diverse starts
        if _remaining() > 25.0:
            for extra_seed in [7777, 8888]:
                if _remaining() < 15.0:
                    break
                try:
                    extra = _molten_cluster(n, spacing, seed=extra_seed, temp_scale=1.2)
                    energy, pos = _relax_task((extra, potential, kind))
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = pos
                except Exception:
                    pass

    # Phase 4: Final polish
    if best_positions is not None and _remaining() > 10.0:
        try:
            energy, relaxed_pos = relax(best_positions, potential, kind)
            if _valid_result(energy, relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        try:
            pos = _random_cluster(n, spacing, 12345, min_dist=0.4)
            energy, relaxed_pos = relax(pos, potential, kind)
            if _valid_result(energy, relaxed_pos):
                best_positions = relaxed_pos
            else:
                best_positions = np.zeros((n, 3))
                for i in range(n):
                    best_positions[i] = [i * 0.8, 0, 0]
        except Exception:
            best_positions = np.zeros((n, 3))
            for i in range(n):
                best_positions[i] = [i * 0.8, 0, 0]
            
    return best_positions
# EVOLVE-BLOCK-END
