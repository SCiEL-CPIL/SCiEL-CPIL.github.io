"""Two-phase hierarchical optimization for SC cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=290.0):
    return deadline - (time.time() - start_time)


def _fcc_template(n, spacing=1.0):
    """Generate n points from FCC lattice, centered."""
    if n == 0:
        return np.zeros((0, 3))
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 2.5)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.3) ** 2
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (ix + bx) * spacing
                    y = (iy + by) * spacing
                    z = (iz + bz) * spacing
                    if x*x + y*y + z*z <= radius_sq:
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


def _hcp_template(n, spacing=1.0):
    """Generate n points from HCP lattice, centered."""
    if n == 0:
        return np.zeros((0, 3))
    points = []
    a = spacing
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3) * 2.5)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.3) ** 2
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                if iz % 2 == 0:
                    x = ix * a + 0.25 * iy * a
                    y = iy * a * np.sqrt(3)/2
                else:
                    x = ix * a + 0.5*a + 0.25 * iy * a
                    y = iy * a * np.sqrt(3)/2
                z = iz * c / 2
                if x*x + y*y + z*z <= radius_sq:
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


def _bcc_template(n, spacing=1.0):
    """Generate n points from BCC lattice, centered."""
    if n == 0:
        return np.zeros((0, 3))
    points = []
    a = spacing * 0.9
    max_coord = int(np.ceil(n ** (1/3) * 2.5)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.3) ** 2
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for bx, by, bz in [(0,0,0), (0.5,0.5,0.5)]:
                    x = (ix + bx) * a
                    y = (iy + by) * a
                    z = (iz + bz) * a
                    if x*x + y*y + z*z <= radius_sq:
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


def _icosahedral_template(n, spacing=1.0):
    """Generate points based on icosahedral symmetry."""
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


def _random_packed(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 1.0
    min_dist = 0.55 * spacing
    points = []
    attempts = 0
    max_attempts = n * 400
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


def _enforce_min_dist(positions, min_dist=0.35):
    """Ensure no two atoms are closer than min_dist."""
    n = len(positions)
    for _ in range(100):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        mask = dists < min_dist
        if not np.any(mask):
            break
        i, j = np.where(mask)
        i, j = i[0], j[0]
        d = positions[i] - positions[j]
        dist = np.linalg.norm(d)
        if dist < 1e-10:
            d = np.array([1.0, 0.0, 0.0])
        else:
            d = d / dist
        push = (min_dist - dist) / 2.0 + 0.01
        positions[i] += push * d
        positions[j] -= push * d
    return positions


def _sc_energy_and_grad(x_flat, kind):
    """Energy and gradient for scipy minimize from flat array."""
    pos = x_flat.reshape(-1, 3)
    e, g = sutton_chen_energy_gradient(pos, kind)
    return e, g.flatten()


def _polish(positions, kind, start_time, deadline=290.0, maxiter=200):
    """Polish a structure using L-BFGS-B."""
    if _time_left(start_time, deadline) < 5.0:
        e, _ = sutton_chen_energy_gradient(positions, kind)
        return e, positions
    try:
        res = minimize(_sc_energy_and_grad, positions.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-10,
                                                     'maxfun': maxiter * 20})
        if np.isfinite(res.fun):
            return res.fun, res.x.reshape(-1, 3)
        return sutton_chen_energy_gradient(positions, kind)
    except Exception:
        try:
            return relax(positions, 'sutton_chen', kind)
        except Exception:
            return sutton_chen_energy_gradient(positions, kind)


def _compute_voronoi_descriptor(positions):
    """
    Compute a coarse-grained descriptor based on local environment statistics.
    This serves as a proxy for structural similarity to known global minima.
    """
    n = len(positions)
    if n < 2:
        return np.array([0.0, 0.0, 0.0])
    
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # For each atom, find nearest neighbors (within a reasonable cutoff)
    cutoff = 1.5 * np.mean(dists[np.isfinite(dists)])
    
    coord_numbers = []
    local_env_variances = []
    
    for i in range(n):
        neighbors = dists[i] < cutoff
        num_neighbors = np.sum(neighbors)
        coord_numbers.append(num_neighbors)
        
        if num_neighbors > 1:
            local_dists = dists[i][neighbors]
            local_env_variances.append(np.var(local_dists))
        else:
            local_env_variances.append(0.0)
    
    coord_numbers = np.array(coord_numbers)
    local_env_variances = np.array(local_env_variances)
    
    # Descriptor: mean coordination number, variance of coordination numbers, mean local env variance
    descriptor = np.array([
        np.mean(coord_numbers),
        np.std(coord_numbers),
        np.mean(local_env_variances)
    ])
    
    return descriptor


def _coarse_energy(positions, kind, target_descriptor):
    """
    Coarse-grained energy: SC energy + descriptor mismatch penalty.
    """
    e, _ = sutton_chen_energy_gradient(positions, kind)
    if not np.isfinite(e):
        return 1e10
    
    descriptor = _compute_voronoi_descriptor(positions)
    mismatch = np.sum((descriptor - target_descriptor) ** 2)
    
    # Weight the descriptor penalty
    penalty_weight = 0.1
    return e + penalty_weight * mismatch


def _coarse_energy_grad(x_flat, kind, target_descriptor):
    """Coarse energy with gradient (approximate gradient for descriptor part)."""
    pos = x_flat.reshape(-1, 3)
    e, g = sutton_chen_energy_gradient(pos, kind)
    if not np.isfinite(e):
        return 1e10, np.zeros_like(x_flat)
    
    descriptor = _compute_voronoi_descriptor(pos)
    mismatch = np.sum((descriptor - target_descriptor) ** 2)
    
    # Approximate gradient of descriptor mismatch (finite difference)
    eps = 1e-4
    grad_descriptor = np.zeros_like(x_flat)
    for i in range(len(x_flat)):
        x_plus = x_flat.copy()
        x_plus[i] += eps
        pos_plus = x_plus.reshape(-1, 3)
        desc_plus = _compute_voronoi_descriptor(pos_plus)
        mismatch_plus = np.sum((desc_plus - target_descriptor) ** 2)
        grad_descriptor[i] = (mismatch_plus - mismatch) / eps
    
    total_energy = e + 0.1 * mismatch
    total_grad = g.flatten() + 0.1 * grad_descriptor
    
    return total_energy, total_grad


def _coarse_optimize(positions, kind, target_descriptor, start_time, deadline=290.0, maxiter=100):
    """
    Phase 1: Optimize using coarse-grained energy with descriptor matching.
    """
    if _time_left(start_time, deadline) < 10.0:
        e, _ = sutton_chen_energy_gradient(positions, kind)
        return e, positions
    
    try:
        res = minimize(_coarse_energy_grad, positions.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-8,
                                                     'maxfun': maxiter * 15})
        if np.isfinite(res.fun):
            # Return the actual SC energy, not the coarse energy
            e_actual, _ = sutton_chen_energy_gradient(res.x.reshape(-1, 3), kind)
            return e_actual, res.x.reshape(-1, 3)
        return sutton_chen_energy_gradient(positions, kind)
    except Exception:
        try:
            return relax(positions, 'sutton_chen', kind)
        except Exception:
            return sutton_chen_energy_gradient(positions, kind)


def _basin_hop(positions, kind, start_time, deadline, rng, 
               max_hops=200, step_size=0.5, cooling_rate=0.999):
    """
    Basin hopping with adaptive step size.
    """
    current_energy, current_pos = sutton_chen_energy_gradient(positions, kind)
    best_energy, best_pos = current_energy, positions.copy()
    
    for hop in range(max_hops):
        if _time_left(start_time, deadline) < 20.0:
            break
        
        # Perturb the structure
        displacement = rng.normal(0, step_size, (len(current_pos), 3))
        new_pos = current_pos + displacement
        
        # Polish the perturbed structure
        try:
            polished_energy, polished_pos = _polish(new_pos, kind, start_time, deadline, maxiter=50)
        except Exception:
            continue
        
        if not np.isfinite(polished_energy):
            continue
        
        # Accept if lower energy
        if polished_energy < current_energy:
            current_energy = polished_energy
            current_pos = polished_pos.copy()
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished_pos.copy()
        else:
            # Occasionally accept higher energy to escape local minima
            if rng.random() < 0.1:
                current_energy = polished_energy
                current_pos = polished_pos.copy()
        
        # Adapt step size
        step_size *= cooling_rate
    
    return best_energy, best_pos


def _parallel_search(starters, kind, start_time, deadline, max_workers=4, seed_offset=0):
    """Run parallel basin hopping from multiple starting structures."""
    results = []
    
    def _bh_one(args):
        idx, starter = args
        try:
            rng = np.random.default_rng(seed_offset + idx * 1000)
            energy, positions = _basin_hop(starter, kind, start_time, deadline, rng)
            if np.isfinite(energy):
                return (energy, positions.copy())
        except Exception:
            pass
        return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_bh_one, (i, s)): i for i, s in enumerate(starters)}
        for future in as_completed(futures):
            if _time_left(start_time, deadline) < 25.0:
                break
            try:
                result = future.result(timeout=5.0)
                if result is not None:
                    results.append(result)
            except Exception:
                continue
    
    return results


def construct(instance):
    """
    Two-phase hierarchical optimization for SC cluster global minima.
    
    Phase 1: Coarse-grained optimization using Voronoi descriptor matching
    Phase 2: High-precision basin hopping from descriptor-optimal configurations
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = 290.0
    
    # Generate diverse starting structures
    starters = [
        _fcc_template(size, spacing=1.0),
        _hcp_template(size, spacing=1.0),
        _bcc_template(size, spacing=1.0),
        _icosahedral_template(size, spacing=1.0),
        _random_packed(size, spacing=1.0, seed=size),
        _random_packed(size, spacing=1.0, seed=size + 7),
        _random_packed(size, spacing=1.0, seed=size + 13),
        _random_packed(size, spacing=1.0, seed=size + 23),
        _random_packed(size, spacing=1.0, seed=size + 37),
        _random_packed(size, spacing=1.0, seed=size + 41),
    ]
    
    # Phase 0: Initial polish of all starters to get good local minima
    polished_starters = []
    for i, starter in enumerate(starters):
        if _time_left(start_time, deadline) < 30.0:
            break
        try:
            e, pos = _polish(starter, kind, start_time, deadline, maxiter=30)
            if np.isfinite(e):
                polished_starters.append((e, pos.copy()))
        except Exception:
            pass
    
    if not polished_starters:
        # Emergency fallback
        best_positions = _random_packed(size, spacing=1.0, seed=size)
        return best_positions
    
    # Sort by energy
    polished_starters.sort(key=lambda x: x[0])
    
    # Phase 1: Coarse-grained optimization using descriptor matching
    # Use the best polished structure to define the target descriptor
    target_descriptor = _compute_voronoi_descriptor(polished_starters[0][1])
    
    coarse_results = []
    for i, (e, pos) in enumerate(polished_starters[:5]):  # Only top 5 to save time
        if _time_left(start_time, deadline) < 40.0:
            break
        try:
            coarse_e, coarse_pos = _coarse_optimize(pos, kind, target_descriptor, 
                                                     start_time, deadline, maxiter=80)
            if np.isfinite(coarse_e):
                coarse_results.append((coarse_e, coarse_pos.copy()))
        except Exception:
            pass
    
    # Combine all results (polished starters + coarse results)
    all_results = polished_starters + coarse_results
    all_results.sort(key=lambda x: x[0])
    
    # Phase 2: Basin hopping from the best structures
    # Use top 4 structures for parallel basin hopping
    bh_starters = [pos for _, pos in all_results[:4]]
    
    bh_results = _parallel_search(bh_starters, kind, start_time, deadline, 
                                   max_workers=4, seed_offset=1000)
    
    if not bh_results:
        # Fallback to best from phase 1
        best_energy, best_positions = all_results[0]
    else:
        bh_results.sort(key=lambda x: x[0])
        best_energy, best_positions = bh_results[0]
    
    # Phase 3: Final polish
    if _time_left(start_time, deadline) > 15.0:
        try:
            final_energy, final_pos = _polish(best_positions, kind, start_time, deadline, maxiter=200)
            if np.isfinite(final_energy) and final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos.copy()
        except Exception:
            pass
    
    # Phase 4: One more basin hop from the best if time allows
    if _time_left(start_time, deadline) > 50.0:
        try:
            rng_final = np.random.default_rng(size + 999)
            bh2_energy, bh2_pos = _basin_hop(best_positions, kind, start_time, deadline, 
                                              rng_final, max_hops=100, step_size=0.3)
            if np.isfinite(bh2_energy) and bh2_energy < best_energy:
                best_energy = bh2_energy
                best_positions = bh2_pos.copy()
                if _time_left(start_time, deadline) > 10.0:
                    final2_energy, final2_pos = _polish(best_positions, kind, start_time, 
                                                         deadline, maxiter=150)
                    if np.isfinite(final2_energy) and final2_energy < best_energy:
                        best_energy = final2_energy
                        best_positions = final2_pos.copy()
        except Exception:
            pass
    
    # Enforce minimum distance
    best_positions = _enforce_min_dist(best_positions, min_dist=0.35)
    
    # Final validity check
    if not np.all(np.isfinite(best_positions)):
        best_positions = _random_packed(size, spacing=1.0, seed=size + 999)
        best_positions = _enforce_min_dist(best_positions, min_dist=0.35)
    
    return best_positions
# EVOLVE-BLOCK-END
