"""Simulated annealing + diverse templates for SC cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=285.0):
    return deadline - (time.time() - start_time)


def _fcc_template(n, spacing=1.0):
    """Generate n points from FCC lattice, centered."""
    if n == 0:
        return np.zeros((0, 3))
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 2.0)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.2) ** 2
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
    max_coord = int(np.ceil(n ** (1/3) * 2.0)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.2) ** 2
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
    max_coord = int(np.ceil(n ** (1/3) * 2.0)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.2) ** 2
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
    radius = spacing * n ** (1.0 / 3.0) * 0.95
    min_dist = 0.55 * spacing
    points = []
    attempts = 0
    max_attempts = n * 300
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


def _polish(positions, kind, start_time, deadline=285.0, maxiter=200):
    """Polish a structure using L-BFGS-B."""
    if _time_left(start_time, deadline) < 5.0:
        e, _ = sutton_chen_energy_gradient(positions, kind)
        return e, positions
    try:
        res = minimize(_sc_energy_and_grad, positions.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-8,
                                                     'maxfun': maxiter * 15})
        if np.isfinite(res.fun):
            return res.fun, res.x.reshape(-1, 3)
        return sutton_chen_energy_gradient(positions, kind)
    except Exception:
        try:
            return relax(positions, 'sutton_chen', kind)
        except Exception:
            return sutton_chen_energy_gradient(positions, kind)


def _simulated_annealing(positions, kind, start_time, deadline, rng, 
                         T_init=2.0, T_min=0.01, alpha=0.995, max_steps=5000):
    """
    Simulated annealing for SC cluster optimization.
    Uses Metropolis acceptance to escape local minima.
    """
    current_energy, current_pos = sutton_chen_energy_gradient(positions, kind)
    best_energy, best_pos = current_energy, positions.copy()
    T = T_init
    
    for step in range(max_steps):
        if _time_left(start_time, deadline) < 15.0:
            break
        
        # Perturb: move one atom by a random displacement
        atom_idx = rng.integers(0, len(current_pos))
        displacement = rng.normal(0, 0.3, 3)
        new_pos = current_pos.copy()
        new_pos[atom_idx] += displacement
        
        # Compute new energy
        try:
            new_energy, _ = sutton_chen_energy_gradient(new_pos, kind)
        except Exception:
            continue
        
        if not np.isfinite(new_energy):
            continue
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > T_min and rng.random() < np.exp(-delta_e / T)):
            current_energy = new_energy
            current_pos = new_pos.copy()
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        # Cool down
        T = max(T_min, T * alpha)
    
    return best_energy, best_pos


def _parallel_sa(starters, kind, start_time, deadline, max_workers=4, seed_offset=0):
    """Run simulated annealing in parallel from multiple starting structures."""
    results = []
    
    def _sa_one(args):
        idx, starter = args
        try:
            rng = np.random.default_rng(seed_offset + idx * 1000)
            energy, positions = _simulated_annealing(starter, kind, start_time, deadline, rng)
            return (energy, positions.copy())
        except Exception:
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_sa_one, (i, s)): i for i, s in enumerate(starters)}
        for future in as_completed(futures):
            if _time_left(start_time, deadline) < 20.0:
                break
            result = future.result()
            if result is not None and np.isfinite(result[0]):
                results.append(result)
    
    return results


def construct(instance):
    """
    Simulated annealing + diverse templates for SC cluster global minima.
    Uses SA with Metropolis acceptance to escape local minima,
    combined with diverse structural templates and parallel execution.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = 285.0
    
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
    ]
    
    # Phase 1: Parallel simulated annealing from all starters
    sa_results = _parallel_sa(starters, kind, start_time, deadline, max_workers=4, seed_offset=0)
    
    if not sa_results:
        # Emergency fallback
        best_positions = _random_packed(size, spacing=1.0, seed=size)
        return best_positions
    
    # Sort by energy
    sa_results.sort(key=lambda x: x[0])
    best_energy, best_positions = sa_results[0]
    
    # Phase 2: Polish the best SA result with L-BFGS-B
    if _time_left(start_time, deadline) > 20.0:
        try:
            polished_energy, polished_pos = _polish(best_positions, kind, start_time, deadline, maxiter=150)
            if np.isfinite(polished_energy) and polished_energy < best_energy:
                best_energy = polished_energy
                best_positions = polished_pos.copy()
        except Exception:
            pass
    
    # Phase 3: Run SA again from the polished best with different parameters
    if _time_left(start_time, deadline) > 40.0:
        try:
            rng2 = np.random.default_rng(size + 500)
            sa2_energy, sa2_pos = _simulated_annealing(
                best_positions, kind, start_time, deadline, rng2,
                T_init=1.0, T_min=0.005, alpha=0.997, max_steps=3000
            )
            if np.isfinite(sa2_energy) and sa2_energy < best_energy:
                best_energy = sa2_energy
                best_positions = sa2_pos.copy()
                # Polish again
                if _time_left(start_time, deadline) > 15.0:
                    polished2_energy, polished2_pos = _polish(best_positions, kind, start_time, deadline, maxiter=100)
                    if np.isfinite(polished2_energy) and polished2_energy < best_energy:
                        best_energy = polished2_energy
                        best_positions = polished2_pos.copy()
        except Exception:
            pass
    
    # Phase 4: Run SA from second-best SA result
    if _time_left(start_time, deadline) > 30.0 and len(sa_results) > 1:
        try:
            rng3 = np.random.default_rng(size + 600)
            sa3_energy, sa3_pos = _simulated_annealing(
                sa_results[1][1], kind, start_time, deadline, rng3,
                T_init=1.5, T_min=0.01, alpha=0.995, max_steps=3000
            )
            if np.isfinite(sa3_energy) and sa3_energy < best_energy:
                best_energy = sa3_energy
                best_positions = sa3_pos.copy()
                if _time_left(start_time, deadline) > 10.0:
                    polished3_energy, polished3_pos = _polish(best_positions, kind, start_time, deadline, maxiter=100)
                    if np.isfinite(polished3_energy) and polished3_energy < best_energy:
                        best_energy = polished3_energy
                        best_positions = polished3_pos.copy()
        except Exception:
            pass
    
    # Phase 5: Final polish
    if _time_left(start_time, deadline) > 10.0:
        try:
            final_energy, final_pos = _polish(best_positions, kind, start_time, deadline, maxiter=200)
            if np.isfinite(final_energy) and final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos.copy()
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
