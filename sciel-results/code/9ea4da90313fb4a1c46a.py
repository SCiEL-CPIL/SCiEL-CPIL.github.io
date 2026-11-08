# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_random_compact(n, rng, min_dist=0.65):
    """Generate a random compact cluster with minimum distance constraints."""
    points = []
    radius = 0.9 * n ** (1.0 / 3.0) + 1.5
    max_attempts = 10000 * n
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        if not points:
            points.append(p)
        else:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) >= min_dist:
                points.append(p)
            else:
                attempts += 1
    if len(points) < n:
        points = []
        side = int(np.ceil(n ** (1/3))) + 1
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if len(points) < n:
                        points.append(np.array([i, j, k]) * 0.75)
        points = np.array(points[:n])
        points -= points.mean(axis=0)
    else:
        points = np.array(points)
        points -= points.mean(axis=0)
    return points


def _generate_icosahedral(n, rng):
    """Generate an icosahedral-like cluster with multiple shells."""
    points = []
    # Center atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Icosahedral vertices (12 atoms) - first shell
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    scale = 1.0
    for v in ico_vertices:
        if len(points) < n:
            points.append(np.array(v, dtype=float) * scale)
    
    # Second shell: 30 atoms (edge midpoints of icosahedron)
    if len(points) < n:
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                v1 = np.array(ico_vertices[i], dtype=float) * scale
                v2 = np.array(ico_vertices[j], dtype=float) * scale
                d = np.linalg.norm(v1 - v2)
                if 1.8 < d < 2.2:
                    mid = (v1 + v2) / 2
                    edges.append(mid)
        
        for e in edges:
            if len(points) < n:
                if points:
                    dists = np.linalg.norm(np.array(points) - e, axis=1)
                    if np.min(dists) >= 0.5:
                        points.append(e)
    
    # Third shell: 20 atoms (face centers of icosahedron)
    if len(points) < n:
        faces = []
        for i in range(12):
            for j in range(12):
                for k in range(12):
                    if i < j < k:
                        v1 = np.array(ico_vertices[i], dtype=float) * scale
                        v2 = np.array(ico_vertices[j], dtype=float) * scale
                        v3 = np.array(ico_vertices[k], dtype=float) * scale
                        d12 = np.linalg.norm(v1 - v2)
                        d13 = np.linalg.norm(v1 - v3)
                        d23 = np.linalg.norm(v2 - v3)
                        if all(1.8 < d < 2.2 for d in [d12, d13, d23]):
                            center = (v1 + v2 + v3) / 3
                            faces.append(center)
        
        for f in faces:
            if len(points) < n:
                if points:
                    dists = np.linalg.norm(np.array(points) - f, axis=1)
                    if np.min(dists) >= 0.5:
                        points.append(f)
    
    # Fourth shell: fill randomly
    if len(points) < n:
        rng_local = np.random.default_rng(42)
        while len(points) < n:
            r = 2.0 * scale * (1 + 0.3 * rng_local.random())
            theta = rng_local.random() * 2 * np.pi
            z = 2 * rng_local.random() - 1
            xy = np.sqrt(1 - z*z)
            p = np.array([r * xy * np.cos(theta), r * xy * np.sin(theta), r * z])
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.55:
                    points.append(p)
            else:
                points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_fcc_cluster(n, rng):
    """Generate an FCC-like cluster."""
    points = []
    a = 1.0
    max_coord = int(np.ceil(n ** (1/3))) + 2
    fcc_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                fcc_points.append(np.array([i, j, k]) * a)
                fcc_points.append(np.array([i + 0.5, j + 0.5, k]) * a)
                fcc_points.append(np.array([i + 0.5, j, k + 0.5]) * a)
                fcc_points.append(np.array([i, j + 0.5, k + 0.5]) * a)
    
    fcc_points = np.array(fcc_points)
    center = fcc_points.mean(axis=0)
    fcc_points -= center
    norms = np.linalg.norm(fcc_points, axis=1)
    max_r = 1.2 * n ** (1/3)
    mask = norms <= max_r
    fcc_points = fcc_points[mask]
    
    idx = rng.permutation(len(fcc_points))
    points = fcc_points[idx[:min(n, len(fcc_points))]]
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_r, max_r, 3)
            if np.linalg.norm(p) <= max_r:
                if len(points) > 0:
                    dists = np.linalg.norm(points - p, axis=1)
                    if np.min(dists) >= 0.5:
                        points = np.vstack([points, p])
                else:
                    points = p.reshape(1, 3)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_hcp_cluster(n, rng):
    """Generate an HCP-like cluster."""
    points = []
    a = 1.0
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3))) + 2
    
    hcp_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                x = a * (i + j/2)
                y = a * np.sqrt(3)/2 * j
                z = c/2 * k
                hcp_points.append(np.array([x, y, z]))
                x2 = a * (i + j/2 + 1/3)
                y2 = a * np.sqrt(3)/2 * (j + 2/3)
                z2 = c/2 * k + c/2
                hcp_points.append(np.array([x2, y2, z2]))
    
    hcp_points = np.array(hcp_points)
    center = hcp_points.mean(axis=0)
    hcp_points -= center
    norms = np.linalg.norm(hcp_points, axis=1)
    max_r = 1.2 * n ** (1/3)
    mask = norms <= max_r
    hcp_points = hcp_points[mask]
    
    idx = rng.permutation(len(hcp_points))
    points = hcp_points[idx[:min(n, len(fcc_points))]]
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_r, max_r, 3)
            if np.linalg.norm(p) <= max_r:
                if len(points) > 0:
                    dists = np.linalg.norm(points - p, axis=1)
                    if np.min(dists) >= 0.5:
                        points = np.vstack([points, p])
                else:
                    points = p.reshape(1, 3)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _descriptor_cost(flat_pos, n, target_r=1.0, width=0.3):
    """
    Coarse-grained descriptor cost function.
    Encourages nearest neighbors to be at target_r and penalizes large deviations.
    Also penalizes if atoms are too close (overlap) or too far (disconnected).
    """
    pos = flat_pos.reshape(n, 3)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    dists = pdist(pos)
    
    # For each atom, find the nearest neighbor distance
    # This is tricky with pdist. Let's reconstruct the matrix for simplicity 
    # or use a more efficient method. For n<=80, squareform is fine.
    dist_matrix = squareform(dists)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Nearest neighbor distance for each atom
    nn_dists = np.min(dist_matrix, axis=1)
    
    # Cost 1: Nearest neighbors should be close to target_r
    # Gaussian penalty
    cost_nn = np.sum((nn_dists - target_r) ** 2)
    
    # Cost 2: Penalize overlaps (dist < 0.5)
    overlaps = np.sum(np.maximum(0, 0.5 - dists) ** 2)
    
    # Cost 3: Penalize very long nearest neighbor distances (disconnection)
    # If an atom is far from its nearest neighbor, it's likely a defect
    cost_disconnect = np.sum(np.maximum(0, nn_dists - 1.5) ** 2)
    
    # Cost 4: Compactness (minimize radius of gyration)
    center_of_mass = np.mean(pos, axis=0)
    radius_of_gyration = np.mean(np.linalg.norm(pos - center_of_mass, axis=1) ** 2)
    cost_compact = radius_of_gyration * 0.1
    
    return cost_nn + 10.0 * overlaps + 5.0 * cost_disconnect + cost_compact


def _phase1_optimize(initial_pos, n, time_limit):
    """
    Phase 1: Minimize the coarse-grained descriptor using Nelder-Mead.
    """
    start_time = time.time()
    flat_pos = initial_pos.flatten()
    
    # Callback to check time
    def time_callback(xk):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit reached")
    
    try:
        result = minimize(
            _descriptor_cost,
            flat_pos,
            args=(n,),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 0.01, 'fatol': 1e-4, 'adaptive': True},
            callback=time_callback
        )
        best_pos = result.x.reshape(n, 3)
    except Exception:
        # If time runs out or error, return the current best (or initial if failed)
        # Nelder-Mead doesn't easily expose intermediate bests in callback without custom logic.
        # We'll just use the result if it exists, otherwise fallback.
        if hasattr(result, 'x') and np.isfinite(result.x).all():
            best_pos = result.x.reshape(n, 3)
        else:
            best_pos = initial_pos
    
    # Ensure validity
    if n > 1:
        min_dist = pdist(best_pos).min()
        if min_dist < 0.35:
            # Push apart
            for _ in range(10):
                best_pos += np.random.default_rng(0).normal(0, 0.05, best_pos.shape)
                min_dist = pdist(best_pos).min()
                if min_dist >= 0.35:
                    break
    
    return best_pos


def _hybrid_worker(initial_pos, kind, n, time_limit_phase1, time_limit_phase2, rng_seed):
    """
    Two-phase optimization:
    1. Coarse-grained descriptor minimization (Nelder-Mead)
    2. High-precision Sutton-Chen relaxation (L-BFGS-B)
    3. Short basin hopping polish
    """
    start_time = time.time()
    rng = np.random.default_rng(rng_seed)
    
    # Phase 1: Descriptor Optimization
    pos_phase1 = _phase1_optimize(initial_pos, n, time_limit_phase1)
    
    # Phase 2: High-Precision Relaxation
    energy, pos_phase2 = relax(pos_phase1, 'sutton_chen', kind)
    best_energy = energy
    best_pos = pos_phase2
    
    # Phase 3: Short Basin Hopping Polish
    remaining_time = time_limit_phase2 - (time.time() - start_time)
    if remaining_time > 5:
        current_energy = best_energy
        current_pos = best_pos
        step_size = 0.2
        k_atoms = max(2, int(0.1 * n))
        
        # Simple loop for a few iterations
        max_iter = int(remaining_time / 2)  # Estimate ~2s per relax+step
        for i in range(max(1, min(50, max_iter))):
            if time.time() - start_time > time_limit_phase1 + time_limit_phase2 - 5:
                break
            
            perturbed = current_pos.copy()
            if k_atoms < n:
                indices = rng.choice(n, size=k_atoms, replace=False)
                perturbed[indices] += rng.normal(0, step_size, (k_atoms, 3))
            else:
                perturbed += rng.normal(0, step_size, current_pos.shape)
            
            new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
            
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos = new_pos
                step_size *= 0.9
            else:
                # Metropolis accept
                delta_e = new_energy - current_energy
                temp = 0.2 * np.exp(-i / 10.0) + 0.01
                if rng.random() < np.exp(-delta_e / temp):
                    current_energy = new_energy
                    current_pos = new_pos
                    step_size *= 1.05
                else:
                    step_size *= 0.9
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
            
            # Validity check
            min_dist = pdist(current_pos).min()
            if min_dist < 0.35:
                for _ in range(5):
                    current_pos += rng.normal(0, 0.05, current_pos.shape)
                    min_dist = pdist(current_pos).min()
                    if min_dist >= 0.35:
                        break
                current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Parallel two-phase hierarchical optimization for Sutton-Chen cluster global minima.
    """
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate diverse initial structures
    rng_master = np.random.default_rng(42)
    
    initial_structures = []
    
    # Random compact
    initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.65))
    
    # Icosahedral-like
    try:
        initial_structures.append(_generate_icosahedral(n, rng_master))
    except Exception:
        initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.65))
    
    # FCC-like
    try:
        initial_structures.append(_generate_fcc_cluster(n, rng_master))
    except Exception:
        initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.65))
    
    # HCP-like
    try:
        initial_structures.append(_generate_hcp_cluster(n, rng_master))
    except Exception:
        initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.65))
    
    results = []
    
    # Use ThreadPoolExecutor with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        num_workers = min(4, len(initial_structures))
        
        # Time allocation: 40% Phase 1, 60% Phase 2+3
        total_time_per_worker = (deadline - time.time()) / num_workers - 10
        if total_time_per_worker < 20:
            total_time_per_worker = 20
        
        time_phase1 = 0.4 * total_time_per_worker
        time_phase2 = 0.6 * total_time_per_worker
        
        for i in range(num_workers):
            start_pos = initial_structures[i % len(initial_structures)]
            seed = 100 + i
            futures.append(executor.submit(
                _hybrid_worker, 
                start_pos, 
                kind, 
                n, 
                time_phase1, 
                time_phase2, 
                seed
            ))
        
        for future in futures:
            try:
                energy, pos = future.result()
                if np.isfinite(energy):
                    results.append((energy, pos))
            except Exception:
                pass
    
    if not results:
        # Fallback: just relax the first initial structure
        try:
            _, best_pos = relax(initial_structures[0], 'sutton_chen', kind)
            return best_pos
        except Exception:
            return initial_structures[0]
    
    # Find the best result
    best_energy, best_pos = min(results, key=lambda x: x[0])
    
    # Final check for validity
    if n > 1:
        min_dist = pdist(best_pos).min()
        if min_dist < 0.35:
            rng_fix = np.random.default_rng(123)
            for _ in range(10):
                best_pos += rng_fix.normal(0, 0.05, best_pos.shape)
                min_dist = pdist(best_pos).min()
                if min_dist >= 0.35:
                    break
            try:
                best_energy, best_pos = relax(best_pos, 'sutton_chen', kind)
            except Exception:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
