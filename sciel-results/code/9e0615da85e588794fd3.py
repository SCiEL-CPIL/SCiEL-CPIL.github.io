import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize
import time

def _get_icosahedron_vertices(radius=1.0):
    """Generate the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _get_icosahedron_edges():
    """Get the 30 edges of the icosahedron (pairs of vertex indices)."""
    verts = _get_icosahedron_vertices(1.0)
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, min_dist, rtol=1e-6):
                edges.append((i, j))
    return edges

def _build_mackay_55(r1=1.0, r2=1.9):
    """
    Build the 55-atom Mackay icosahedron:
    - 1 center
    - 12 vertices (shell 1)
    - 42 atoms (shell 2: 12 vertex dirs + 30 edge mid dirs)
    """
    base_verts = _get_icosahedron_vertices(1.0)
    edges = _get_icosahedron_edges()
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 icosahedron vertices at radius r1
    for v in base_verts:
        points.append(v * r1)
    
    # Shell 2: 42 atoms at radius r2
    shell2_dirs = []
    for v in base_verts:
        shell2_dirs.append(v)
    for i, j in edges:
        mid = (base_verts[i] + base_verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        shell2_dirs.append(mid)
    
    for d in shell2_dirs:
        points.append(d * r2)
    
    return np.array(points)

def _vectorized_energy_and_gradient(positions):
    """
    Compute Dzugutov energy and gradient using vectorized NumPy operations.
    Returns (energy, gradient).
    """
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute all pair differences
    # diff[i, j] = positions[i] - positions[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask for i < j to avoid double counting and self-interaction
    i, j = np.triu_indices(n, k=1)
    r = dists[i, j]
    rij_vec = diff[i, j]
    
    # Initialize energy and gradient
    energy = 0.0
    grad = np.zeros_like(positions)
    
    # Mask for r < a
    mask_a = r < a
    # Mask for a <= r < b
    mask_b = (r >= a) & (r < b)
    
    if np.any(mask_a):
        r_a = r[mask_a]
        vec_a = rij_vec[mask_a]
        idx_a = (i[mask_a], j[mask_a])
        
        # V(r) = A * (r^-m - B) * exp(c / (r - a))
        exp_term = np.exp(c / (r_a - a))
        r_inv_m = r_a ** (-m)
        v = A * (r_inv_m - B) * exp_term
        energy += np.sum(v)
        
        # dV/dr = A * [ -m * r^(-m-1) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c/(r-a)^2) ]
        # dV/dr = A * exp(c/(r-a)) * [ -m * r^(-m-1) - c * (r^-m - B) / (r-a)^2 ]
        dV_dr = A * exp_term * ( -m * r_a ** (-m - 1) - c * (r_a ** (-m) - B) / (r_a - a)**2 )
        
        # Gradient contribution: -dV/dr * (rij_vec / r)
        # grad_i += dV_dr * (rij_vec / r)  (since force is -grad, and we want grad of energy)
        # Actually, grad_i = sum_j dV/dr * (r_i - r_j) / r
        # Let's accumulate
        factor = dV_dr / r_a
        # We need to scatter-add to grad[i] and grad[j]
        # grad[i] += factor * vec
        # grad[j] -= factor * vec
        np.add.at(grad, idx_a[0], factor[:, np.newaxis] * vec_a)
        np.add.at(grad, idx_a[1], -factor[:, np.newaxis] * vec_a)

    if np.any(mask_b):
        r_b = r[mask_b]
        vec_b = rij_vec[mask_b]
        idx_b = (i[mask_b], j[mask_b])
        
        # V(r) = B * exp(d / (r - b))
        exp_term = np.exp(d / (r_b - b))
        v = B * exp_term
        energy += np.sum(v)
        
        # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
        dV_dr = B * exp_term * (-d / (r_b - b)**2)
        
        factor = dV_dr / r_b
        np.add.at(grad, idx_b[0], factor[:, np.newaxis] * vec_b)
        np.add.at(grad, idx_b[1], -factor[:, np.newaxis] * vec_b)

    return energy, grad

def _relax_lbfgs(positions, max_iter=200, tol=1e-6):
    """Relax positions using L-BFGS-B with vectorized gradient."""
    def objective_and_grad(flat_pos):
        pos = flat_pos.reshape(-1, 3)
        try:
            e, g = _vectorized_energy_and_gradient(pos)
            if not np.isfinite(e):
                return 1e10, np.zeros_like(flat_pos)
            return e, g.ravel()
        except Exception:
            return 1e10, np.zeros_like(flat_pos)
    
    try:
        result = minimize(
            objective_and_grad,
            positions.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'gtol': tol}
        )
        if result.success or result.fun < 1e10:
            opt_pos = result.x.reshape(-1, 3)
            opt_energy = result.fun
            return opt_energy, opt_pos
    except Exception:
        pass
    
    # Fallback to chem_clusters relax if L-BFGS fails
    try:
        energy, pos = relax(positions, 'dzugutov', None)
        return energy, pos
    except Exception:
        return float('inf'), positions

def _check_min_distance(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(len(positions), k=1)
    return np.min(dists[i, j]) >= min_dist

def construct(instance):
    """
    Find low-energy Dzugutov cluster structures using Mackay icosahedral seeds
    and L-BFGS-B gradient-based optimization with basin hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290  # 10s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Generate a high-quality 55-atom Mackay icosahedron and relax it
    # This serves as the reference structure for all sizes
    mackay55 = _build_mackay_55(r1=1.0, r2=1.9)
    
    # Relax the 55-atom cluster to get a good baseline
    # We use a stricter tolerance for the baseline
    baseline_energy, baseline_positions = _relax_lbfgs(mackay55, max_iter=500, tol=1e-8)
    
    # If baseline is bad, try a few random restarts for the 55-atom cluster
    if baseline_energy > 1e5 or not _check_min_distance(baseline_positions):
        for _ in range(3):
            rng = np.random.default_rng(np.random.randint(100000))
            perturbed = mackay55 + rng.normal(0, 0.1, mackay55.shape)
            e, p = _relax_lbfgs(perturbed, max_iter=200)
            if e < baseline_energy and _check_min_distance(p):
                baseline_energy = e
                baseline_positions = p

    def try_optimize(init_positions, label=""):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return False
        
        # Relax the initial positions
        try:
            energy, positions = _relax_lbfgs(init_positions, max_iter=200)
            if energy < best_energy and _check_min_distance(positions):
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
        
        return True

    # Strategy based on size relative to 55
    if size == 55:
        # Use the baseline directly
        try_optimize(baseline_positions, "baseline_55")
        
        # Basin hopping around the baseline
        rng = np.random.default_rng(42)
        for i in range(10):
            if time.time() > deadline - 10:
                break
            perturbed = baseline_positions + rng.normal(0, 0.2, baseline_positions.shape)
            try_optimize(perturbed, f"bh_55_{i}")

    elif size < 55:
        # Remove atoms from the outer shell of the relaxed 55-atom cluster
        # Shell 1: indices 1-12
        # Shell 2: indices 13-54
        # Center: index 0
        num_to_remove = 55 - size
        
        # Identify shell 2 atoms (indices 13-54)
        shell2_indices = list(range(13, 55))
        
        # We want to remove 'num_to_remove' atoms.
        # To maintain some symmetry or just get a good start, we can remove random atoms from shell 2.
        # If we need to remove more than 42, we remove from shell 1.
        
        rng = np.random.default_rng(size)
        
        # Shuffle shell 2 indices
        rng.shuffle(shell2_indices)
        
        remove_indices = []
        for idx in shell2_indices:
            if len(remove_indices) < num_to_remove:
                remove_indices.append(idx)
            else:
                break
        
        if len(remove_indices) < num_to_remove:
            # Need to remove from shell 1
            shell1_indices = list(range(1, 13))
            rng.shuffle(shell1_indices)
            for idx in shell1_indices:
                if len(remove_indices) < num_to_remove:
                    remove_indices.append(idx)
                else:
                    break
        
        remove_set = set(remove_indices[:num_to_remove])
        keep_indices = [i for i in range(55) if i not in remove_set]
        
        init_positions = baseline_positions[keep_indices]
        try_optimize(init_positions, f"defect_{size}")
        
        # Basin hopping: perturb the defect structure
        rng = np.random.default_rng(size + 100)
        for i in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = init_positions + rng.normal(0, 0.15, init_positions.shape)
            try_optimize(perturbed, f"bh_defect_{size}_{i}")

    else: # size > 55
        # Add atoms to the relaxed 55-atom cluster
        num_to_add = size - 55
        
        # Generate candidate positions for new atoms
        # Use the 3rd shell of the Mackay icosahedron (120 atoms) or just random positions near the surface
        # For simplicity and robustness, let's use a "kick" strategy:
        # Place new atoms at random positions within a shell around the cluster and relax.
        
        # A better initial guess: Place them at the vertices of a larger icosahedron or similar
        # But for now, let's use random positions in a spherical shell
        rng = np.random.default_rng(size + 200)
        
        init_positions = list(baseline_positions)
        
        # Estimate radius of cluster
        max_r = np.max(np.linalg.norm(baseline_positions, axis=1))
        
        for _ in range(num_to_add):
            # Place new atom in a shell between max_r and max_r + 1.0
            r_new = max_r + rng.uniform(0.5, 1.5)
            # Random direction
            dir_new = rng.normal(0, 1, 3)
            dir_new /= np.linalg.norm(dir_new)
            new_pos = dir_new * r_new
            init_positions.append(new_pos)
        
        init_positions = np.array(init_positions)
        try_optimize(init_positions, f"add_{size}")
        
        # Basin hopping: perturb the added-atom structure
        rng = np.random.default_rng(size + 300)
        for i in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = init_positions + rng.normal(0, 0.15, init_positions.shape)
            try_optimize(perturbed, f"bh_add_{size}_{i}")

    # Final check
    if best_positions is None:
        # Fallback: return the relaxed Mackay 55 truncated or extended
        if size == 55:
            return baseline_positions
        elif size < 55:
            return baseline_positions[:size]
        else:
            # Pad with zeros? No, just return baseline if we can't do better
            # This is a failure case, but we must return something
            return baseline_positions

    return best_positions
