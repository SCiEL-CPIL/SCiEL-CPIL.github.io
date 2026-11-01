"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    # Avoid division by zero or negative if r2 is 0
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0) # Avoid div by 0 in calculation, handled later
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # Gradient calculation
    # dE/dr2 = 4 * (-6/r2^4 + 12/r2^7)
    # dE/dpos_i = sum_j dE/dr2_ij * 2*(pos_i - pos_j)
    # Let's use the explicit form:
    # grad_i = sum_j 4 * (-12/r2^7 + 6/r2^4) * (pos_i - pos_j)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_icosahedron_coords():
    """
    Returns unit vectors for the 55 positions in a Mackay icosahedron.
    1 center (0,0,0)
    12 first shell (icosahedron vertices)
    30 second shell (icosahedron edge midpoints)
    12 third shell (icosahedron vertices, same direction as first shell)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # Construct full unit vectors
    # Order: Center, Shell1 (12), Shell2 (30), Shell3 (12)
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit # Same directions as shell 1
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _build_direction_manifold():
    """
    Build the direction manifold for the 55-atom Mackay icosahedron.
    Returns:
        directions: (55, 3) array of unit direction vectors
        center_idx: index of center atom (0)
    """
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    # Directions for all 55 atoms (center has zero direction, handled separately)
    directions = np.vstack([np.zeros((1, 3)), s1, s2, s3])
    return directions, 0

def _radial_to_positions(radii, directions, center_idx=0):
    """Convert radial distances to Cartesian positions."""
    pos = np.zeros((55, 3))
    # Center atom stays at origin
    pos[center_idx] = [0.0, 0.0, 0.0]
    # Other atoms: position = radius * direction
    non_center = np.arange(55) != center_idx
    pos[non_center] = radii[:, np.newaxis] * directions[non_center]
    return pos

def _radial_energy_and_grad(radii, directions, center_idx=0):
    """
    Compute energy and gradient with respect to radial distances.
    Uses the chain rule: dE/dr_i = sum_j dE/dpos_i · dpos_i/dr_i
    where dpos_i/dr_i = direction_i
    """
    pos = _radial_to_positions(radii, directions, center_idx)
    energy, grad_pos = _lj_energy_grad(pos)
    
    # Gradient with respect to radii: dE/dr_i = grad_pos_i · direction_i
    non_center = np.arange(55) != center_idx
    grad_radial = np.zeros(54)
    grad_radial[:] = np.sum(grad_pos[non_center] * directions[non_center], axis=1)
    
    return energy, grad_radial

def _optimize_radial_manifold(deadline):
    """
    Optimize all 54 radial distances on the icosahedral direction manifold.
    This allows each atom to have an independent radius while maintaining
    the angular structure of the Mackay icosahedron.
    """
    directions, center_idx = _build_direction_manifold()
    
    # Initial guess: use typical Mackay radii
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    r1, r2, r3 = 1.05, 1.75, 1.95
    initial_radii = np.concatenate([
        np.full(12, r1),
        np.full(30, r2),
        np.full(12, r3)
    ])
    
    from scipy.optimize import minimize
    
    def obj_and_grad(x):
        e, g = _radial_energy_and_grad(x, directions, center_idx)
        return e, g
    
    # Use L-BFGS-B for the 54D radial optimization
    try:
        result = minimize(
            obj_and_grad, initial_radii, jac=True, method='L-BFGS-B',
            options={'maxiter': 2000, 'gtol': 1e-10, 'maxls': 100, 'ftol': 1e-14}
        )
        optimal_radii = result.x
    except Exception:
        optimal_radii = initial_radii
    
    return optimal_radii, directions, center_idx

def _relax_lbfgs(pos, deadline, maxiter=1000, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum using radial manifold + full relaxation."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Optimize radial manifold (54D) - allows independent radii
    try:
        if time.time() < deadline - 100.0:
            optimal_radii, directions, center_idx = _optimize_radial_manifold(deadline)
            base_pos = _radial_to_positions(optimal_radii, directions, center_idx)
        else:
            # Fallback to simple Mackay
            center, s1, s2, s3 = _get_mackay_icosahedron_coords()
            base_pos = np.vstack([center, s1 * 1.05, s2 * 1.75, s3 * 1.95])
    except Exception:
        center, s1, s2, s3 = _get_mackay_icosahedron_coords()
        base_pos = np.vstack([center, s1 * 1.05, s2 * 1.75, s3 * 1.95])
    
    best_pos = base_pos.copy()
    best_energy = _lj_energy(base_pos)
    
    # 2. Full 3D relaxation from the manifold-optimized structure
    polish_time = 20.0
    search_deadline = deadline - polish_time
    
    # First, relax the base structure
    if time.time() < search_deadline - 30.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(base_pos, search_deadline, maxiter=3000, gtol=1e-10)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 3. Multi-start with small perturbations around the best structure
    rng = np.random.default_rng(42)
    perturbation_scales = [0.03, 0.05, 0.08, 0.1, 0.15]
    
    for scale in perturbation_scales:
        if time.time() >= search_deadline - 15.0:
            break
        
        noise = rng.normal(0, scale, size=best_pos.shape)
        perturbed_pos = best_pos + noise
        
        # Check minimum distance constraint
        diff = perturbed_pos[:, np.newaxis, :] - perturbed_pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5:
            scale_factor = 0.5 / min_dist * 1.01
            perturbed_pos = perturbed_pos * scale_factor
        
        maxiter = max(1000, 3000 - int(scale * 10000))
        gtol = 1e-10
        
        pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed_pos, search_deadline, maxiter=maxiter, gtol=gtol)
        
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 4. Final high-precision polish with tighter tolerance
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, deadline, maxiter=10000, gtol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Final Check: Minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
