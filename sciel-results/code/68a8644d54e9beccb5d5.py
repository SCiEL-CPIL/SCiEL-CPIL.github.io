"""Lennard-Jones 55-atom global minimum: symmetry-adapted radial optimization + Cartesian polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_icosahedral_symmetry_ops():
    """
    Generate all 120 rotation matrices of the icosahedral group I_h.
    Returns array of shape (120, 3, 3).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    
    # 12 icosahedron vertices (unit vectors)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Generate rotations by finding all rotation matrices that map the vertex set to itself
    # Use a simpler approach: generate from known generators
    # Icosahedral group generators:
    # C5 rotation about z-axis by 72 degrees
    # C2 rotation about x-axis by 180 degrees
    
    theta = 2.0 * np.pi / 5.0
    c5 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    c2_x = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    # Generate all 60 proper rotations by combining generators
    # Start with identity
    rotations = [np.eye(3)]
    
    # BFS/DFS to generate group
    visited = set()
    def _key(mat):
        return tuple(np.round(mat, 10).flatten())
    
    visited.add(_key(np.eye(3)))
    queue = [np.eye(3)]
    
    while queue:
        current = queue.pop(0)
        for gen in [c5, c5.T, c2_x]:
            new_rot = gen @ current
            key = _key(new_rot)
            if key not in visited:
                visited.add(key)
                rotations.append(new_rot)
                queue.append(new_rot)
    
    # Should have 60 proper rotations
    # Add improper rotations (reflections) for I_h: multiply by -I for inversion
    # Actually I_h = I x {E, i}, so 60 proper + 60 improper
    # The improper ones are: -rotation (inversion composed with rotation)
    all_ops = list(rotations)
    for r in rotations:
        all_ops.append(-r)
    
    # Deduplicate
    seen = set()
    unique_ops = []
    for op in all_ops:
        key = _key(op)
        if key not in seen:
            seen.add(key)
            unique_ops.append(op)
    
    return np.array(unique_ops)

def _build_symmetric_structure(radii, ops):
    """
    Build 55-atom structure from symmetry-adapted radial parameters.
    
    radii: array of 6 unique radial distances
      radii[0]: center (should be 0)
      radii[1]: inner shell (12 atoms) - icosahedron vertices
      radii[2]: middle shell (30 atoms) - edge midpoints  
      radii[3]: outer shell (12 atoms) - same directions as inner
      radii[4]: additional parameter for fine-tuning
      radii[5]: additional parameter for fine-tuning
    
    Actually, let me use a cleaner approach:
    - Start with the Mackay icosahedron template
    - Apply radial scaling per shell
    - Allow small angular perturbations for the 30 middle-shell atoms
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts_unit = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Find edges
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
    
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # radii: [r_center, r_inner, r_middle, r_outer]
    # Use 4 main parameters + 2 for perturbation
    r_center = 0.0  # always at center
    r_inner = radii[0]
    r_middle = radii[1]
    r_outer = radii[2]
    perturb_amp = radii[3] if len(radii) > 3 else 0.0
    perturb_freq = radii[4] if len(radii) > 4 else 1.0
    
    # Build positions
    pos = np.zeros((55, 3))
    pos[0] = [0, 0, 0]  # center
    
    # Inner shell: 12 atoms at icosahedron vertices
    for i in range(12):
        pos[1 + i] = verts_unit[i] * r_inner
    
    # Middle shell: 30 atoms at edge midpoints
    for i in range(30):
        direction = edge_mids[i].copy()
        if perturb_amp > 1e-10:
            # Add small angular perturbation
            # Rotate slightly around an axis perpendicular to the direction
            axis = np.cross(direction, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-10:
                axis = np.cross(direction, np.array([0.0, 1.0, 0.0]))
            axis = axis / np.linalg.norm(axis)
            angle = perturb_amp * np.sin(perturb_freq * i)
            # Rodrigues rotation
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            direction = R @ direction
            direction = direction / np.linalg.norm(direction)
        pos[13 + i] = direction * r_middle
    
    # Outer shell: 12 atoms at icosahedron vertices (same directions as inner)
    for i in range(12):
        pos[43 + i] = verts_unit[i] * r_outer
    
    return pos

def _symmetric_energy(radii):
    """Energy of symmetric structure given radial parameters."""
    pos = _build_symmetric_structure(radii)
    return _lj_energy(pos)

def _relax_lbfgs(pos, maxiter=2000, gtol=1e-12):
    """Relax using L-BFGS-B."""
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

def _check_min_dist(pos, min_d=0.5):
    """Check and fix minimum distance constraint."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d_found = np.min(dists)
    if min_d_found < min_d:
        scale = min_d / min_d_found * 1.01
        pos = pos * scale
    return pos

def construct():
    """
    Construct 55-atom LJ cluster.
    
    Strategy:
    1. Optimize 4-6 symmetry-adapted radial parameters via Nelder-Mead.
    2. Reconstruct full 55-atom structure.
    3. Full Cartesian L-BFGS relaxation.
    4. Multi-start with different initial parameters.
    5. Final polish.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_pos = None
    best_energy = np.inf
    
    # Initial guesses for [r_inner, r_middle, r_outer, perturb_amp, perturb_freq]
    initial_guesses = [
        [1.05, 1.75, 1.95, 0.0, 1.0],
        [1.10, 1.80, 2.00, 0.0, 1.0],
        [1.00, 1.70, 1.90, 0.0, 1.0],
        [1.05, 1.75, 1.95, 0.02, 1.0],
        [1.05, 1.75, 1.95, 0.05, 2.0],
    ]
    
    for guess in initial_guesses:
        if time.time() >= deadline - 25.0:
            break
        
        # Step 1: Optimize radial parameters
        try:
            result = minimize(
                _symmetric_energy,
                np.array(guess),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500}
            )
            radii_opt = result.x
            # Clamp to reasonable range
            radii_opt[0] = np.clip(radii_opt[0], 0.8, 1.5)
            radii_opt[1] = np.clip(radii_opt[1], 1.4, 2.2)
            radii_opt[2] = np.clip(radii_opt[2], 1.6, 2.4)
            radii_opt[3] = np.clip(radii_opt[3], -0.1, 0.1)
            radii_opt[4] = np.clip(radii_opt[4], 0.5, 3.0)
        except Exception:
            radii_opt = np.array(guess)
        
        # Step 2: Build structure
        pos_sym = _build_symmetric_structure(radii_opt)
        pos_sym = _check_min_dist(pos_sym)
        
        # Step 3: Full Cartesian relaxation
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_sym, maxiter=2000, gtol=1e-12)
        
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # Step 4: Multi-start perturbations around best structure
    if best_pos is not None and time.time() < deadline - 15.0:
        rng = np.random.default_rng(42)
        search_deadline = deadline - 10.0
        
        for scale in [0.02, 0.05, 0.1]:
            if time.time() >= search_deadline:
                break
            noise = rng.normal(0, scale, size=best_pos.shape)
            perturbed = best_pos + noise
            perturbed = _check_min_dist(perturbed)
            
            pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed, maxiter=1000, gtol=1e-12)
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # Step 5: Final high-precision polish
    if best_pos is not None and time.time() < deadline - 5.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, maxiter=3000, gtol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    if best_pos is None:
        # Fallback: simple Mackay icosahedron
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        verts_unit = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        pos = np.zeros((55, 3))
        for i in range(12):
            pos[1 + i] = verts_unit[i] * 1.05
        for i in range(30):
            pos[13 + i] = verts_unit[i % 12] * 1.75  # approximate
        for i in range(12):
            pos[43 + i] = verts_unit[i] * 1.95
        best_pos = _check_min_dist(pos)
    
    best_pos = _check_min_dist(best_pos)
    return best_pos

# EVOLVE-BLOCK-END
