import numpy as np
from scipy.optimize import minimize
import time

N = 55

def _lj_energy_grad(pos):
    """
    Compute total LJ energy and gradient for a set of positions.
    Vectorized for speed.
    """
    n = pos.shape[0]
    # pos shape (N, 3)
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    # Energy term: 4 * (r^-12 - r^-6)
    # Sum over all pairs i,j and divide by 2
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # Gradient calculation
    # dE/dr2 = 4 * (-6*(r^2)^-7 - (-3)*(r^2)^-4) = 4 * (-6/r2^7 + 3/r2^4)
    # dE/dpos_i = sum_j (dE/dr2_ij) * d(r2_ij)/dpos_i
    # d(r2_ij)/dpos_i = 2 * (pos_i - pos_j)
    # So grad_i = sum_j 2 * (4 * (-6/r2^7 + 3/r2^4)) * (pos_i - pos_j)
    #             = sum_j 8 * (-6/r2^7 + 3/r2^4) * (pos_i - pos_j)
    #             = sum_j (-48/r2^7 + 24/r2^4) * (pos_i - pos_j)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = -48.0 / (r2 ** 7) + 24.0 / (r2 ** 4)
    
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * (pos_i - pos_j)
    #         = pos_i * sum_j factor_ij - sum_j factor_ij * pos_j
    
    sum_factor = np.sum(factor, axis=1, keepdims=True) # Shape (N, 1)
    # factor shape (N, N), pos shape (N, 3)
    # sum_j factor_ij * pos_j -> matrix multiplication factor @ pos
    sum_factor_pos = factor @ pos # Shape (N, 3)
    
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_icosahedron_unit_vectors():
    """
    Returns unit vectors for the 55 positions in a standard Mackay icosahedron.
    Returns:
        center (1, 3)
        shell1 (12, 3) - vertices
        shell2 (30, 3) - edge midpoints
        shell3 (12, 3) - vertices (same as shell1)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges (shortest distances between vertices)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    # The minimum non-zero distance is the edge length
    mask = dists > 1e-6
    min_dist = np.min(dists[mask])
    
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
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _make_mackay_icosahedron(r1, r2, r3):
    """
    Construct a two-shell Mackay icosahedron for 55 atoms with specific radii.
    r1: radius of 1st shell (12 atoms)
    r2: radius of 2nd shell (30 atoms)
    r3: radius of 3rd shell (12 atoms)
    """
    center, s1, s2, s3 = _get_mackay_icosahedron_unit_vectors()
    
    first_shell = s1 * r1
    second_shell = s2 * r2
    third_shell = s3 * r3
    
    all_pos = np.vstack([center, first_shell, second_shell, third_shell])
    return all_pos

def _radial_energy_and_grad(radii, dirs):
    """
    Compute energy and gradient with respect to radial coordinates.
    radii: shape (N-1,) for the 54 non-center atoms
    dirs: shape (N-1, 3) unit direction vectors
    """
    n_noncenter = len(radii)
    # Build full positions: center at origin, then dirs * radii
    pos = np.zeros((n_noncenter + 1, 3))
    pos[1:] = dirs * radii[:, np.newaxis]
    
    # Compute full energy and gradient w.r.t. positions
    e, g_pos = _lj_energy_grad(pos)
    
    # Gradient w.r.t. radii: g_r[i] = dot(g_pos[i+1], dirs[i])
    g_r = np.sum(g_pos[1:] * dirs, axis=1)
    
    return e, g_r

def _optimize_radial_manifold(dirs, r0, deadline, maxiter=200):
    """
    Optimize on the radial manifold: each atom moves along its fixed icosahedral direction.
    This reduces the problem from 165D to 54D and preserves the icosahedral symmetry.
    """
    def obj_and_grad(r):
        e, g = _radial_energy_and_grad(r, dirs)
        return e, g
    
    x0 = r0.copy()
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(0.5, 5.0)] * len(r0),
            options={'maxiter': maxiter, 'gtol': 1e-8, 'maxls': 50, 'ftol': 1e-10}
        )
        return result.x, result.fun
    except Exception:
        return x0, _radial_energy_and_grad(x0, dirs)[0]

def _relax_lbfgs(pos, deadline, maxiter=500, gtol=1e-8):
    """
    Relax using L-BFGS-B with analytic gradient and deadline guard.
    """
    best_energy = _lj_energy_grad(pos)[0]
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        # Use maxls=100 to allow more line search steps for difficult landscapes
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """
    Constructs the 55-atom LJ cluster by:
    1. Starting from the Mackay icosahedron
    2. Optimizing on the radial manifold (54D) to find optimal shell radii
    3. Full 165D relaxation
    4. Multi-start with symmetry-breaking perturbations if time permits
    5. Final high-precision polish
    """
    start_time = time.time()
    # Evaluator limit is 150s. We use 145s to be safe.
    deadline = start_time + 145.0
    
    # Get icosahedral directions
    center, s1, s2, s3 = _get_mackay_icosahedron_unit_vectors()
    
    # Build direction matrix for all 54 non-center atoms
    # Order: 12 shell1, 30 shell2, 12 shell3
    dirs = np.vstack([s1, s2, s3])  # shape (54, 3)
    
    # Initial radii guess (standard Mackay)
    r1_init = 1.12
    r2_init = 1.75
    r3_init = 1.95
    r0 = np.concatenate([np.full(12, r1_init), np.full(30, r2_init), np.full(12, r3_init)])
    
    best_pos = None
    best_energy = np.inf
    
    # Step 1: Optimize on radial manifold (fast, 54D)
    if time.time() < deadline - 60.0:
        r_opt, e_radial = _optimize_radial_manifold(dirs, r0, deadline, maxiter=300)
        
        # Build positions from optimized radii
        pos_radial = np.zeros((55, 3))
        pos_radial[1:] = dirs * r_opt[:, np.newaxis]
        
        best_energy = e_radial
        best_pos = pos_radial.copy()
        
        # Step 2: Full 3D relaxation from radial manifold result
        # Use a moderate tolerance first to get close
        if time.time() < deadline - 40.0:
            pos_relaxed, energy_relaxed = _relax_lbfgs(pos_radial, deadline, maxiter=500, gtol=1e-6)
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed.copy()
    
    # Step 3: Multi-start search with symmetry-breaking perturbations
    rng = np.random.default_rng(42)
    candidates = []
    
    if best_pos is not None:
        candidates.append((best_energy, best_pos.copy()))
    
    # Generate diverse perturbation candidates
    # Reduce number of starts if time is tight, but ensure we have a few
    num_starts = 4 if time.time() < deadline - 50.0 else 2
    
    for i in range(num_starts):
        if time.time() > deadline - 20.0:
            break
        
        # Strategy: perturb the best current solution with small noise
        if best_pos is not None:
            pos_seed = best_pos.copy()
        else:
            pos_seed = _make_mackay_icosahedron(1.12, 1.75, 1.95)
        
        # Add random noise to break symmetry
        # Smaller noise for later iterations to refine
        noise_scale = 0.02 + 0.01 * i
        pos_seed += rng.normal(0, noise_scale, size=pos_seed.shape)
        
        # Relax this candidate
        # Use tighter tolerance for these starts as they are close to optimum
        maxiter_this = 300
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_seed, deadline, maxiter=maxiter_this, gtol=1e-8)
        
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed.copy()
        
        candidates.append((energy_relaxed, pos_relaxed))
    
    # Step 4: Final high-precision polish
    if time.time() < deadline - 5.0 and best_pos is not None:
        # Sort candidates by energy and take the best
        candidates.sort(key=lambda x: x[0])
        best_candidate_pos = candidates[0][1]
        
        # Do a final high-precision relaxation with very tight tolerance
        # This is critical for reaching the global minimum precision
        pos_final, energy_final = _relax_lbfgs(
            best_candidate_pos, deadline, maxiter=1000, gtol=1e-12
        )
        if energy_final < best_energy:
            best_energy = energy_final
            best_pos = pos_final
            
        # One more ultra-fine polish if time permits
        if time.time() < deadline - 2.0:
            pos_ultra, energy_ultra = _relax_lbfgs(
                best_pos, deadline, maxiter=200, gtol=1e-14
            )
            if energy_ultra < best_energy:
                best_energy = energy_ultra
                best_pos = pos_ultra

    # Final Safety Check: Minimum Distance
    # Only scale if strictly violated to avoid energy increase from unnecessary scaling
    if best_pos is not None:
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5 - 1e-9:
            scale_factor = 0.5 / min_dist * 1.001
            best_pos *= scale_factor
    
    if best_pos is None:
        # Fallback: return standard Mackay
        best_pos = _make_mackay_icosahedron(1.12, 1.75, 1.95)
    
    return best_pos
