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
    # dE/dr2 = 4 * (-6/r2^4 + 12/r2^7)  ... wait, let's derive carefully.
    # E = 4 * (r^-12 - r^-6) = 4 * ((r^2)^-6 - (r^2)^-3)
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

def _relax_lbfgs(pos, deadline, maxiter=1000):
    """
    Relax using L-BFGS-B with analytic gradient and deadline guard.
    """
    from scipy.optimize import minimize
    
    best_energy = _lj_energy_grad(pos)[0]
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-10, 'maxls': 100, 'ftol': 1e-12}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """
    Constructs the 55-atom LJ cluster by starting from the Mackay icosahedron 
    and performing a multi-start search with symmetry-breaking perturbations 
    to find the global minimum (a distorted Mackay icosahedron).
    """
    start_time = time.time()
    # Evaluator limit is 120s. We use 115s to be safe.
    deadline = start_time + 115.0
    
    # 1. Generate base Mackay icosahedron with standard radii
    # Standard Mackay radii: r1 ~ 1.12, r2 ~ 1.75, r3 ~ 1.95 (approx)
    # Let's use a slightly optimized starting point for the symmetric case first
    base_pos = _make_mackay_icosahedron(1.12, 1.75, 1.95)
    
    best_pos = base_pos.copy()
    best_energy = _lj_energy_grad(base_pos)[0]
    
    # 2. Initial relaxation of the symmetric structure
    if time.time() < deadline - 30.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(base_pos, deadline, maxiter=500)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed.copy()
        # Use the relaxed symmetric structure as the base for perturbations
        base_pos = pos_relaxed
    
    # 3. Multi-start search with symmetry-breaking perturbations
    # The global minimum for N=55 is a distorted Mackay.
    # We generate perturbations that break the icosahedral symmetry.
    rng = np.random.default_rng(42)
    
    # Define perturbation strategies
    # Strategy 1: Random radial perturbations for each shell
    # Strategy 2: Small random Cartesian noise
    
    # We will try several candidates.
    # Time budget: ~10s per candidate relaxation.
    # Total time ~80s for search, leaving 35s for final polish/verification.
    
    candidates = []
    
    # Generate 5 diverse candidates
    # Candidate 0: The relaxed symmetric one (already in best_pos)
    
    # Candidate 1: Radial perturbation
    # Add random noise to the radii of the shells
    center, s1, s2, s3 = _get_mackay_icosahedron_unit_vectors()
    
    for i in range(5):
        if time.time() > deadline - 20.0:
            break
        
        # Create a perturbed seed
        # Randomize radii slightly
        r1 = 1.12 + rng.uniform(-0.1, 0.1)
        r2 = 1.75 + rng.uniform(-0.1, 0.1)
        r3 = 1.95 + rng.uniform(-0.1, 0.1)
        
        pos_seed = _make_mackay_icosahedron(r1, r2, r3)
        
        # Add small Cartesian noise to break symmetry further
        noise_scale = 0.05
        pos_seed += rng.normal(0, noise_scale, size=pos_seed.shape)
        
        # Ensure minimum distance constraint (0.5)
        # If atoms are too close, scale up slightly
        diff = pos_seed[:, np.newaxis, :] - pos_seed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5:
            scale_factor = 0.5 / min_dist * 1.01
            pos_seed *= scale_factor
            
        # Relax this candidate
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_seed, deadline, maxiter=500)
        
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed.copy()
        
        candidates.append((energy_relaxed, pos_relaxed))
        
        # If we found a significantly better energy, we might want to perturb from there
        # But for now, let's stick to the base seed to ensure coverage.
        
    # 4. Final Polish
    # Take the best candidate and relax it with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        # Sort candidates by energy
        candidates.sort(key=lambda x: x[0])
        if candidates:
            best_candidate_pos = candidates[0][1]
            # Do a final high-precision relaxation
            pos_final, energy_final = _relax_lbfgs(best_candidate_pos, deadline, maxiter=1000)
            if energy_final < best_energy:
                best_energy = energy_final
                best_pos = pos_final
    
    # 5. Final Safety Check: Minimum Distance
    # Ensure the final result satisfies the constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < 0.5:
        # This should be rare after relaxation, but handle it just in case
        scale_factor = 0.5 / min_dist * 1.01
        best_pos *= scale_factor
        # Re-check energy after scaling (though scaling changes energy)
        # If this happens, the structure was likely invalid.
        # We return the scaled version as it's the best valid option.
        
    return best_pos
