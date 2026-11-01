"""Lennard-Jones cluster global minimum, 55 atoms: Mackay icosahedron construction + relaxation."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    inv_r6 = 1.0 / (r2 * r2 * r2)
    inv_r12 = inv_r6 * inv_r6
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    # Subtract self terms (diagonal was set to 1, so we added 4*(1-1)=0, fine)
    return energy

def _get_mackay_topology():
    """
    Returns the unit direction vectors for the 55 atoms of the Mackay icosahedron.
    Returns:
        center: (1, 3)
        inner_dirs: (12, 3)
        mid_dirs: (30, 3)
        outer_dirs: (12, 3)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of a regular icosahedron
    verts = []
    for sign1 in [1, -1]:
        for sign2 in [1, -1]:
            verts.append([0.0, sign1 * 1.0, sign2 * phi])
            verts.append([sign1 * 1.0, sign2 * phi, 0.0])
            verts.append([sign2 * phi, 0.0, sign1 * 1.0])
    verts = np.array(verts)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # Find edges for midpoints
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[j, i] = dists[i, j]
    
    min_dist = np.min(dists[dists > 1e-6])
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-8:
                edges.append((i, j))
    
    # 30 edges expected
    if len(edges) != 30:
        # Fallback if edge detection fails (should not happen)
        edges = [(i, (i+1)%12) for i in range(12)] # Incorrect fallback, but prevents crash
        # Actually, let's just raise an error if it fails, as it's deterministic
        raise ValueError("Failed to find icosahedron edges")
        
    midpoints = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid_unit = mid / mid_norm
        midpoints.append(mid_unit)
    midpoints = np.array(midpoints)
    
    center = np.array([[0.0, 0.0, 0.0]])
    inner_dirs = verts_unit
    mid_dirs = midpoints
    outer_dirs = verts_unit
    
    return center, inner_dirs, mid_dirs, outer_dirs

def _build_structure(r_inner, r_mid, r_outer):
    """Build the 55-atom structure given radii."""
    center, inner_dirs, mid_dirs, outer_dirs = _get_mackay_topology()
    
    inner_atoms = inner_dirs * r_inner
    mid_atoms = mid_dirs * r_mid
    outer_atoms = outer_dirs * r_outer
    
    all_atoms = np.vstack([center, inner_atoms, mid_atoms, outer_atoms])
    return all_atoms

def _energy_symmetric(params):
    """Energy of the symmetric Mackay icosahedron given radii."""
    r_inner, r_mid, r_outer = params
    pos = _build_structure(r_inner, r_mid, r_outer)
    return _lj_energy(pos)

def _optimize_mackay_radii():
    """
    Optimize the 3 radial parameters of the Mackay icosahedron.
    Returns the best positions found.
    """
    # Initial guess based on typical LJ cluster sizes
    # Nearest neighbor distance ~ 1.122
    # Inner shell atoms are nearest to center? No, center is isolated in 13-atom icosahedron core?
    # In 55-atom Mackay:
    # Center is 1.
    # Inner shell (12) is at distance r1.
    # Mid shell (30) is at distance r2.
    # Outer shell (12) is at distance r3.
    
    # Good initial guess:
    # r1 ~ 1.0, r2 ~ 1.4, r3 ~ 1.9
    x0 = np.array([1.0, 1.4, 1.9])
    
    # Use Nelder-Mead or L-BFGS-B. Nelder-Mead is robust for 3 params.
    result = minimize(_energy_symmetric, x0, method='Nelder-Mead', 
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    
    best_params = result.x
    best_pos = _build_structure(best_params[0], best_params[1], best_params[2])
    
    return best_pos

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (evaluator kills at 150s, but we want to be safe)
    
    # Step 1: Optimize radial parameters of Mackay Icosahedron
    pos = _optimize_mackay_radii()
    current_best_energy = _lj_energy(pos)
    
    # Step 2: Full Cartesian L-BFGS relaxation
    try:
        result = minimize(
            _lj_energy,
            pos,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if time.time() > deadline:
            return result.x
        pos = result.x
        current_best_energy = _lj_energy(pos)
    except Exception:
        pass
    
    # Step 3: Population-based restarts (Differential Evolution style exploration)
    # We use the best structure as a reference and generate perturbations.
    # This helps escape shallow local minima if the radial optimization landed in a suboptimal symmetry-breaking state.
    
    np.random.seed(42) # Reproducibility
    
    num_restarts = 0
    max_restarts = 10
    
    while time.time() < deadline - 10.0 and num_restarts < max_restarts:
        num_restarts += 1
        
        # Generate a perturbed structure
        # Use a "mutation" vector from random noise
        noise = np.random.normal(0, 0.1, size=(55, 3))
        # Add a small amount of "crossover" by mixing with original? 
        # Simple random restart around the best known solution is effective here.
        pos_perturbed = pos + noise
        
        # Relax the perturbed structure
        try:
            result_r = minimize(
                _lj_energy,
                pos_perturbed,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            pos_r = result_r.x
            energy_r = _lj_energy(pos_r)
            
            if energy_r < current_best_energy:
                pos = pos_r
                current_best_energy = energy_r
        except Exception:
            pass
            
    return pos
# EVOLVE-BLOCK-END
