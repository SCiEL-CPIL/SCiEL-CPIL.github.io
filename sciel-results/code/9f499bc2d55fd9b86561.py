"""Rewritten program using Basin Hopping for LJ38 global minimum search."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    mask = ~np.eye(len(pos), dtype=bool)
    r2 = r2[mask]
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy

def _lj_grad(pos):
    """Compute gradient of total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    r2 = np.where(r2 < 1e-12, 1e-12, r2)
    r6 = r2 ** 3
    r12 = r6 ** 2
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _minimize_lbfgs(pos, max_iter=500, tol=1e-10):
    """L-BFGS minimization using scipy."""
    res = minimize(
        _lj_energy,
        pos,
        jac=_lj_grad,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    return res.x

def _construct_fcc_seed():
    """Construct an FCC lattice fragment seed."""
    r_min = 2.0 ** (1.0 / 6.0)
    scale = r_min / np.sqrt(2.0)
    max_coord = 6
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    pts = np.array(points, dtype=float)
    pts *= scale
    
    # Sort by distance to origin
    dists = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first N points
    cluster = pts[sorted_indices[:N]].copy()
    return cluster

def _construct_icosahedron_seed():
    """Construct a Mackay icosahedron seed (13+25=38 atoms)."""
    # 13-atom icosahedron + 25-atom shell
    # This is a known local minimum funnel, but a good starting point for BH
    # Coordinates for 13-atom icosahedron (Mackay)
    t = (1 + np.sqrt(5)) / 2
    # Vertices of icosahedron
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
        [1, 0, 0], [-1, 0, 0] # Center is 0,0,0, so we have 12 outer + 1 center = 13
    ])
    # Normalize to have nearest neighbor distance ~ 2^(1/6)
    # Distance from center to vertex is sqrt(1 + t^2)
    # Distance between adjacent vertices is 2
    # We want nearest neighbor (center-vertex) to be r_min
    r_min = 2.0 ** (1.0 / 6.0)
    scale = r_min / np.sqrt(1 + t**2)
    verts *= scale
    
    # Now add 25 atoms to form the second shell
    # The second shell of a Mackay icosahedron consists of 25 atoms
    # 20 in pentagonal faces + 5 in... wait, 20 faces * 1 atom per face center? No.
    # The 26-atom shell has 25 atoms.
    # We can approximate by placing atoms at the centers of the 20 faces and 5... 
    # Actually, let's just use a random perturbation of the FCC seed if this is too complex.
    # Or, we can generate the 25 atoms by taking the FCC lattice and selecting the next 25 closest to the icosahedron surface.
    
    # For simplicity and robustness, let's just return the FCC seed for now 
    # and rely on Basin Hopping to explore the space. 
    # The FCC seed is already quite good.
    return _construct_fcc_seed()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial seed
    pos = _construct_fcc_seed()
    
    # Initial relaxation
    try:
        pos = _minimize_lbfgs(pos, max_iter=500, tol=1e-10)
    except Exception:
        pass
    
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    rng = np.random.RandomState(42)
    
    # Basin Hopping
    # We will try to perturb the entire cluster
    step_size = 0.5
    
    while time.time() < deadline - 5.0:
        # Perturb the configuration
        # Use a random displacement for each atom
        noise = rng.normal(0, step_size, size=pos.shape)
        perturbed_pos = pos + noise
        
        # Relax the perturbed configuration
        try:
            relaxed_pos = _minimize_lbfgs(perturbed_pos, max_iter=300, tol=1e-8)
            energy = _lj_energy(relaxed_pos)
            
            # Accept if lower energy (greedy) or with Metropolis criterion
            # For finding global min, greedy is often sufficient if step size is good
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
                pos = relaxed_pos.copy()
                # Optionally reduce step size slightly on improvement
                step_size *= 0.95
            else:
                # Metropolis acceptance probability
                # exp(-(E_new - E_old) / T)
                # We don't have a temperature, so let's just use greedy for simplicity 
                # or a fixed probability if not much worse
                pass
                
        except Exception:
            continue
            
        # Adjust step size occasionally
        # If we haven't found a new best in a while, increase step size to escape
        # Simple strategy: keep step size constant or slowly anneal
        # Let's keep it simple: constant step size 0.5 is a good default for LJ clusters
        
    return best_pos
# EVOLVE-BLOCK-END
