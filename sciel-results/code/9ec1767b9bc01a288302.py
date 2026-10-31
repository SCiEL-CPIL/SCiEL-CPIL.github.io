"""LJ38 cluster global minimum via FCC truncated-octahedron seed + basin hopping."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 38

def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum(r^-12 - r^-6) over all pairs."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    d2 = d * d
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    r2 = d2[iu]
    inv_r6 = 1.0 / (r2 * r2 * r2)
    return 4.0 * np.sum(inv_r6 * inv_r6 - inv_r6)

def _lj_grad(pos):
    """Gradient of LJ energy (vectorized)."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    d2 = d * d
    # Prevent division by zero
    d2 = np.maximum(d2, 1e-12)
    
    inv_r2 = 1.0 / d2
    inv_r6 = inv_r2 ** 3
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    # coeff[i,j] = 4 * (12 * r^-14 - 6 * r^-8)
    coeff = 4.0 * (12.0 * inv_r14 - 6.0 * inv_r8)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # = pos[i] * sum_j coeff[i,j] - sum_j coeff[i,j] * pos[j]
    sum_coeff = np.sum(coeff, axis=1)
    grad = pos * sum_coeff[:, np.newaxis] - coeff @ pos
    return grad

def _relax(pos):
    """Local L-BFGS relaxation."""
    try:
        res = minimize(_lj_energy, pos, jac=_lj_grad, method='L-BFGS-B',
                       options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-12})
        return res.x
    except Exception:
        return pos

def _enforce_min_separation(pos, min_sep=0.5):
    """Push atoms apart if any pair is closer than min_sep."""
    for _ in range(50):
        moved = False
        d = cdist(pos, pos)
        np.fill_diagonal(d, np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < min_sep:
                    rij = pos[i] - pos[j]
                    r = np.linalg.norm(rij)
                    if r < 1e-12:
                        rij = np.array([1.0, 0.0, 0.0])
                        r = 1.0
                    push = (min_sep - r) * 0.5
                    unit = rij / r
                    pos[i] += push * unit
                    pos[j] -= push * unit
                    moved = True
        if not moved:
            break
    return pos

def _fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron 38-atom cluster.
    
    The structure is built from FCC lattice points. FCC lattice points are:
    (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) scaled by 2 (so nearest neighbor = 1.0 in reduced units where sigma=1, but we use the conventional lattice constant).
    
    Actually, for LJ with sigma=1, the equilibrium nearest-neighbor distance is 2^(1/6) ≈ 1.12246.
    The FCC lattice with conventional cube edge a has nearest-neighbor distance a/sqrt(2).
    So a = 2^(1/6) * sqrt(2) = 2^(2/3) ≈ 1.5874.
    
    The truncated octahedron for 38 atoms: we take FCC points within a certain radius.
    """
    # FCC lattice: generate points
    # Conventional cubic cell with a = 2^(2/3)
    a = 2.0 ** (2.0 / 3.0)
    
    # Generate FCC lattice points in a range
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    points.append([x, y, z])
    
    points = np.array(points)
    # Center the structure
    center = points.mean(axis=0)
    points -= center
    
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    # Take the 38 closest
    # But we need to be careful: the truncated octahedron has a specific shape.
    # Let's just take the 38 closest to origin.
    pts = points[:N].copy()
    
    return pts

def construct():
    # Start with the FCC truncated-octahedron seed
    pos = _fcc_truncated_octahedron()
    pos = _enforce_min_separation(pos)
    
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Basin hopping with multiple restarts
    np.random.seed(42)
    
    # First, relax the initial structure
    relaxed = _relax(pos)
    relaxed = _enforce_min_separation(relaxed)
    e = _lj_energy(relaxed)
    if e < best_energy:
        best_energy = e
        best_pos = relaxed.copy()
    
    # Basin hopping: perturb, relax, keep if better
    step_size = 0.3
    no_improve_count = 0
    max_iterations = 100
    
    for iteration in range(max_iterations):
        # Perturb the current best
        perturbation = np.random.randn(N, 3) * step_size
        new_pos = best_pos + perturbation
        new_pos = _enforce_min_separation(new_pos)
        
        # Relax
        new_pos = _relax(new_pos)
        new_pos = _enforce_min_separation(new_pos)
        new_energy = _lj_energy(new_pos)
        
        # Accept if better
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            no_improve_count = 0
            # Shrink step size slightly on success to refine
            step_size *= 0.95
        else:
            no_improve_count += 1
            # If stuck, increase step size to explore more
            if no_improve_count > 5:
                step_size *= 1.2
                no_improve_count = 0
                
    return best_pos
