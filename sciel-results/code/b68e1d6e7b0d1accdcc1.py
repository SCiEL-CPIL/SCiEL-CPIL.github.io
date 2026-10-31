"""LJ38 cluster global minimum search: FCC truncated octahedron + L-BFGS relaxation."""
from scipy.spatial.distance import pdist
import numpy as np
import time
import sys

N = 38
SEED_ENERGY = -173.928427  # approximate FCC energy for score normalization reference

# Deadline guard
_DEADLINE = 150.0  # evaluator kills at 150s wall-clock
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    # Zero out self-interactions
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy


def _lj_grad(pos):
    """Compute gradient of LJ energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero; will zero out later
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    # dE/dr2 = 4 * (-12*r12 + 6*r6) / r2 * ... actually:
    # E = 4*(r^-12 - r^-6), dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # Using r2: dE/dr2 = 4*(-6*r^-14 + 3*r^-8) = 4*(-6*r12/r2 + 3*r6/r2)
    # Actually let's be careful:
    # E = 4*(r2^-6 - r2^-3)
    # dE/dr2 = 4*(-6*r2^-7 + 3*r2^-4)
    # grad wrt pos_i = sum_j dE/dr2 * dr2/dpos_i = sum_j dE/dr2 * 2*(pos_i - pos_j)
    # But by symmetry, grad_i = -2 * sum_{j != i} (dE/dr2)_ij * (pos_i - pos_j)
    # Wait, let me redo:
    # E = sum_{i<j} 4*(r_ij^-12 - r_ij^-6) where r_ij = |pos_i - pos_j|
    # r_ij^2 = sum_k (pos_i[k] - pos_j[k])^2
    # dE/dpos_i[k] = sum_{j != i} 4*(-12*r_ij^-14 + 6*r_ij^-7) * (pos_i[k] - pos_j[k])
    # = sum_{j != i} 4*(-12*r_ij^-14 + 6*r_ij^-7) * diff[i,j,k]
    
    # r6 = r^-6 = (r2)^-3, r12 = r^-12 = (r2)^-6
    # -12*r^-14 + 6*r^-7 = -12*r12/r2 + 6*r6/r2 = (-12*r12 + 6*r6)/r2
    coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2
    np.fill_diagonal(coeff, 0.0)
    # grad[i] = sum_j coeff[i,j] * diff[i,j]
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad


def _fcc_truncated_octahedron():
    """
    Generate FCC truncated octahedron coordinates for 38 atoms.
    
    The FCC truncated octahedron for LJ38 has the following structure:
    - It's a 3D cluster with FCC symmetry.
    - The 38 atoms are arranged in layers.
    
    Using known coordinates from literature (Cambridge Cluster Database):
    The structure can be described as an FCC cluster with specific truncation.
    
    A common parameterization: place atoms on FCC lattice points within a truncated octahedron.
    FCC lattice vectors: a/2*(0,1,1), a/2*(1,0,1), a/2*(1,1,0)
    
    For the 38-atom FCC cluster, we use the following layer structure:
    The cluster has a cubic core with atoms added at specific positions.
    
    Let me use a known set of coordinates. The FCC 38 cluster can be generated
    by taking FCC lattice points with |r| <= some cutoff.
    
    FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of lattice constant a.
    We scale so nearest neighbor distance = 1.12 (LJ minimum).
    
    Actually, let me use a more direct approach: generate FCC points and select 38 closest to center.
    """
    # FCC lattice points in a cube
    # Lattice constant a such that nearest neighbor = 1 (we'll scale later)
    # In FCC, nearest neighbor distance = a/sqrt(2)
    # We want nearest neighbor = 1.12, so a = 1.12 * sqrt(2)
    
    a = 1.12 * np.sqrt(2)
    
    # Generate FCC points in a grid
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC basis vectors
                for bx, by, bz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    points.append((x, y, z))
    
    points = np.array(points)
    # Center of mass
    com = points.mean(axis=0)
    points = points - com
    
    # Sort by distance from center
    dists = np.sum(points ** 2, axis=1)
    order = np.argsort(dists)
    points = points[order[:N]]
    
    # Center the cluster
    points = points - points.mean(axis=0)
    
    return points


def _icosahedral_seed():
    """Generate an icosahedral seed (local minimum, but useful for comparison)."""
    # 12 vertices of icosahedron + 5 in each of 2 pentagonal faces + ...
    # This is complex; skip for now and use FCC primarily
    return None


def _perturb(pos, rng, scale=0.1):
    """Add small random perturbation."""
    return pos + rng.normal(0, scale, pos.shape)


def _relax(pos, max_iter=2000, tol=1e-10):
    """
    Perform L-BFGS-like relaxation using scipy if available, else simple gradient descent.
    """
    try:
        from scipy.optimize import minimize
        result = minimize(
            _lj_energy,
            pos,
            jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
        )
        if result.success or result.fun < _lj_energy(pos):
            return result.x
        return pos
    except Exception:
        # Fallback: simple gradient descent with adaptive step
        pos = pos.copy()
        lr = 0.01
        for it in range(max_iter):
            if _time_left() < 5.0:
                break
            g = _lj_grad(pos)
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-10:
                break
            pos_new = pos - lr * g
            e_old = _lj_energy(pos)
            e_new = _lj_energy(pos_new)
            if e_new < e_old:
                pos = pos_new
                lr = min(lr * 1.5, 0.1)
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
        return pos


def construct():
    """
    Construct 38-atom LJ cluster configuration with minimal energy.
    Returns array of shape (38, 3).
    """
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.RandomState(42)
    
    # Start with FCC truncated octahedron
    best_pos = _fcc_truncated_octahedron()
    best_energy = _lj_energy(best_pos)
    
    # Relax the FCC seed
    if _time_left() > 10.0:
        relaxed = _relax(best_pos, max_iter=3000, tol=1e-12)
        e_relaxed = _lj_energy(relaxed)
        if e_relaxed < best_energy:
            best_pos = relaxed
            best_energy = e_relaxed
    
    # Try multiple perturbations and relaxations
    n_tries = 0
    while _time_left() > 15.0 and n_tries < 20:
        # Perturb the best solution
        pert_scale = 0.05 + 0.05 * n_tries
        perturbed = _perturb(best_pos, rng, scale=pert_scale)
        
        # Relax
        relaxed = _relax(perturbed, max_iter=1500, tol=1e-10)
        e_relaxed = _lj_energy(relaxed)
        
        if e_relaxed < best_energy:
            best_pos = relaxed
            best_energy = e_relaxed
            pert_scale = 0.05  # reset perturbation scale
        else:
            n_tries += 1
    
    # Final high-precision polish
    if _time_left() > 5.0:
        polished = _relax(best_pos, max_iter=5000, tol=1e-14)
        e_polished = _lj_energy(polished)
        if e_polished < best_energy:
            best_pos = polished
            best_energy = e_polished
    
    return best_pos


# Quick test
if __name__ == "__main__":
    pos = construct()
    e = _lj_energy(pos)
    print(f"Energy: {e:.10f}")
    print(f"Shape: {pos.shape}")
    # Check minimum distance
    from scipy.spatial.distance import pdist
    min_dist = pdist(pos).min()
    print(f"Min distance: {min_dist:.4f}")
