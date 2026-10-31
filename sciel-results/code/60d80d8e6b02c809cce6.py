"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Known FCC truncated octahedron coordinates for N=38 (reduced LJ units, sigma=1)
# This is the global minimum with energy ~-173.928427
# Generated from FCC lattice: positions are FCC lattice points within a truncated octahedron

def _fcc_truncated_octahedron():
    """Generate FCC truncated octahedron coordinates for 38 atoms."""
    # FCC lattice points: (i,j,k) where i+j+k is even, scaled by 1/sqrt(2) for nearest neighbor = 1
    # Actually for LJ, we want nearest neighbor distance = 2^(1/6) ≈ 1.12246
    # FCC nearest neighbor distance in lattice units = a/sqrt(2), so a = sqrt(2) * 2^(1/6)
    
    # Let's use the standard FCC construction
    # FCC basis vectors: a1=(0,a,a), a2=(a,0,a), a3=(a,a,0) where a = 2^(1/6)/sqrt(2)
    # No wait, let me think more carefully.
    
    # In FCC, the nearest neighbor distance is a/sqrt(2) where a is the cubic lattice constant.
    # We want nearest neighbor = 2^(1/6) for LJ minimum.
    # So a/sqrt(2) = 2^(1/6) => a = sqrt(2) * 2^(1/6) = 2^(1/2 + 1/6) = 2^(2/3)
    
    a = 2.0 ** (2.0/3.0)  # cubic lattice constant
    
    # FCC points: all (i*a, j*a, k*a) where i,j,k are integers and i+j+k is even
    # Plus the face centers which are already included if we use the standard FCC construction
    # Standard FCC: points at (i,j,k) where i+j+k even, in units of a
    
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if (i + j + k) % 2 == 0:
                    points.append((i*a, j*a, k*a))
    
    points = np.array(points)
    
    # Center the cluster
    center = points.mean(axis=0)
    points -= center
    
    # Sort by distance from center
    dists = np.sqrt(np.sum(points**2, axis=1))
    indices = np.argsort(dists)
    points = points[indices]
    
    # Take the 38 closest
    points = points[:38]
    
    return points


def _lj_energy(pos):
    """Compute LJ energy and gradient."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) over i<j
    E = 4.0 * np.sum(r12 - r6) / 2.0  # divide by 2 for double counting
    
    # Gradient
    # dE/dr = 4*(-12*r^-13 + 6*r^-7) = 24*r^-7 - 48*r^-13
    # dE/dpos_i = sum_j dE/dr_ij * d r_ij/dpos_i = sum_j (24*r^-7 - 48*r^-13) * (pos_i - pos_j)/r
    # = sum_j (24*r^-6 - 48*r^-12) * (pos_i - pos_j) / r^2 * r ... let me redo
    
    # r_ij = |pos_i - pos_j|
    # dE/dpos_i = sum_{j!=i} 4*(-12*r_ij^-13 + 6*r_ij^-7) * (pos_i - pos_j)/r_ij
    # = sum_{j!=i} 4*(-12*r_ij^-14 + 6*r_ij^-8) * (pos_i - pos_j)
    # = sum_{j!=i} (24*r_ij^-8 - 48*r_ij^-14) * (pos_i - pos_j)
    
    # In terms of r2: r^-8 = r2^-4, r^-14 = r2^-7
    # factor = 24*r2^-4 - 48*r2^-7 = 24*r2^-7*(r2^3 - 2)
    
    factor = 24.0 * r2 ** (-7.0) * (r2 ** 3 - 2.0)
    # factor[i,j] multiplies (pos_i - pos_j)
    
    grad = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return E, grad


def _cubic_lattice_seed():
    """Generate cubic lattice seed."""
    side = int(np.ceil(N ** (1/3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
    return np.array(points[:N], dtype=float) * 1.12


def _relax(pos, time_limit=2.0):
    """Relax positions using L-BFGS with a time limit."""
    start = time.time()
    try:
        res = minimize(
            _lj_energy,
            pos,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
        )
        return res.x
    except Exception:
        return pos


def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Start with the known FCC truncated octahedron
    best_pos = _fcc_truncated_octahedron()
    best_energy, _ = _lj_energy(best_pos)
    
    # Relax it
    if time.time() < deadline - 5:
        relaxed = _relax(best_pos, time_limit=5.0)
        e, _ = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed
    
    # Also try from cubic lattice seed with large random perturbations
    seed = _cubic_lattice_seed()
    
    rng = np.random.RandomState(42)
    
    # Differential-evolution-like search: perturb best with large displacements
    n_trials = 30
    for trial in range(n_trials):
        if time.time() > deadline - 10:
            break
        
        # Generate candidate by perturbing the best solution with large random displacement
        # Scale: jump over icosahedral barrier (~a few units in energy, ~0.5-1.0 in position)
        scale = 0.8 + 0.4 * rng.random()
        candidate = best_pos + scale * rng.randn(*best_pos.shape)
        
        # Also try perturbing the cubic seed
        if trial % 3 == 0:
            candidate = seed + 0.5 * rng.randn(*seed.shape)
        
        # Relax the candidate
        if time.time() > deadline - 5:
            break
        
        relaxed = _relax(candidate, time_limit=3.0)
        e, _ = _lj_energy(relaxed)
        
        if e < best_energy:
            best_energy = e
            best_pos = relaxed
    
    # Final high-precision polish
    if time.time() < deadline - 3:
        final = _relax(best_pos, time_limit=5.0)
        e, _ = _lj_energy(final)
        if e < best_energy:
            best_energy = e
            best_pos = final
    
    # Ensure minimum separation >= 0.5
    # Check and fix if needed
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    if min_dist < 0.5:
        # Simple fix: nudge atoms apart
        for _ in range(100):
            diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = dists.min()
            if min_dist >= 0.5:
                break
            # Find closest pair
            idx = np.unravel_index(dists.argmin(), dists.shape)
            i, j = idx
            direction = best_pos[i] - best_pos[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.array([1.0, 0, 0])
            best_pos[i] += 0.01 * direction
            best_pos[j] -= 0.01 * direction
    
    return best_pos
# EVOLVE-BLOCK-END
