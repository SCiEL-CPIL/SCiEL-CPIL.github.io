"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_FRACTION = 0.85  # safety margin: stop at 85% of the 150s limit

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero on diagonal
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * (r12 - r6).sum()
    return energy

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * (r12 - r6).sum()
    # Gradient: dE/dr2 = 4*(-12*r12/r2 + 6*r6/r2) = 4*(-12*r12 + 6*r6)/r2
    # dE/dxi = 2 * sum_j dE/dr2_ij * (xi - xj) ... actually dE/dxi = 2 * (xi-xj) * dE/dr2
    # More precisely: E = sum_{i<j} 4(r_ij^-12 - r_ij^-6)
    # dE/dx_i = sum_{j!=i} 4*(-12*r_ij^-14 + 6*r_ij^-8) * (x_i - x_j)
    # Let's compute via r2: dE/dr2 = 4*(-12*r12/r2 + 6*r6/r2) = (4/r2)*(-12*r12 + 6*r6)
    # dE/dx_i = 2 * sum_j (dE/dr2_ij) * (x_i - x_j)
    dEdr2 = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2
    np.fill_diagonal(dEdr2, 0.0)
    grad = 2.0 * np.einsum('ij,ijk->ik', dEdr2, diff)
    return energy, grad

def _local_minimize(pos, maxiter=200):
    """L-BFGS local relaxation."""
    def fun(x):
        return _lj_energy(x)
    def jac(x):
        return _lj_energy_grad(x)[1]
    res = minimize(fun, pos, jac=jac, method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-8})
    return res.x, res.fun

def _make_fcc_truncated_octahedron():
    """
    Construct a face-centered-cubic truncated octahedron cluster for 38 atoms.
    FCC truncated octahedron with 38 atoms: this is the known global minimum structure.
    We build it by taking FCC lattice points within a certain radius.
    """
    a = 1.0  # lattice constant in reduced units (will be scaled)
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    # Generate FCC points in a cube from -2 to 2 in lattice units
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    pts = np.array(pts) * a
    # Center
    center = pts.mean(axis=0)
    pts = pts - center
    # Sort by distance from center
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    # Take 38 closest
    pts = pts[:38]
    # The FCC lattice constant for LJ minimum: nearest neighbor distance = a/sqrt(2)
    # We want nearest neighbor ~ 1.122 (2^(1/6))
    # So a = 1.122 * sqrt(2)
    a_target = 2.0 ** (1.0/6.0) * np.sqrt(2.0)
    pts = pts * (a_target / a)
    return pts

def _random_initial(n, scale=3.0):
    """Random initial configuration."""
    return np.random.randn(n, 3) * scale

def _perturb(pos, sigma=0.3):
    """Add random perturbation."""
    return pos + np.random.randn(*pos.shape) * sigma

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_energy = np.inf
    
    def try_config(pos, label=""):
        nonlocal best_pos, best_energy
        try:
            pos_relaxed, energy = _local_minimize(pos)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos_relaxed.copy()
        except Exception:
            pass
    
    # Strategy 1: Start from FCC truncated octahedron (known global min structure)
    try:
        fcc_pos = _make_fcc_truncated_octahedron()
        try_config(fcc_pos, "FCC")
    except Exception:
        pass
    
    # Strategy 2: Basin hopping from FCC and other seeds
    seeds = [fcc_pos] if best_pos is not None else []
    # Add a few random seeds
    for i in range(5):
        seeds.append(_random_initial(N, scale=2.0 + i * 0.5))
    
    # Basin hopping
    bh_sigma = 0.4
    bh_max_iter = 200
    
    for seed_idx, seed in enumerate(seeds):
        current_pos = seed.copy()
        current_energy = best_energy if seed_idx == 0 else _lj_energy(seed)
        
        for it in range(bh_max_iter):
            if time.time() > deadline:
                break
            
            # Perturb
            perturbed = _perturb(current_pos, bh_sigma)
            
            # Local minimize
            try:
                new_pos, new_energy = _local_minimize(perturbed, maxiter=100)
            except Exception:
                continue
            
            # Accept if lower energy (or with some probability for exploration)
            if new_energy < current_energy or (rng.random() < 0.1 and new_energy < current_energy + 5.0):
                current_pos = new_pos
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            else:
                # Keep current, maybe try smaller perturbation
                bh_sigma = max(0.1, bh_sigma * 0.98)
        
        if time.time() > deadline:
            break
    
    # If we haven't found anything, fall back to a simple cubic lattice
    if best_pos is None:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    # Final relaxation to ensure we're at a local minimum
    try:
        final_pos, final_energy = _local_minimize(best_pos, maxiter=500)
        best_pos = final_pos
    except Exception:
        pass
    
    return best_pos
# EVOLVE-BLOCK-END
