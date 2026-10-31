"""Symmetry-first LJ38 cluster: FCC truncated octahedron with constrained relaxation."""
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

def _relax(pos, max_iter=1000, tol=1e-10):
    """L-BFGS minimization with gradient."""
    res = minimize(
        _lj_energy,
        pos,
        jac=_lj_grad,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    return res.x

def _make_fcc_truncated_octahedron():
    """Generate the 38-atom FCC truncated octahedron with explicit shell structure.
    
    The FCC truncated octahedron for LJ38 has:
    - 1 atom at center (shell 0)
    - 12 atoms at first neighbor positions (shell 1) - octahedral face centers
    - 6 atoms at second neighbor positions (shell 2) - cubic face centers  
    - 19 atoms at third neighbor positions (shell 3) - mixed positions
    
    Using FCC lattice with conventional cell: atoms at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    in fractional coordinates of the cubic cell.
    """
    # FCC lattice vectors (conventional cubic cell, a=1)
    # Atoms in the unit cell at: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    
    # Generate FCC points in a cube around origin
    # For a truncated octahedron, we need points within a certain radius
    # The 38 atoms of the FCC truncated octahedron:
    # Shell 0: (0,0,0) - 1 atom
    # Shell 1: permutations of (±0.5, ±0.5, 0) and (±0.5, 0, ±0.5) and (0, ±0.5, ±0.5) - 12 atoms
    # Shell 2: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1) - 6 atoms
    # Shell 3: remaining 19 atoms
    
    # Let's generate all FCC points within a radius and pick the closest 38
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC: add all four atom positions in the unit cell
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x, y, z = i + dx, j + dy, k + dz
                    points.append([x, y, z])
    
    pts = np.array(points, dtype=float)
    # Remove duplicates
    pts = np.unique(np.round(pts, 10), axis=0)
    
    # Sort by distance to origin
    dists = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first 38 points
    cluster = pts[sorted_indices[:N]].copy()
    
    # Center at origin
    cluster -= np.mean(cluster, axis=0)
    
    return cluster

def _optimize_scale(pos):
    """Optimize the global scale factor for minimum LJ energy."""
    def scale_energy(log_s):
        s = np.exp(log_s)
        return _lj_energy(pos * s)
    
    # Initial guess: r_min = 2^(1/6), nearest neighbor in FCC ~ sqrt(0.5)
    # So scale ~ r_min / sqrt(0.5)
    r_min = 2.0 ** (1.0 / 6.0)
    # Estimate typical nearest neighbor distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    nn_dists = np.min(r2, axis=1)
    avg_nn = np.mean(nn_dists) ** 0.5
    init_log_s = np.log(r_min / avg_nn)
    
    res = minimize(scale_energy, [init_log_s], method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
    return np.exp(res.x[0])

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairwise distances are at least min_sep."""
    for _ in range(100):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        r = np.sqrt(r2)
        mask = r < min_sep
        if not np.any(mask):
            break
        # Push apart atoms that are too close
        for i in range(N):
            for j in range(i+1, N):
                if mask[i, j]:
                    d = pos[j] - pos[i]
                    r = np.linalg.norm(d)
                    if r < 1e-10:
                        d = np.array([0.01, 0.01, 0.01])
                        r = np.linalg.norm(d)
                    d = d / r
                    push = (min_sep - r) / 2.0
                    pos[i] -= push * d
                    pos[j] += push * d
    return pos

def _basin_hop(pos, rng, n_hops=5, max_iter=300):
    """Simple basin hopping: perturb, relax, keep if better."""
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    
    for _ in range(n_hops):
        # Random perturbation
        perturbed = best_pos + rng.normal(0, 0.1, size=(N, 3))
        perturbed = _ensure_min_sep(perturbed)
        try:
            relaxed = _relax(perturbed, max_iter=max_iter, tol=1e-10)
            e = _lj_energy(relaxed)
            if e < best_e:
                best_e = e
                best_pos = relaxed
        except Exception:
            continue
    
    return best_pos, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Step 1: Generate FCC truncated octahedron
    base_pos = _make_fcc_truncated_octahedron()
    
    # Step 2: Optimize scale
    scale = _optimize_scale(base_pos)
    pos = base_pos * scale
    pos = _ensure_min_sep(pos)
    
    # Step 3: Initial relaxation
    try:
        pos = _relax(pos, max_iter=500, tol=1e-10)
    except Exception:
        pass
    
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    
    if time.time() > deadline:
        return best_pos
    
    # Step 4: Basin hopping with multiple restarts
    rng = np.random.RandomState(42)
    
    # Try several perturbation strengths
    for attempt in range(20):
        if time.time() > deadline - 15.0:
            break
        
        # Vary perturbation strength
        strength = 0.05 + 0.15 * rng.random()
        perturbed = best_pos + rng.normal(0, strength, size=(N, 3))
        perturbed = _ensure_min_sep(perturbed)
        
        try:
            relaxed = _relax(perturbed, max_iter=400, tol=1e-10)
            e = _lj_energy(relaxed)
            if e < best_e:
                best_e = e
                best_pos = relaxed
        except Exception:
            continue
    
    # Step 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            polished = _relax(best_pos, max_iter=3000, tol=1e-13)
            e = _lj_energy(polished)
            if e < best_e:
                best_pos = polished
                best_e = e
        except Exception:
            pass
    
    # Ensure minimum separation
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
