"""
LJ38 global minimum search using exact FCC truncated-octahedron construction 
and L-BFGS relaxation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE_MARGIN = 10.0

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _relax_lbfgs(pos, maxiter=2000):
    """L-BFGS relaxation."""
    pos = pos.copy()
    
    def fun_and_grad(x):
        x = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(x)
        return e, g.ravel()
    
    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B',
                       jac=True, options={'maxiter': maxiter, 'gtol': 1e-10})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]

def _make_fcc_truncated_octahedron():
    """
    Construct the 38-atom FCC truncated octahedron.
    This is the global minimum for LJ38.
    We generate FCC lattice points and select the 38 closest to the origin.
    """
    # LJ minimum distance r_min = 2^(1/6)
    r_min = 2.0 ** (1.0 / 6.0)
    # FCC lattice constant a such that nearest neighbor distance is r_min
    # Nearest neighbors in FCC are at distance a/sqrt(2)
    a = r_min * np.sqrt(2.0)
    
    points = []
    # Generate FCC points in a cubic box centered at origin
    # Range -3 to 3 should be more than enough for 38 atoms
    # FCC points in conventional cell:
    # (i, j, k)
    # (i+0.5, j+0.5, k)
    # (i+0.5, j, k+0.5)
    # (i, j+0.5, k+0.5)
    # where i, j, k are integers.
    
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # Type 1: Corners
                p1 = a * np.array([float(i), float(j), float(k)])
                # Type 2: Face centers
                p2 = a * np.array([float(i) + 0.5, float(j) + 0.5, float(k)])
                p3 = a * np.array([float(i) + 0.5, float(j), float(k) + 0.5])
                p4 = a * np.array([float(i), float(j) + 0.5, float(k) + 0.5])
                
                points.extend([p1, p2, p3, p4])
                
    points = np.array(points)
    
    # Remove duplicates (due to overlapping generation from adjacent cells)
    # Round to avoid floating point issues in unique
    points = np.round(points, decimals=10)
    points = np.unique(points, axis=0)
    
    # Calculate distance from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance and take the first 38
    indices = np.argsort(dists)[:N]
    pos = points[indices]
    
    # Center the cluster
    pos -= np.mean(pos, axis=0)
    
    return pos

def construct():
    t_start = time.time()
    deadline = t_start + 150.0 - DEADLINE_MARGIN
    
    best_pos = None
    best_energy = np.inf
    
    # Primary strategy: Exact FCC truncated octahedron
    try:
        pos_fcc = _make_fcc_truncated_octahedron()
        pos_relaxed, e = _relax_lbfgs(pos_fcc, maxiter=5000)
        if e < best_energy:
            best_energy = e
            best_pos = pos_relaxed.copy()
    except Exception as ex:
        print(f"FCC construction or relaxation failed: {ex}")
        
    if time.time() > deadline:
        if best_pos is not None:
            return best_pos
        # Fallback if everything fails
        return np.random.randn(N, 3) * 1.0

    # Secondary strategy: Perturbed FCC (in case of numerical issues or symmetry breaking)
    # This is usually not needed if the exact construction is good, but adds robustness.
    rng = np.random.RandomState(42)
    for i in range(5):
        if time.time() > deadline:
            break
        pos_pert = _make_fcc_truncated_octahedron() + rng.randn(N, 3) * 0.01
        pos_relaxed, e = _relax_lbfgs(pos_pert, maxiter=2000)
        if e < best_energy:
            best_energy = e
            best_pos = pos_relaxed.copy()

    # Tertiary strategy: Random starts (unlikely to beat FCC, but good for safety)
    for i in range(10):
        if time.time() > deadline:
            break
        pos_rand = rng.randn(N, 3) * 2.0
        pos_relaxed, e = _relax_lbfgs(pos_rand, maxiter=1000)
        if e < best_energy:
            best_energy = e
            best_pos = pos_relaxed.copy()

    if best_pos is None:
        # Last resort fallback
        return np.random.randn(N, 3) * 1.0

    return best_pos
