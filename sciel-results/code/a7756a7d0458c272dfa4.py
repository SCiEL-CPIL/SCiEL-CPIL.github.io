import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 38
TIME_LIMIT = 120.0
SAFETY_MARGIN = 15.0

try:
    import science_chemistry
    HAS_RELAX = True
except ImportError:
    HAS_RELAX = False


def _lj_energy_and_grad(pos):
    """
    Compute Lennard-Jones energy and gradient for a cluster of N atoms.
    pos: (N, 3) array of coordinates
    Returns: (energy, gradient) where gradient is (N, 3)
    """
    # Pairwise displacement vectors: diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    
    # Pairwise distances
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    
    # Mask for self-interactions
    mask = np.eye(N, dtype=bool)
    dists[mask] = 1.0  # Avoid division by zero; will be zeroed out later
    
    # Inverse powers
    inv_r2 = 1.0 / (dists ** 2)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy term: 4 * (r^-12 - r^-6)
    term = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(term, 0.0)
    energy = 0.5 * np.sum(term)
    
    # Gradient: grad_i = sum_j 24 * (x_i - x_j) * (r_ij^-8 - 2 * r_ij^-14)
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    coeff = 24.0 * (inv_r8 - 2.0 * inv_r14)
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _make_fcc_truncated_octahedron():
    """
    Construct the FCC truncated octahedron geometry for LJ38.
    The global minimum of LJ38 is an FCC-based truncated octahedron.
    """
    # FCC lattice constant for LJ: a = 2^(2/3) ≈ 1.5874
    a = 2.0 ** (2.0 / 3.0)
    
    # FCC basis vectors
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    # Generate a 4x4x4 block of FCC unit cells
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    points.append(p * a)
    
    points = np.array(points)
    
    # Center the structure
    center = np.mean(points, axis=0)
    points = points - center
    
    # Sort by distance to center and take the 38 closest
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    return points[idx[:N]]


def _relax(pos, max_iter=3000, gtol=1e-10, ftol=1e-12):
    """
    Perform L-BFGS-B local relaxation.
    Returns: (relaxed_positions, energy)
    """
    x0 = pos.flatten()
    
    def fun(x):
        p = x.reshape(N, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.flatten()
    
    res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50})
    
    relaxed_pos = res.x.reshape(N, 3)
    energy = res.fun
    
    return relaxed_pos, energy


def _ensure_min_sep(pos, min_sep=0.5):
    """
    Ensure all atoms are at least min_sep apart by moving violating pairs apart.
    """
    pos = pos.copy()
    for _ in range(100):  # Max iterations
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_sep:
                    if d < 1e-10:
                        # Random separation
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                    else:
                        direction = (pos[i] - pos[j]) / d
                    # Move each atom half the required distance
                    needed = (min_sep - d) / 2.0
                    pos[i] += needed * direction
                    pos[j] -= needed * direction
                    changed = True
        if not changed:
            break
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # 1. Construct initial FCC truncated octahedron geometry
    fcc_pos = _make_fcc_truncated_octahedron()
    
    best_pos = fcc_pos.copy()
    best_energy = float('inf')
    
    # 2. Relax the FCC structure
    try:
        relaxed_pos, e = _relax(fcc_pos, max_iter=3000, gtol=1e-10, ftol=1e-12)
        relaxed_pos = _ensure_min_sep(relaxed_pos)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed_pos
    except Exception:
        pass
    
    # 3. High-precision polish
    if time.time() < deadline - 5.0:
        try:
            polished_pos, e = _relax(best_pos, max_iter=5000, gtol=1e-14, ftol=1e-15)
            polished_pos = _ensure_min_sep(polished_pos)
            if e < best_energy:
                best_energy = e
                best_pos = polished_pos
        except Exception:
            pass
    
    # 4. Final micro-polish
    if time.time() < deadline - 2.0:
        try:
            micro_pos, e = _relax(best_pos, max_iter=1000, gtol=1e-16, ftol=1e-16)
            micro_pos = _ensure_min_sep(micro_pos)
            if e < best_energy:
                best_energy = e
                best_pos = micro_pos
        except Exception:
            pass
    
    # Ensure final positions satisfy minimum separation
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
