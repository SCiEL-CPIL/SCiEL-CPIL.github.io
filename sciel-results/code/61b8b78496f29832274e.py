"""LJ38 cluster global minimum - FCC truncated octahedron construction with L-BFGS relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # Energy: 4 * sum over pairs (r^-12 - r^-6)
    # Using upper triangle to avoid double counting
    energy = 4.0 * np.sum(inv_r12 - inv_r6) - 4.0 * np.trace(inv_r12 - inv_r6)
    # Actually let's be more careful: sum over all i<j
    iu = np.triu_indices(n, k=1)
    energy = 4.0 * np.sum((inv_r12[iu] - inv_r6[iu]))
    
    # Gradient: dE/dpos_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (pos_i - pos_j) / r
    # = sum_j 4*(-12*inv_r14 + 6*inv_r8) * diff[i,j]
    # inv_r8 = inv_r6 * inv_r2, inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    inv_r14 = inv_r12 * inv_r2
    force_factor = 4.0 * (-12.0 * inv_r14 + 6.0 * inv_r8)  # shape (n, n)
    np.fill_diagonal(force_factor, 0.0)
    grad = np.sum(force_factor[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _generate_fcc_truncated_octahedron():
    """Generate FCC lattice points and select 38 closest to origin."""
    # FCC lattice: points at (i,j,k) where i+j+k is even, scaled by a/sqrt(2)
    # Generate a cubic region of FCC points
    max_coord = 6
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    # FCC nearest neighbor distance is sqrt(2) in these coords
    # Scale so nearest neighbor = 1.12 (LJ minimum)
    # In FCC, nearest neighbors are at distance sqrt(2)
    scale = 1.12 / np.sqrt(2.0)
    pts *= scale
    # Sort by distance from origin
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    # Select 38 closest
    return pts[:38].copy()


def construct():
    global _start_time
    _start_time = time.time()
    
    # Generate initial FCC truncated octahedron
    pos = _generate_fcc_truncated_octahedron()
    
    # Verify minimum distance constraint
    def _min_dist_check(p):
        d = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r = np.sqrt(np.sum(d ** 2, axis=2))
        np.fill_diagonal(r, np.inf)
        return np.min(r)
    
    if _min_dist_check(pos) < 0.5:
        # Scale up if needed
        scale_factor = 0.5 / _min_dist_check(pos) * 1.01
        pos *= scale_factor
    
    best_pos = pos.copy()
    best_energy = _lj_energy_and_grad(pos)[0]
    
    # L-BFGS relaxation
    def energy_func(p):
        return _lj_energy_and_grad(p)[0]
    
    def grad_func(p):
        return _lj_energy_and_grad(p)[1]
    
    try:
        result = minimize(
            fun=energy_func,
            x0=pos,
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
        )
        if result.success or result.fun < best_energy:
            best_pos = result.x.copy()
            best_energy = result.fun
    except Exception:
        pass
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed < _DEADLINE - 5:
        # Try a second refinement with tighter tolerance
        try:
            result2 = minimize(
                fun=energy_func,
                x0=best_pos,
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-16}
            )
            if result2.fun < best_energy:
                best_pos = result2.x.copy()
                best_energy = result2.fun
        except Exception:
            pass
    
    # Final check: ensure minimum distance >= 0.5
    md = _min_dist_check(best_pos)
    if md < 0.5:
        # Nudge apart any too-close pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = best_pos[i] - best_pos[j]
                r = np.linalg.norm(d)
                if r < 0.5 and r > 1e-10:
                    direction = d / r
                    shift = (0.5 - r) / 2.0 + 0.01
                    best_pos[i] += direction * shift
                    best_pos[j] -= direction * shift
                elif r <= 1e-10:
                    best_pos[i] += np.array([0.01, 0, 0])
    
    return best_pos
# EVOLVE-BLOCK-END
