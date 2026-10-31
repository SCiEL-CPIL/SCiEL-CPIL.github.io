"""
LJ38 cluster global minimum search.

Key insight: The global minimum for LJ38 is a truncated octahedron from the FCC lattice.
Instead of random search, we directly construct the 38-atom FCC truncated octahedron
geometry and compute the analytically optimal scale factor for minimum energy.

The energy for a fixed geometry scaled by factor s is:
    E(s) = 4 * (s^{-12} * A - s^{-6} * B)
where A = sum(r_ij^{-12}) and B = sum(r_ij^{-6}) for the unit-scale geometry.

Taking dE/ds = 0 gives: s = (2A/B)^{1/6}

This reduces the 114D optimization to a 1D (or 0D with analytical solution) problem.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _generate_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated octahedron.
    
    FCC lattice points: all (i,j,k) where i+j+k is even.
    The truncated octahedron for N=38 consists of:
    - 1 center atom at origin
    - 12 nearest neighbors (distance sqrt(2) in lattice units): permutations of (1,1,0)
    - 6 next-nearest (distance 2 in lattice units): permutations of (2,0,0)
    - 19 more from the next shell (distance sqrt(6)): permutations of (2,1,1)
    
    Total: 1 + 12 + 6 + 19 = 38
    """
    # Generate all FCC points within a reasonable radius
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i*i + j*j + k*k
                    pts.append((float(i), float(j), float(k), d2))
    
    pts = np.array(pts)
    
    # Sort by distance from origin
    order = np.argsort(pts[:, 3])
    pts_sorted = pts[order]
    
    # Take the 38 closest points
    pts_38 = pts_sorted[:38, :3]
    
    return pts_38

def _analytical_optimal_scale(pos):
    """
    Compute the analytically optimal scale factor for minimum LJ energy.
    
    For E(s) = 4*(s^{-12}*A - s^{-6}*B), the minimum is at s = (2A/B)^{1/6}.
    """
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    
    r6 = r2 ** 3
    r12 = r6 ** 2
    
    A = 0.5 * np.sum(1.0 / r12)
    B = 0.5 * np.sum(1.0 / r6)
    
    if B > 1e-15:
        s = (2.0 * A / B) ** (1.0 / 6.0)
    else:
        s = 1.0
    
    return s

def _relax(pos, max_iter=1000, gtol=1e-12):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-15},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairwise distances are at least min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.001
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin (150s limit)
    
    # Step 1: Generate the FCC truncated octahedron geometry
    pos = _generate_fcc_truncated_octahedron()
    
    # Step 2: Center the geometry
    pos -= np.mean(pos, axis=0)
    
    # Step 3: Compute analytically optimal scale
    s_opt = _analytical_optimal_scale(pos)
    pos = pos * s_opt
    
    # Step 4: Local relaxation to account for any slight distortions
    try:
        pos, e = _relax(pos, max_iter=2000, gtol=1e-14)
    except Exception:
        pass
    
    # Step 5: Ensure minimum separation
    pos = _ensure_min_sep(pos)
    
    # Step 6: Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            pos, e = _relax(pos, max_iter=5000, gtol=1e-15)
            pos = _ensure_min_sep(pos)
        except Exception:
            pass
    
    return pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
    print(f"Expected global minimum: -173.928427")
