"""
LJ38 cluster global minimum search.

Combines the direct FCC truncated-octahedron construction (from the current 
program) with a Basin Hopping global optimization strategy (inspired by the 
higher-scoring solution). This approach starts from a physically motivated 
structure and uses stochastic perturbations followed by local relaxation to 
escape local minima and converge to the global minimum.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin within 150s limit

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    # coeff_ij = 4 * (6/r8 - 12/r14)
    # grad_i = sum_j coeff_ij * (pos_i - pos_j)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron geometry for LJ38.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]

    # Take the 38 closest to the origin
    pts = pts[:N].copy()

    # Centre
    pts -= np.mean(pts, axis=0)

    # Scale so that the nearest-neighbour distance is close to 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale

    return pts

def _relax(pos, max_iter=500, gtol=1e-8, ftol=1e-12):
    """L-BFGS-B local relaxation with analytic gradient."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol, "maxls": 50},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        e, _ = _lj_energy_and_grad(pos)
        return pos, e

def _polish(pos, max_iter=1000):
    """High-precision polish pass."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    
    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": 1e-12, "ftol": 1e-15, "maxls": 50},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        e, _ = _lj_energy_and_grad(pos)
        return pos, e

def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    
    # 1. Initial Geometry: FCC Truncated Octahedron
    try:
        pos0 = _make_fcc_truncated_octahedron()
    except Exception:
        # Fallback to cubic lattice
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pos0 = np.array(pts[:N], dtype=np.float64)
        pos0 -= np.mean(pos0, axis=0)
        pos0 *= 1.12

    # Initial relaxation to get a good starting point
    best_pos, best_energy = _relax(pos0, max_iter=300)
    best_pos = _ensure_min_sep(best_pos)
    best_energy, _ = _lj_energy_and_grad(best_pos)

    # 2. Basin Hopping Global Search
    rng = np.random.default_rng(42)
    
    # Parameters
    step_size = 0.4
    T = 1.0  # Temperature for Metropolis acceptance
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    # Adaptive step size logic: if we keep accepting, we might be in a good basin, 
    # but we want to explore. If we keep rejecting, we might be stuck or T is too low.
    # For simplicity and robustness in a short time budget, we use a fixed step size 
    # that is large enough to jump between local minima but small enough to 
    # frequently land in valid basins.
    
    while time.time() - t0 < DEADLINE - 10.0:
        # Perturb
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        new_pos, new_energy = _relax(perturbed, max_iter=300, gtol=1e-6, ftol=1e-10)
        
        # Check separation constraint
        new_pos = _ensure_min_sep(new_pos)
        new_energy, _ = _lj_energy_and_grad(new_pos)
        
        # Acceptance Criterion
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            # Better energy, accept
            current_pos = new_pos
            current_energy = new_energy
            
            # Update global best
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            # Worse energy, accept with probability exp(-delta_e / T)
            # Use a slightly higher T to encourage exploration
            if rng.random() < np.exp(-delta_e / T):
                current_pos = new_pos
                current_energy = new_energy
                
        # Optional: Adapt step size slightly
        # If we found a new global best, we might reduce step size to refine locally?
        # Or keep it constant to ensure we don't get stuck.
        # For now, constant step size is robust.

    # 3. Final High-Precision Polish
    if time.time() - t0 < DEADLINE - 2.0:
        try:
            p, e = _polish(best_pos, max_iter=1000)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
