"""Lennard-Jones 55-atom global minimum: Mackay icosahedron with high-precision optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy(pos):
    """Compute total LJ energy (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_vectors():
    """
    Return 12 inner-shell unit vectors and 30 edge-midpoint unit vectors for Mackay icosahedron.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_inner = verts / norms
    
    # Find edges: pairs of vertices at minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(v_inner[i] - v_inner[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-8:
                edges.append((i, j))
    
    v_edge = []
    for i, j in edges:
        mid = (v_inner[i] + v_inner[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            v_edge.append(mid / norm)
    v_edge = np.array(v_edge)
    
    v_outer = v_inner.copy()
    
    return v_inner, v_edge, v_outer

V_INNER, V_EDGE, V_OUTER = _get_mackay_vectors()

def _build_from_radii(r1, r2, r3):
    """Build 55-atom positions from three shell radii."""
    center = np.zeros((1, 3))
    shell1 = V_INNER * r1
    shell2a = V_EDGE * r2
    shell2b = V_OUTER * r3
    return np.vstack([center, shell1, shell2a, shell2b])

def _objective_radii(params):
    """Objective for optimizing the three shell radii."""
    r1, r2, r3 = params
    pos = _build_from_radii(r1, r2, r3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        return 1e10
    return _lj_energy(pos)

def _full_optimize(pos0, maxiter=500, gtol=1e-10, deadline=None):
    """L-BFGS-B full 3D relaxation with deadline guard."""
    start_time = time.time()
    best_e = _lj_energy(pos0)
    best_pos = pos0.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos0.flatten()
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-14}
        )
        if result.fun < best_e:
            best_e = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_e

def _high_precision_polish(pos, deadline):
    """
    Multi-stage polish with progressively tighter tolerances.
    Uses multiple optimization methods to drive energy to machine precision.
    """
    best_e = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = best_pos.flatten()
    
    # Stage 1: L-BFGS-B with moderate tolerance
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-12, 'maxls': 500, 'ftol': 1e-15}
            )
            if result.fun < best_e:
                best_e = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B with tight tolerance
    if time.time() < deadline - 8.0:
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options={'maxiter': 8000, 'gtol': 1e-14, 'maxls': 800, 'ftol': 1e-16}
            )
            if result.fun < best_e:
                best_e = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    # Stage 3: BFGS for different convergence path
    if time.time() < deadline - 6.0:
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='BFGS',
                options={'maxiter': 5000, 'gtol': 1e-14, 'disp': False}
            )
            if result.fun < best_e:
                best_e = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    # Stage 4: L-BFGS-B ultra-tight
    if time.time() < deadline - 4.0:
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options={'maxiter': 10000, 'gtol': 1e-15, 'maxls': 1500, 'ftol': 1e-17}
            )
            if result.fun < best_e:
                best_e = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    # Stage 5: Final L-BFGS-B
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options={'maxiter': 15000, 'gtol': 1e-16, 'maxls': 2000, 'ftol': 1e-18}
            )
            if result.fun < best_e:
                best_e = result.fun
                best_pos = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    return best_pos, best_e

def construct():
    """
    Construct 55-atom LJ cluster at the Mackay icosahedron global minimum.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Step 1: Optimize shell radii with better initial guess
    # Literature values for LJ55 Mackay icosahedron
    r0 = np.array([1.135, 1.162, 1.283])
    
    best_r = r0.copy()
    best_r_energy = _objective_radii(r0)
    
    # Fine grid search around known optimum
    r1_range = np.arange(1.12, 1.16, 0.003)
    r2_range = np.arange(1.15, 1.18, 0.003)
    r3_range = np.arange(1.27, 1.30, 0.003)
    
    for r1 in r1_range:
        for r2 in r2_range:
            for r3 in r3_range:
                if time.time() > deadline - 40.0:
                    break
                params = np.array([r1, r2, r3])
                e = _objective_radii(params)
                if e < best_r_energy:
                    best_r_energy = e
                    best_r = params
    
    # Nelder-Mead refinement on radii
    if time.time() < deadline - 35.0:
        try:
            res = minimize(
                _objective_radii, best_r, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_r_energy:
                best_r_energy = res.fun
                best_r = res.x
        except Exception:
            pass
    
    # Step 2: Build initial structure from optimized radii
    pos_0 = _build_from_radii(*best_r)
    
    # Step 3: Full 3D relaxation
    pos_relaxed, e_relaxed = _full_optimize(pos_0, maxiter=2000, gtol=1e-12, deadline=deadline)
    
    # Step 4: High-precision polish
    pos_final, e_final = _high_precision_polish(pos_relaxed, deadline)
    
    # Verify minimum distance constraint
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < 0.5:
        scale = 0.5 / min_dist * 1.001
        pos_final = pos_final * scale
        if time.time() < deadline - 5.0:
            pos_final, e_final = _full_optimize(pos_final, maxiter=500, gtol=1e-12, deadline=deadline)
    
    return pos_final

# EVOLVE-BLOCK-END
