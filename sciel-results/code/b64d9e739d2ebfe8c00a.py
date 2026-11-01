"""Lennard-Jones cluster global minimum, 55 atoms.
Strategy: build a Mackay icosahedron (2 shells: 1+12+42) with optimized radii,
then perform multi-stage relaxation with L-BFGS.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE = 145.0  # Safety margin before 150s kill


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions of shape (N,3)."""
    n = pos.shape[0]
    # Pairwise displacements
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n,n,3)
    r2 = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    # Using full matrix, each pair counted twice
    e_pair = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = 0.5 * np.sum(e_pair)
    
    # Gradient: dE/dr_i = sum_j 4 * (-12*r^-14 + 6*r^-8) * (r_i - r_j)
    # = sum_j 4 * (6*r^-8 - 12*r^-14) * (r_i - r_j)
    # Note: r^-14 = r^-12 / r^2, r^-8 = r^-6 / r^2
    coeff = 4.0 * (6.0 * inv_r6 - 12.0 * inv_r12) * inv_r2  # (n,n)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _relax(pos, maxiter=1000, gtol=1e-8):
    """L-BFGS relaxation using scipy."""
    def fun(x):
        p = x.reshape(-1, 3)
        e, _ = _lj_energy_and_grad(p)
        return e
    
    def jac(x):
        p = x.reshape(-1, 3)
        _, g = _lj_energy_and_grad(p)
        return g.ravel()
    
    res = minimize(fun, pos.ravel(), jac=jac, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': gtol})
    return res.x.reshape(-1, 3), res.fun


def _get_ico_directions():
    """Get 12 vertex directions and 30 edge-midpoint directions of icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Find all 30 edges (pairs of vertices at minimum distance)
    # For unit circumradius icosahedron, edge length = 4/sqrt(10+2*sqrt(5))
    edge_len = 4.0 / np.sqrt(10.0 + 2.0 * np.sqrt(5.0))
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - edge_len) < 1e-6:
                edges.append((i, j))
    
    # Get edge midpoint directions (normalized)
    edge_dirs = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        edge_dirs.append(mid)
    
    return verts, np.array(edge_dirs)


def _make_mackay(r_in, r_out):
    """Construct Mackay icosahedron with given inner and outer shell radii."""
    verts, edge_dirs = _get_ico_directions()
    
    atoms = [np.zeros(3)]  # center
    for v in verts:
        atoms.append(v * r_in)  # inner shell: 12
    for v in verts:
        atoms.append(v * r_out)  # outer shell vertices: 12
    for v in edge_dirs:
        atoms.append(v * r_out)  # outer shell edge midpoints: 30
    
    return np.array(atoms, dtype=float)


def _optimize_radii():
    """Optimize the two shell radii to minimize energy."""
    def energy_from_radii(x):
        r_in, r_out = x
        pos = _make_mackay(r_in, r_out)
        e, _ = _lj_energy_and_grad(pos)
        return e
    
    # Initial guess
    x0 = np.array([1.0, 1.7])
    res = minimize(energy_from_radii, x0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    best_r_in, best_r_out = res.x
    best_e = res.fun
    
    # Try a few different starting points
    for start in [(0.8, 1.5), (1.2, 2.0), (1.0, 1.8), (0.9, 1.6)]:
        try:
            r = minimize(energy_from_radii, np.array(start), method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if r.fun < best_e:
                best_r_in, best_r_out = r.x
                best_e = r.fun
        except Exception:
            pass
    
    return best_r_in, best_r_out, best_e


def construct():
    t_start = time.time()
    
    # Step 1: Optimize radii
    r_in, r_out, e_rad = _optimize_radii()
    
    # Step 2: Build initial structure with optimized radii
    pos0 = _make_mackay(r_in, r_out)
    best_pos = pos0.copy()
    best_energy = _lj_energy_and_grad(pos0)[0]
    
    # Step 3: Multi-stage relaxation
    # Stage 1: Coarse relaxation
    if time.time() - t_start < DEADLINE - 30:
        try:
            relaxed, e = _relax(pos0, maxiter=300, gtol=1e-8)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Stage 2: Medium precision
    if time.time() - t_start < DEADLINE - 15:
        try:
            relaxed, e = _relax(best_pos, maxiter=500, gtol=1e-10)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Step 4: Try several perturbed restarts
    rng = np.random.default_rng(42)
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() - t_start > DEADLINE - 10.0:
            break
        
        # Different perturbation strategies
        if i < 3:
            # Small Gaussian noise
            noise = 0.05
            perturbed = best_pos + rng.normal(0, noise, size=best_pos.shape)
        elif i < 6:
            # Radial perturbation (scale inner/outer shells)
            perturbed = best_pos.copy()
            # Center is atom 0, inner shell is atoms 1-12, outer is 13-54
            scale_in = 1.0 + rng.uniform(-0.05, 0.05)
            scale_out = 1.0 + rng.uniform(-0.05, 0.05)
            for j in range(1, 13):
                perturbed[j] *= scale_in
            for j in range(13, 55):
                perturbed[j] *= scale_out
        else:
            # Angular perturbation
            perturbed = best_pos.copy()
            for j in range(1, 55):
                # Small random rotation
                angle = rng.uniform(-0.05, 0.05)
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                # Rodrigues rotation
                c, s = np.cos(angle), np.sin(angle)
                k = np.array([axis[1]*axis[2], axis[2]*axis[0], axis[0]*axis[1]])
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + s * K + (1 - c) * (K @ K)
                perturbed[j] = R @ perturbed[j]
        
        # Ensure minimum separation
        for _ in range(20):
            diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
            r2 = np.sum(diffs * diffs, axis=2)
            np.fill_diagonal(r2, 1.0)
            if np.min(r2) >= 0.25:
                break
            # Push apart close pairs
            close = r2 < 0.25
            for idx in range(55):
                for jdx in range(idx + 1, 55):
                    if close[idx, jdx]:
                        d = perturbed[idx] - perturbed[jdx]
                        dist = np.linalg.norm(d)
                        if dist < 1e-8:
                            d = rng.normal(size=3) * 0.1
                        d /= np.linalg.norm(d)
                        perturbed[idx] += 0.5 * d
                        perturbed[jdx] -= 0.5 * d
        
        try:
            relaxed, e = _relax(perturbed, maxiter=200, gtol=1e-9)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Step 5: Final high-precision polish
    if time.time() - t_start < DEADLINE - 5.0:
        try:
            polished, e = _relax(best_pos, maxiter=2000, gtol=1e-12)
            if e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass
    
    return best_pos
