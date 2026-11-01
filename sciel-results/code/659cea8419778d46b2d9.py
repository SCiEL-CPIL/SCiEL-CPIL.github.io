"""Lennard-Jones cluster global minimum, 55 atoms — Mackay icosahedron with expanded radial manifold."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_MARGIN = 5.0  # seconds before the 150 s wall-clock kill
TOTAL_BUDGET = 150.0


def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for N atoms. Vectorized."""
    # pos: (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 ** 2

    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    # The matrix has each pair twice, so divide by 2
    pair_e = 4.0 * (r12 - r6)
    np.fill_diagonal(pair_e, 0.0)
    energy = 0.5 * np.sum(pair_e)

    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    # In terms of r2: r^14 = r2^7, r^8 = r2^4
    # dE/dx_i = 4*(-12*r2^-7 + 6*r2^-4) * diff[i,j]

    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)

    return energy, grad


def _get_icosahedral_directions():
    """
    Returns:
        verts: (12, 3) array of unit vectors to icosahedron vertices.
        edge_mids: (30, 3) array of unit vectors to icosahedron edge midpoints.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Icosahedral vertices (12)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Find edges: pairs of vertices with minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # The 30 edges have the smallest distances
    # There are 12*11/2 = 66 pairs. The 30 edges are the closest ones.
    # We can just find the 30 smallest distances.
    triu_indices = np.triu_indices(12, k=1)
    triu_dists = dists[triu_indices]
    sorted_indices = np.argsort(triu_dists)
    edge_pairs = list(zip(triu_indices[0][sorted_indices[:30]], triu_indices[1][sorted_indices[:30]]))
    
    edge_mids = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
    
    edge_mids = np.array(edge_mids)
    return verts, edge_mids


def _construct_mackay_from_radii(radii, verts, edge_mids):
    """
    Construct the 55-atom Mackay icosahedron from radii.
    radii: [r1, r2, r3]
    r1: radius of 12 vertex atoms (inner shell)
    r2: radius of 30 edge-midpoint atoms (middle shell)
    r3: radius of 12 vertex-direction atoms (outer shell)
    """
    r1, r2, r3 = radii
    
    # 1 center atom
    center = np.zeros((1, 3))
    
    # 12 atoms at vertex directions, radius r1
    shell1 = verts * r1
    
    # 30 atoms at edge midpoint directions, radius r2
    shell2 = edge_mids * r2
    
    # 12 atoms at vertex directions, radius r3
    shell3 = verts * r3
    
    all_pos = np.vstack([center, shell1, shell2, shell3])
    return all_pos


def _radial_energy_and_grad(radii, verts, edge_mids):
    """
    Compute energy and gradient with respect to radii (r1, r2, r3).
    """
    pos = _construct_mackay_from_radii(radii, verts, edge_mids)
    energy, grad_cart = _lj_energy_and_grad(pos)
    
    # grad_cart is (55, 3)
    # We need dE/dr_k = sum_{i in orbit k} grad_cart[i] . dir_i
    # Orbit 1 (center): 1 atom at (0,0,0). Gradient contribution to radial param? 
    # The center atom is fixed at origin in this manifold? 
    # Actually, the center atom's position is (0,0,0). If we allow it to move, we break symmetry.
    # In the standard Mackay, the center is fixed. 
    # However, the manifold defined by (r1, r2, r3) fixes the center at 0.
    # The gradient w.r.t r1 is sum of grad_cart[1:13] dot verts
    # The gradient w.r.t r2 is sum of grad_cart[13:43] dot edge_mids
    # The gradient w.r.t r3 is sum of grad_cart[43:55] dot verts
    
    grad_r1 = np.sum(grad_cart[1:13] * verts)
    grad_r2 = np.sum(grad_cart[13:43] * edge_mids)
    grad_r3 = np.sum(grad_cart[43:55] * verts)
    
    grad_r = np.array([grad_r1, grad_r2, grad_r3])
    
    return energy, grad_r


def construct():
    start_time = time.time()
    
    # Get geometric directions
    verts, edge_mids = _get_icosahedral_directions()
    
    # Try multiple initial guesses for radii and pick the best
    initial_guesses = [
        np.array([1.12, 2.24, 3.36]),
        np.array([1.10, 2.20, 3.30]),
        np.array([1.15, 2.25, 3.35]),
        np.array([1.08, 2.18, 3.28]),
        np.array([1.13, 2.26, 3.38]),
    ]
    
    best_pos = None
    best_energy = np.inf
    
    for r0 in initial_guesses:
        elapsed = time.time() - start_time
        if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
            break
        
        # Optimize radii
        try:
            res_radial = minimize(
                _radial_energy_and_grad,
                r0,
                args=(verts, edge_mids),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-12, 'maxls': 100}
            )
            pos = _construct_mackay_from_radii(res_radial.x, verts, edge_mids)
            e = _lj_energy_and_grad(pos)[0]
            
            if e < best_energy:
                best_energy = e
                best_pos = pos
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback to original guess
        best_pos = _construct_mackay_from_radii(np.array([1.12, 2.24, 3.36]), verts, edge_mids)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
        return best_pos
    
    # Polish with full Cartesian L-BFGS - first pass
    try:
        res_cart = minimize(
            _lj_energy_and_grad,
            best_pos,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 100}
        )
        if res_cart.fun < best_energy:
            best_pos = res_cart.x
            best_energy = res_cart.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
        return best_pos
    
    # Second polish pass with even tighter tolerance
    try:
        res_cart2 = minimize(
            _lj_energy_and_grad,
            best_pos,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-14, 'ftol': 1e-16, 'maxls': 200}
        )
        if res_cart2.fun < best_energy:
            best_pos = res_cart2.x
            best_energy = res_cart2.fun
    except Exception:
        pass
    
    # Final check: ensure atoms are at least 0.5 apart
    if best_pos is not None:
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, 1.0)
        min_dist = np.sqrt(np.min(r2[r2 > 0]))
        if min_dist < 0.5:
            # Fallback to radial-only solution
            best_pos = _construct_mackay_from_radii(np.array([1.12, 2.24, 3.36]), verts, edge_mids)
    
    return best_pos
