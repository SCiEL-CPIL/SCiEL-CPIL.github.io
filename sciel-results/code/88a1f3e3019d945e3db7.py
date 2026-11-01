# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 55

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def get_mackay_directions():
    """
    Get the 55 unit direction vectors for a Mackay icosahedron.
    Returns:
      directions: (55, 3) array of unit vectors (first is [0,0,0] for center)
      shell_ids: (55,) array of shell indices (0=center, 1=inner verts, 2=midpoints, 3=outer verts)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Build direction array and shell assignments
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    # Atom 0: center
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        directions[1+i] = verts[i]
        shell_ids[1+i] = 1
    
    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        directions[13+i] = midpoints[i]
        shell_ids[13+i] = 2
    
    # Atoms 43-54: Shell 3 (icosahedron vertices, outer)
    for i in range(12):
        directions[43+i] = verts[i]
        shell_ids[43+i] = 3
    
    return directions, shell_ids

def make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    directions, shell_ids = get_mackay_directions()
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = directions[i] * radii[0]
        elif shell == 2:
            pos[i] = directions[i] * radii[1]
        elif shell == 3:
            pos[i] = directions[i] * radii[2]
    return pos

def relax(pos, maxiter=300, gtol=1e-6):
    """Relax using L-BFGS-B with analytical gradient."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def optimize_radii_and_relax(radii_init, maxiter_nm=80, maxiter_lbfgs=200, deadline=None):
    """
    Optimize shell radii via Nelder-Mead, where each evaluation relaxes positions.
    Returns best position, energy, and radii.
    """
    def objective(radii):
        if deadline is not None and time.time() > deadline - 5.0:
            return 1e10
        radii_clipped = np.maximum(radii, 0.5)
        pos = make_mackay_from_radii(radii_clipped)
        pos_relaxed, e = relax(pos, maxiter=maxiter_lbfgs, gtol=1e-5)
        return e
    
    res = minimize(objective, radii_init, method='Nelder-Mead', 
                   options={'maxiter': maxiter_nm, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_radii = np.maximum(res.x, 0.5)
    best_pos = make_mackay_from_radii(best_radii)
    best_pos, best_e = relax(best_pos, maxiter=maxiter_lbfgs, gtol=1e-6)
    
    return best_pos, best_e, best_radii

def make_perturbed_mackay(base_pos, seed, scale=0.05):
    """Generate a perturbed version of a base position."""
    rng = np.random.RandomState(seed)
    pos = base_pos + rng.randn(N, 3) * scale
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Correct initial radius guesses based on known optimal Mackay icosahedron
    # r_inner ≈ 1.04, r_mid ≈ 1.76, r_outer ≈ 1.98 (outer slightly compressed)
    radius_candidates = [
        np.array([1.04, 1.76, 1.98]),   # Best known
        np.array([1.05, 1.75, 1.95]),
        np.array([1.00, 1.70, 1.90]),
        np.array([1.10, 1.80, 2.00]),
        np.array([1.04, 1.72, 1.98]),   # Slightly smaller mid
        np.array([1.04, 1.80, 1.98]),   # Slightly larger mid
        np.array([1.02, 1.74, 1.96]),
        np.array([1.06, 1.78, 2.00]),
    ]
    
    best_pos = make_mackay_from_radii(radius_candidates[0])
    best_e = lj_energy(best_pos)
    
    # Stage 1: Optimize radii from multiple starting points
    for radii_init in radius_candidates:
        if time.time() > deadline - 20:
            break
        try:
            pos, e, radii = optimize_radii_and_relax(radii_init, maxiter_nm=60, maxiter_lbfgs=150, deadline=deadline)
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # Stage 2: Structured perturbation search with small angular displacements
    # Instead of random perturbations, try small rotations of shells
    perturbation_seeds = list(range(12))
    for s in perturbation_seeds:
        if time.time() > deadline - 10:
            break
        
        # Small angular perturbation: rotate each shell by a small angle
        rng = np.random.RandomState(s)
        pos_pert = best_pos.copy()
        
        # Apply small random rotation to non-center atoms
        angle = rng.uniform(0.01, 0.05)
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        
        # Rodrigues rotation formula
        c = np.cos(angle)
        s_ = np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        R = c * np.eye(3) + s_ * K + (1 - c) * np.outer(axis, axis)
        
        # Apply rotation to atoms 1-54 (keep center fixed)
        pos_pert[1:] = pos_pert[1:] @ R.T
        
        pos_relaxed, e_relaxed = relax(pos_pert, maxiter=250, gtol=1e-5)
        
        if e_relaxed < best_e:
            best_e = e_relaxed
            best_pos = pos_relaxed.copy()
    
    # Stage 3: Final high-precision polish with progressively tighter tolerances
    if time.time() < deadline - 5:
        # First polish pass
        pos_final, e_final = relax(best_pos, maxiter=500, gtol=1e-8)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
    
    if time.time() < deadline - 3:
        # Second polish pass with tighter tolerance
        pos_final, e_final = relax(best_pos, maxiter=1000, gtol=1e-10)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
    
    if time.time() < deadline - 1:
        # Third polish pass with very tight tolerance
        pos_final, e_final = relax(best_pos, maxiter=2000, gtol=1e-12)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
    
    return best_pos

# EVOLVE-BLOCK-END
