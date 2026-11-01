"""Lennard-Jones cluster global minimum, 55 atoms."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE = 135.0  # Safety margin before 150s kill

def _lj_energy(pos):
    """Compute total LJ energy for N atoms (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient (vectorized)."""
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

def _min_distance(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _get_mackay_directions():
    """
    Get the 55 unit direction vectors for a Mackay icosahedron M(2,2,2).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    for i in range(12):
        directions[1+i] = verts[i]
        shell_ids[1+i] = 1
    
    for i in range(30):
        directions[13+i] = midpoints[i]
        shell_ids[13+i] = 2
    
    for i in range(12):
        directions[43+i] = verts[i]
        shell_ids[43+i] = 3
    
    return directions, shell_ids

def _make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    directions, shell_ids = _get_mackay_directions()
    
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

def _compute_hessian(pos):
    """Compute the 165x165 Hessian matrix of the LJ energy."""
    n = len(pos)
    dim = n * 3
    H = np.zeros((dim, dim))
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    
    # Second derivatives of LJ potential
    # V = 4(r^-12 - r^-6)
    # dV/dr = 4(-12r^-13 + 6r^-7)
    # d2V/dr2 = 4(156r^-14 - 42r^-8)
    # 
    # Hessian element for atoms i,j in direction a,b:
    # H_ia, jb = d2V/dr2 * (r_a r_b / r2) + dV/dr * delta_ab / r  (for i!=j)
    # H_ia, ia = d2V/dr2 * sum_b (r_b^2 / r2) - dV/dr / r  (for i!=j, summed over j)
    
    r6 = r2 ** (-3)
    r12 = r6 * r6
    dr = 4.0 * (-12.0 * r2 ** (-6.5) + 6.0 * r2 ** (-3.5))
    d2r = 4.0 * (156.0 * r2 ** (-7) - 42.0 * r2 ** (-4))
    
    np.fill_diagonal(dr, 0.0)
    np.fill_diagonal(d2r, 0.0)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_ij = np.sqrt(r2[i, j])
            if r_ij < 1e-10:
                continue
            dr_ij = dr[i, j]
            d2r_ij = d2r[i, j]
            rij_vec = diff[i, j]
            
            # Off-diagonal block H[i*3:(i+1)*3, j*3:(j+1)*3]
            # = (d2V/dr2 / r2) * r_i r_j^T - (dV/dr / r) * I
            block = (d2r_ij / r2[i, j]) * np.outer(rij_vec, rij_vec) - (dr_ij / r_ij) * np.eye(3)
            H[i*3:(i+1)*3, j*3:(j+1)*3] = block
            
            # Diagonal block H[i*3:(i+1)*3, i*3:(i+1)*3] += -block
            H[i*3:(i+1)*3, i*3:(i+1)*3] -= block
    
    return H

def _relax(positions, tol=1e-6, maxiter=300):
    """Relax positions using L-BFGS-B."""
    def objective(x):
        pos = x.reshape(N, 3)
        e, g = _lj_energy_grad(pos)
        return e, g.flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'gtol': tol, 'maxiter': maxiter, 'maxls': 50, 'ftol': 1e-8})
        if result.success or result.fun < 1e6:
            return result.x.reshape(N, 3), result.fun
    except Exception:
        pass
    return positions, _lj_energy(positions)

def _perturb(positions, scale=0.1, seed=None):
    """Add small random perturbation."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    return positions + rng.randn(*positions.shape) * scale

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - elapsed()
    
    # Get Mackay directions
    directions, shell_ids = _get_mackay_directions()
    
    # Initial radius candidates
    radius_candidates = [
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.9, 1.7, 2.5]),
        np.array([1.0, 2.0, 2.8]),
        np.array([1.05, 1.85, 2.65]),
        np.array([1.0, 1.75, 2.55]),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Optimize radii for multiple initial guesses
    for radii_init in radius_candidates:
        if remaining() < 20:
            break
        try:
            pos = _make_mackay_from_radii(radii_init)
            pos_relaxed, e = _relax(pos, tol=1e-5, maxiter=200)
            if e < best_energy:
                best_energy = e
                best_positions = pos_relaxed.copy()
        except Exception:
            continue
    
    if best_positions is None:
        best_positions = _make_mackay_from_radii(np.array([1.0, 1.8, 2.6]))
        best_energy = _lj_energy(best_positions)
    
    # Phase 2: Basin-hopping with simulated annealing
    if remaining() > 15:
        rng = np.random.RandomState(42)
        current_energy = best_energy
        current_positions = best_positions.copy()
        T = 0.5  # Initial temperature
        
        max_bh_steps = 30
        bh_steps = 0
        for step in range(max_bh_steps):
            if remaining() < 10:
                break
            
            # Generate perturbation
            scale = 0.1 * (1.0 + 0.5 * np.sin(step * 0.3))
            perturbed = _perturb(current_positions, scale=scale, seed=42 + step)
            
            # Relax
            relaxed, e_relaxed = _relax(perturbed, tol=1e-5, maxiter=200)
            
            # Metropolis acceptance
            delta_e = e_relaxed - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_positions = relaxed
                current_energy = e_relaxed
                if e_relaxed < best_energy:
                    best_energy = e_relaxed
                    best_positions = relaxed.copy()
            
            # Cool down
            T *= 0.95
            bh_steps += 1
    
    # Phase 3: Normal mode search
    if remaining() > 20:
        try:
            # Compute Hessian at best position
            H = _compute_hessian(best_positions)
            
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(H)
            
            # Sort by eigenvalue (ascending)
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Remove translational and rotational modes (first 6 should be near zero)
            # Keep modes with eigenvalue > small threshold
            threshold = 1e-6
            soft_modes = []
            soft_eigvals = []
            for i in range(6, min(30, len(eigvals))):
                if eigvals[i] > threshold and eigvals[i] < 10.0:
                    soft_modes.append(eigvecs[:, i])
                    soft_eigvals.append(eigvals[i])
            
            if len(soft_modes) > 0:
                soft_modes = np.array(soft_modes).T  # (n_modes, 165)
                n_modes = soft_modes.shape[0]
                
                def normal_mode_objective(coeffs):
                    # coeffs: (n_modes,) array of displacement amplitudes
                    displacement = np.outer(coeffs, soft_modes).reshape(N, 3)
                    pos = best_positions + displacement
                    return _lj_energy(pos)
                
                # Optimize normal mode coefficients
                coeffs_init = np.zeros(n_modes)
                res = minimize(normal_mode_objective, coeffs_init, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if res.fun < best_energy:
                    displacement = np.outer(res.x, soft_modes).reshape(N, 3)
                    pos_nm = best_positions + displacement
                    pos_nm_relaxed, e_nm = _relax(pos_nm, tol=1e-6, maxiter=300)
                    if e_nm < best_energy:
                        best_energy = e_nm
                        best_positions = pos_nm_relaxed.copy()
                        
        except Exception:
            pass
    
    # Phase 4: Multi-start stochastic search
    if remaining() > 10:
        num_starts = 5
        for i in range(num_starts):
            if remaining() < 5:
                break
            
            scale = 0.05 * (i + 1)
            try:
                perturbed = _perturb(best_positions, scale=scale, seed=100 + i)
                relaxed, e = _relax(perturbed, tol=1e-5, maxiter=200)
                if e < best_energy:
                    best_energy = e
                    best_positions = relaxed.copy()
            except Exception:
                continue
    
    # Phase 5: Final high-precision polish
    if remaining() > 5:
        try:
            relaxed_final, e_final = _relax(best_positions, tol=1e-8, maxiter=500)
            if e_final < best_energy:
                best_energy = e_final
                best_positions = relaxed_final
        except Exception:
            pass
    
    # Ensure minimum distance constraint
    min_d = _min_distance(best_positions)
    if min_d < 0.5:
        scale = 0.5 / min_d
        best_positions = best_positions * scale
    
    return best_positions
