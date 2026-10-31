import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 135.0
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_and_grad(positions):
    """Compute LJ energy and gradient analytically (vectorized, optimized)."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    # Gradient: dE/dx_i = sum_{j != i} 24 * (2*inv_r12 - inv_r6) * (x_i - x_j)
    # Note: The factor 2 in energy sum (2.0 * sum) accounts for double counting in the full matrix sum.
    # Let's verify the gradient factor.
    # E = sum_{i<j} 4(r^-12 - r^-6)
    # dE/dx_i = sum_{j != i} 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j) / r * (1/r) ? No.
    # d/dr (r^-12) = -12 r^-13. d/dr (r^-6) = -6 r^-7.
    # dE_pair/dx_i = 4 * (-12 r^-13 + 6 r^-7) * (x_i - x_j) / r
    # = 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j)
    # = 24 * (2 r^-14 - r^-8) * (x_i - x_j)  ... wait.
    # 4 * (-12 r^-14 + 6 r^-8) = -48 r^-14 + 24 r^-8 = 24 * (2 r^-8 - 2 r^-14) ? No.
    # 24 * (2 r^-14 - r^-8) = 48 r^-14 - 24 r^-8.
    # My derivative: -48 r^-14 + 24 r^-8.
    # So it is 24 * (r^-8 - 2 r^-14).
    # Let's check the code in the current program:
    # coeff = 24.0 * inv_r2 * (inv_r6 - 2.0 * inv_r12)
    # inv_r2 * inv_r6 = r^-2 * r^-6 = r^-8.
    # inv_r2 * inv_r12 = r^-2 * r^-12 = r^-14.
    # So coeff = 24 * (r^-8 - 2 r^-14).
    # This matches my derivation.
    coeff = 24.0 * inv_r2 * (inv_r6 - 2.0 * inv_r12)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return energy, grad

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation (optimized)."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart (vectorized)."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist

def make_fcc_truncated_octahedron(n_atoms=38):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    """
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 8
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[3])
    
    # Select the first n_atoms points
    selected = [p[:3] for p in points[:n_atoms]]
    return np.array(selected, dtype=float)

def get_fcc_shell_structure(n_atoms=38):
    """
    Return base FCC positions and shell assignments for LJ38 truncated octahedron.
    """
    base_coords = make_fcc_truncated_octahedron(n_atoms)
    radii = np.linalg.norm(base_coords, axis=1)
    
    # Cluster radii into shells based on distinct distances
    # Using a tolerance to group close radii
    unique_r = []
    shell_ids = np.zeros(n_atoms, dtype=int)
    
    sorted_indices = np.argsort(radii)
    current_shell = -1
    current_r = -1.0
    
    for idx in sorted_indices:
        r = radii[idx]
        if current_r == -1.0 or abs(r - current_r) > 0.1:
            current_shell += 1
            unique_r.append(r)
            current_r = r
        shell_ids[idx] = current_shell
    
    return base_coords, shell_ids, unique_r

def symmetry_adapted_energy(params, base_coords, shell_ids):
    """
    Compute energy for symmetry-adapted parameters.
    params: [global_scale, shell_breathing_0, shell_breathing_1, ...]
    """
    global_scale = params[0]
    shell_params = params[1:]
    
    positions = np.zeros((N, 3))
    for i in range(N):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    return lj_energy_vectorized(positions)

def symmetry_adapted_energy_and_grad(params, base_coords, shell_ids):
    """
    Compute energy and gradient for symmetry-adapted parameters.
    Uses numerical gradient for robustness and simplicity in low dimensions.
    """
    energy = symmetry_adapted_energy(params, base_coords, shell_ids)
    
    eps = 1e-6
    grad = np.zeros_like(params)
    for k in range(len(params)):
        params_plus = params.copy()
        params_plus[k] += eps
        e_plus = symmetry_adapted_energy(params_plus, base_coords, shell_ids)
        
        params_minus = params.copy()
        params_minus[k] -= eps
        e_minus = symmetry_adapted_energy(params_minus, base_coords, shell_ids)
        
        grad[k] = (e_plus - e_minus) / (2 * eps)
    
    return energy, grad

def optimize_symmetry_adapted(base_coords, shell_ids, maxiter=200):
    """
    Optimize the low-dimensional symmetry-adapted parameterization.
    """
    n_shells = len(np.unique(shell_ids))
    
    # Initial guess: scale = 2^(1/6) (equilibrium distance for LJ pair), all breathing = 0
    initial_scale = 2.0 ** (1.0/6.0)
    x0 = np.array([initial_scale] + [0.0] * n_shells)
    
    try:
        result = minimize(symmetry_adapted_energy_and_grad, x0, args=(base_coords, shell_ids),
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': 1e-8, 'maxls': 50, 'ftol': 1e-12})
        return result.x
    except Exception:
        return x0

def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B with analytical gradient."""
    def energy_and_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_and_grad_func, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=500, gtol=1e-10):
    """High-precision final polish pass with analytical gradient."""
    def energy_and_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_and_grad_func, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-15})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Get base FCC structure and shell info
    base_coords, shell_ids, unique_r = get_fcc_shell_structure(N)
    
    # 2. Symmetry-Adapted Optimization (Low-Dim)
    try:
        if time.time() - START_TIME < DEADLINE - 60:
            optimized_params = optimize_symmetry_adapted(base_coords, shell_ids, maxiter=100)
            
            # Reconstruct positions from optimized parameters
            global_scale = optimized_params[0]
            shell_params = optimized_params[1:]
            
            positions_sym = np.zeros((N, 3))
            for i in range(N):
                s = shell_ids[i]
                if s < len(shell_params):
                    breath = 1.0 + shell_params[s]
                else:
                    breath = 1.0
                positions_sym[i] = base_coords[i] * breath * global_scale
            
            # 3. High-Dim Relaxation (Polish) on the symmetry-adapted result
            relaxed_sym = relax(positions_sym, maxiter=200, gtol=1e-7)
            
            if check_validity(relaxed_sym):
                energy_sym = lj_energy_vectorized(relaxed_sym)
                if energy_sym < best_energy:
                    best_energy = energy_sym
                    best_positions = relaxed_sym
    except Exception:
        pass
    
    # 4. Fallback/Comparison: Standard FCC Relaxation
    try:
        if time.time() - START_TIME < DEADLINE - 30:
            # Use the raw FCC structure as another candidate
            fcc_raw = make_fcc_truncated_octahedron(N)
            # Scale to approximate equilibrium
            fcc_scaled = fcc_raw * (2.0 ** (1.0/6.0))
            
            relaxed_fcc = relax(fcc_scaled, maxiter=200, gtol=1e-7)
            
            if check_validity(relaxed_fcc):
                energy_fcc = lj_energy_vectorized(relaxed_fcc)
                if energy_fcc < best_energy:
                    best_energy = energy_fcc
                    best_positions = relaxed_fcc
    except Exception:
        pass
    
    # 5. Final High-Precision Polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions, maxiter=500, gtol=1e-10)
            if check_validity(polished):
                energy_polished = lj_energy_vectorized(polished)
                if energy_polished < best_energy:
                    best_energy = energy_polished
                    best_positions = polished
        except Exception:
            pass
    
    # Final check and return
    if best_positions is None:
        # Emergency fallback
        try:
            best_positions = make_fcc_truncated_octahedron(N) * (2.0 ** (1.0/6.0))
            if not check_validity(best_positions):
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions
