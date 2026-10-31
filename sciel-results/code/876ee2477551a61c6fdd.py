import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 147.0
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

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r = np.sqrt(np.dot(rij, rij))
            if r < min_dist:
                return False
    return True

def make_fcc_truncated_octahedron(n_atoms=38):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    This is the known global minimum structure for LJ38.
    
    Build the cluster shell-by-shell from the FCC lattice:
    - Shell 0: origin (1 atom)
    - Shell 1: 12 nearest neighbors (total 13)
    - Shell 2: 24 next-nearest neighbors (total 37)
    - Shell 3: add 1 more from the next shell (total 38)
    
    FCC lattice: points (i, j, k) where i+j+k is even
    Scale by 1/sqrt(2) to get nearest neighbor distance = 1.
    Then scale by 2^(1/6) to reach the LJ equilibrium distance.
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
    coords = np.array(selected, dtype=float)
    
    # Scale to equilibrium distance 2^(1/6)
    coords *= 2.0 ** (1.0 / 6.0)
    
    return coords

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    
    # Avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) / 2 (because double counted in matrix)
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Gradient:
    # F_i = sum_j (-24 r^-14 + 12 r^-8) * (r_i - r_j)
    # Term = (-24 * inv_r12 * inv_r2 + 12 * inv_r6 * inv_r2) * diff
    # inv_r12 * inv_r2 = inv_r14
    # inv_r6 * inv_r2 = inv_r8
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    # coeff shape (N, N), diff shape (N, N, 3)
    # We want sum over j for each i
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    
    return energy, grad_flat

def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=500, gtol=1e-10):
    """High-precision final polish pass with analytic gradient."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def ultra_polish(positions, maxiter=2000, gtol=1e-14, ftol=1e-16):
    """Ultra-high-precision polish pass for final energy minimization."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 200})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def symmetry_refine(positions):
    """
    Refine positions using symmetry-adapted parameterization.
    Treat atoms as belonging to FCC coordination shells and optimize
    radial distances and global distortion parameters.
    """
    # Identify shell assignments based on distance from centroid
    centroid = np.mean(positions, axis=0)
    positions_centered = positions - centroid
    
    # Compute distances from centroid
    dists = np.linalg.norm(positions_centered, axis=1)
    
    # Cluster distances into shells (expect ~4 shells for FCC38)
    sorted_dists = np.sort(dists)
    
    # Simple shell detection: group atoms with similar distances
    shell_threshold = 0.15
    shells = []
    current_shell = [0]
    
    for i in range(1, N):
        if sorted_dists[i] - sorted_dists[i-1] > shell_threshold:
            shells.append(sorted_dists[i-1])
            current_shell = [i]
        else:
            current_shell.append(i)
    shells.append(sorted_dists[-1])
    
    # Assign atoms to shells
    shell_assignments = np.zeros(N, dtype=int)
    for shell_id, shell_dist in enumerate(shells):
        mask = np.abs(dists - shell_dist) < shell_threshold
        shell_assignments[mask] = shell_id
    
    n_shells = len(shells)
    
    # Optimize radial distances for each shell and global scale
    def energy_from_params(params):
        # params: [r1, r2, ..., rn_shells, scale]
        radii = params[:n_shells]
        global_scale = params[n_shells]
        
        new_positions = np.zeros((N, 3))
        for i in range(N):
            shell = shell_assignments[i]
            if shell < n_shells:
                direction = positions_centered[i] / (np.linalg.norm(positions_centered[i]) + 1e-15)
                new_positions[i] = centroid + direction * radii[shell] * global_scale
            else:
                new_positions[i] = positions[i]
        
        return lj_energy_vectorized(new_positions)
    
    # Initial guess: current radii
    initial_radii = np.zeros(n_shells)
    for s in range(n_shells):
        mask = shell_assignments == s
        if np.any(mask):
            initial_radii[s] = np.mean(dists[mask])
    
    initial_params = np.append(initial_radii, 1.0)
    
    try:
        result = minimize(energy_from_params, initial_params, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
        if result.success:
            radii = result.x[:n_shells]
            global_scale = result.x[n_shells]
            
            refined = np.zeros((N, 3))
            for i in range(N):
                shell = shell_assignments[i]
                if shell < n_shells:
                    direction = positions_centered[i] / (np.linalg.norm(positions_centered[i]) + 1e-15)
                    refined[i] = centroid + direction * radii[shell] * global_scale
                else:
                    refined[i] = positions[i]
            
            return refined
    except Exception:
        pass
    
    return positions

def optimize_scale(base_positions):
    """
    Optimize the global scale factor for the given base positions.
    This is a 1D optimization problem: scale all coordinates by s.
    """
    def energy_at_scale(s):
        scaled = base_positions * s
        return lj_energy_vectorized(scaled)
    
    try:
        result = minimize_scalar(energy_at_scale, bounds=(0.9, 1.2), method='bounded',
                                 options={'xatol': 1e-12, 'maxiter': 200})
        if result.success:
            return base_positions * result.x
    except Exception:
        pass
    
    # Fallback: try a few discrete scales
    best_energy = float('inf')
    best_scaled = None
    for s in np.linspace(0.95, 1.15, 21):
        scaled = base_positions * s
        e = lj_energy_vectorized(scaled)
        if e < best_energy:
            best_energy = e
            best_scaled = scaled
    
    return best_scaled if best_scaled is not None else base_positions

def basin_hopping(positions, n_iter=10, step_size=0.5, maxiter=100, temperature=0.05):
    """Perform basin hopping optimization with analytic gradient."""
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    
    rng = np.random.default_rng(123)
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        # Perturb
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        
        # Relax
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B', 
                            jac=True,
                            options={'maxiter': maxiter, 'gtol': 1e-6, 'maxls': 50})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
        except Exception:
            continue
        
        # Metropolis acceptance
        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                accept = math.exp(-delta / temperature)
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy
    
    return best_x.reshape(N, 3), best_energy

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Construct the known FCC truncated octahedron
    try:
        fcc = make_fcc_truncated_octahedron(N)
        if check_validity(fcc):
            # Optimize the global scale
            fcc_scaled = optimize_scale(fcc)
            if check_validity(fcc_scaled):
                energy = lj_energy_vectorized(fcc_scaled)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = fcc_scaled
    except Exception:
        pass
    
    # 2. Relax the scaled FCC structure
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            relaxed = relax(best_positions, maxiter=300, gtol=1e-9)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass
    
    # 3. High-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 15:
        try:
            polished = polish(best_positions, maxiter=1000, gtol=1e-12)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass
    
    # 3b. Symmetry-adapted refinement
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            sym_refined = symmetry_refine(best_positions)
            if check_validity(sym_refined):
                energy = lj_energy_vectorized(sym_refined)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = sym_refined
        except Exception:
            pass
    
    # 4. If we have time, try basin hopping from the best result
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=15, step_size=0.2, maxiter=60, temperature=0.02)
            if check_validity(bh_result) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_result
        except Exception:
            pass
    
    # 5. Final high-precision polish after any basin hopping
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions, maxiter=500, gtol=1e-12)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass
    
    # 6. Ultra-precision final polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 5:
        try:
            ultra_polished = ultra_polish(best_positions, maxiter=2000, gtol=1e-14, ftol=1e-16)
            if check_validity(ultra_polished):
                energy = lj_energy_vectorized(ultra_polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = ultra_polished
        except Exception:
            pass
    
    # Final check and return
    if best_positions is None:
        # Fallback
        try:
            best_positions = make_fcc_truncated_octahedron(N)
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
