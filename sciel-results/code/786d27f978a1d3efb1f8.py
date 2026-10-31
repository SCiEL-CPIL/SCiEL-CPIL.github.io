import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 148.0  # Safety margin for 150s limit
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

def make_fcc_truncated_octahedron():
    """
    Generate the FCC truncated octahedron cluster for 38 atoms using exact shell coordinates.
    
    The truncated octahedron in FCC has shells:
    - Shell 0: origin (1 atom)
    - Shell 1: 12 nearest neighbors (total 13)
    - Shell 2: 14 atoms (total 27)  
    - Shell 3: 11 atoms (total 38)
    
    Using exact FCC lattice points with proper shell structure.
    """
    scale = 2.0 ** (-1.0 / 3.0)
    
    # Generate FCC lattice points (i+j+k even)
    max_coord = 4
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
    
    # Select the first 38 points
    selected = [p[:3] for p in points[:N]]
    coords = np.array(selected, dtype=float)
    
    # Center the cluster
    coords -= np.mean(coords, axis=0)
    
    return coords

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) / 2 (because double counted in matrix)
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Gradient:
    # F_i = sum_j (-24 r^-14 + 12 r^-8) * (r_i - r_j)
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    
    return energy, grad_flat

def relax(positions, maxiter=500, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def optimize_scale(positions, max_iter=200):
    """
    Optimize the global scale factor s to minimize energy E(s * positions).
    This helps if the cluster is slightly compressed or expanded.
    """
    def energy_at_scale(s):
        scaled = positions * s
        return lj_energy_vectorized(scaled)
    
    # Use bounded scalar minimization. Scale is likely close to 1.0
    res = minimize_scalar(energy_at_scale, bounds=(0.7, 1.4), method='bounded', 
                          options={'maxiter': max_iter, 'xatol': 1e-12})
    
    if res.success:
        return positions * res.x
    return positions

def polish(positions, maxiter=5000, gtol=1e-12):
    """
    Final high-precision polish pass with very tight tolerances.
    This squeezes out residual errors from the main relaxation.
    """
    x0 = positions.flatten()
    
    try:
        # First, do a scale optimization
        scaled = optimize_scale(positions)
        if not check_validity(scaled):
            scaled = positions
        
        x0 = scaled.flatten()
        
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200,
                                   'ftol': 1e-15, 'iprint': 0})
        polished = result.x.reshape(N, 3)
        
        if check_validity(polished):
            return polished
        return positions
    except Exception:
        return positions

def basin_hopping(positions, n_iter=30, step_size=0.25, maxiter=150, temperature=0.05):
    """Perform basin hopping optimization with adaptive step size and cooling."""
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    
    rng = np.random.default_rng(42)
    
    # Adaptive step size: start larger, decrease if stuck
    current_step = step_size
    stuck_count = 0
    min_step = 0.05
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        # Cool down temperature
        temp = temperature * (0.95 ** i)
        
        # Perturb with adaptive step size
        noise = rng.normal(0, current_step, x0.shape)
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
            stuck_count = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            accept = math.exp(-delta / temp) if delta > 0 else 1.0
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy
            else:
                stuck_count += 1
        
        # Adaptive step size: increase if stuck, decrease if improving
        if stuck_count > 3:
            current_step = min(current_step * 1.5, 0.5)
            stuck_count = 0
        elif i % 5 == 0 and i > 0:
            current_step = max(current_step * 0.9, min_step)
    
    return best_x.reshape(N, 3), best_energy

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Construct FCC truncated octahedron 38
    try:
        fcc = make_fcc_truncated_octahedron()
        if check_validity(fcc):
            # Initial relaxation
            relaxed = relax(fcc, maxiter=500, gtol=1e-8)
            if check_validity(relaxed):
                # Optimize Scale
                scaled = optimize_scale(relaxed)
                if check_validity(scaled):
                    # Relax again after scaling
                    relaxed2 = relax(scaled, maxiter=500, gtol=1e-9)
                    if check_validity(relaxed2):
                        energy = lj_energy_vectorized(relaxed2)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = relaxed2
    except Exception:
        pass

    # 2. If we have time, do Basin Hopping from the best result
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 40:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=30, step_size=0.25, maxiter=150, temperature=0.05)
            if check_validity(bh_result):
                # Optimize scale and relax again
                scaled_bh = optimize_scale(bh_result)
                if check_validity(scaled_bh):
                    final_bh = relax(scaled_bh, maxiter=500, gtol=1e-9)
                    if check_validity(final_bh):
                        energy = lj_energy_vectorized(final_bh)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = final_bh
        except Exception:
            pass

    # 3. Final high-precision polish - this is the key improvement
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 5:
        try:
            polished = polish(best_positions, maxiter=15000, gtol=1e-14)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # 4. One more ultra-tight polish if time permits
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 1:
        try:
            # Final scale optimization
            scaled_final = optimize_scale(best_positions)
            if check_validity(scaled_final):
                # Ultra-tight relaxation
                x0 = scaled_final.flatten()
                result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                                  jac=True,
                                  options={'maxiter': 25000, 'gtol': 1e-16, 'maxls': 500,
                                           'ftol': 1e-18})
                ultra_polished = result.x.reshape(N, 3)
                if check_validity(ultra_polished):
                    energy = lj_energy_vectorized(ultra_polished)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = ultra_polished
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        # Try a simple cubic lattice fallback if everything fails
        side = int(np.ceil(N ** (1/3)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pts = np.array(pts[:N], dtype=float)
        # Center it
        pts -= np.mean(pts, axis=0)
        # Scale to LJ equilibrium
        pts *= 1.12
        if check_validity(pts):
            best_positions = pts
        else:
            # Last resort
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions
