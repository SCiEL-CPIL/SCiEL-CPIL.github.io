import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 140.0  # Safety margin before 150s kill

def lj_energy_vectorized(positions):
    """Compute LJ energy efficiently."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1e10)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated octahedron points.
    This is the known global minimum topology.
    """
    points = []
    # FCC lattice points are where i+j+k is even in a cubic grid, scaled by 1/sqrt(2)
    max_coord = 5
    scale = 1.0 / math.sqrt(2.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i * i + j * j + k * k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    
    coords = np.array(selected, dtype=float)
    coords -= np.mean(coords, axis=0)
    return coords

def local_minimize(positions, maxiter=500, gtol=1e-8):
    """
    Perform a local L-BFGS minimization.
    """
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        return lj_energy_vectorized(pos)
    
    try:
        result = minimize(
            energy_func,
            positions.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-12}
        )
        if result.success or result.fun < lj_energy_vectorized(positions):
            min_pos = result.x.reshape(N, 3)
            min_energy = result.fun
            if check_validity(min_pos):
                return min_pos, min_energy
    except Exception:
        pass
    
    # Fallback
    return positions, lj_energy_vectorized(positions)

def construct():
    start_time = time.time()
    
    # Start with the known global minimum topology
    current_pos = make_fcc_truncated_octahedron()
    if len(current_pos) != N:
        # Fallback to random if generation fails
        rng = np.random.default_rng(0)
        current_pos = rng.uniform(-2, 2, size=(N, 3))
        current_pos -= np.mean(current_pos, axis=0)
    
    # Initial relaxation
    current_pos, current_energy = local_minimize(current_pos, maxiter=200, gtol=1e-6)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Simulated Annealing / Basin Hopping Loop
    # We use a cooling schedule for the perturbation size
    T_start = 2.0
    T_end = 0.1
    num_steps = 500  # Maximum number of steps
    
    rng = np.random.default_rng(42)
    
    for step in range(num_steps):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Calculate current temperature
        progress = step / num_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Generate perturbation
        # Scale perturbation by temperature
        perturbation = rng.normal(0, T, size=(N, 3))
        
        # Apply perturbation
        new_pos = current_pos + perturbation
        
        # Relax the perturbed structure
        relaxed_pos, relaxed_energy = local_minimize(new_pos, maxiter=100, gtol=1e-5)
        
        # Acceptance criterion
        if relaxed_energy < current_energy:
            # Always accept lower energy
            current_pos = relaxed_pos
            current_energy = relaxed_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        else:
            # Metropolis criterion
            delta_e = relaxed_energy - current_energy
            if delta_e < 0:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta_e / T)
                
            if accept:
                current_pos = relaxed_pos
                current_energy = relaxed_energy
    
    # Final Polish on the best candidate found
    if time.time() - start_time < DEADLINE - 2:
        polished_pos, polished_energy = local_minimize(best_pos, maxiter=1000, gtol=1e-10)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished_pos
    
    # Final Validity Check
    if not check_validity(best_pos):
        # Emergency fallback
        best_pos = make_fcc_truncated_octahedron()
        if not check_validity(best_pos):
            best_pos = np.zeros((N, 3))
            for i in range(N):
                best_pos[i] = [i*0.6, 0, 0]
                
    return best_pos

# EVOLVE-BLOCK-END
