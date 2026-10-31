import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0
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
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def lj_energy_and_grad(positions):
    """Compute LJ energy and gradient vectorized."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Gradient: dE/dr * dr/dpos
    # dE/dr = 4 * (-12*r^-13 + 6*r^-7) = 24/r * (r^-12 - r^-6) ... let me be careful
    # E = 4 * sum(r^-12 - r^-6)
    # dE/dr_ij = 4 * (-12*r^-13 + 6*r^-7) = -48*r^-13 + 24*r^-7
    # grad_pos_i = -sum_j dE/dr_ij * (r_j - r_i)/r_ij
    # grad_pos_j = sum_i dE/dr_ij * (r_j - r_i)/r_ij
    
    # dE/dr for each pair (i,j) where i<j: -48/r^13 + 24/r^7 = 24/r^7 * (1 - 2/r^6)
    # Let's compute per-pair force magnitude
    # f_ij = -dE/dr_ij = 48*r^-13 - 24*r^-7 = 24/r^7 * (2/r^6 - 1)
    # grad_i += f_ij * (pos_i - pos_j)/r_ij = -f_ij * (pos_j - pos_i)/r_ij
    # grad_j -= f_ij * (pos_i - pos_j)/r_ij = f_ij * (pos_j - pos_i)/r_ij
    
    # Using vectorized approach:
    # For each pair, the contribution to gradient of atom i from atom j is:
    # -4 * (-12 * r^-13 + 6 * r^-7) * (pos_j - pos_i) / r
    # = 4 * (12 * r^-13 - 6 * r^-7) * (pos_j - pos_i) / r
    # = (48 * r^-14 - 24 * r^-8) * (pos_j - pos_i)
    
    # Let's use: grad_i = sum_j (48/r_ij^14 - 24/r_ij^8) * (pos_j - pos_i)
    # where the coefficient is applied to the vector (pos_j - pos_i)
    
    # More precisely:
    # dE/dpos_i = sum_{j!=i} dE/dr_ij * d r_ij/dpos_i
    # r_ij = |pos_j - pos_i|
    # d r_ij/dpos_i = -(pos_j - pos_i)/r_ij
    # dE/dr_ij = 4*(-12*r^-13 + 6*r^-7)
    # So dE/dpos_i = sum_j 4*(-12*r^-13 + 6*r^-7) * (-(pos_j - pos_i)/r_ij)
    # = sum_j (48*r^-13 - 24*r^-7) * (pos_j - pos_i)/r_ij
    # = sum_j (48*r^-14 - 24*r^-8) * (pos_j - pos_i)
    
    coeff = 48.0 * inv_r2**7 - 24.0 * inv_r2**4  # 48/r^14 - 24/r^8
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist

def make_fcc_truncated_octahedron(n_atoms=38):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    This is the known global minimum structure for LJ38.
    """
    # FCC lattice points: (i,j,k) with i+j+k even
    # Nearest neighbor distance in FCC with lattice constant a is a/sqrt(2)
    # For LJ equilibrium, nearest neighbor distance should be 2^(1/6)
    # So a/sqrt(2) = 2^(1/6) => a = sqrt(2) * 2^(1/6)
    
    # Generate FCC points
    max_coord = 8
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k, i*i + j*j + k*k))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[3])
    
    # Select the first n_atoms points
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    # Scale: FCC nearest neighbor is sqrt(2) in integer coords
    # We want nearest neighbor distance = 2^(1/6)
    # So scale factor = 2^(1/6) / sqrt(2) = 2^(1/6 - 1/2) = 2^(-1/3)
    scale = 2.0 ** (-1.0 / 3.0)
    coords *= scale
    
    return coords

def make_fcc_with_noise(n_atoms=38, noise_scale=0.05, seed=42):
    """Generate FCC cluster with small noise to break symmetry."""
    base = make_fcc_truncated_octahedron(n_atoms)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_scale, base.shape)
    return base + noise

def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
    
    def energy_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g.ravel()
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_grad_func, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=500, gtol=1e-10):
    """High-precision final polish pass."""
    def energy_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g.ravel()
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_grad_func, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def basin_hopping(positions, n_iter=10, step_size=0.5, maxiter=100, temperature=0.05):
    """Perform basin hopping optimization."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
    
    def energy_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g.ravel()
    
    x0 = positions.flatten()
    current_energy = energy_func(x0)
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
            result = minimize(energy_grad_func, perturbed, method='L-BFGS-B', jac=True,
                            options={'maxiter': maxiter, 'gtol': 1e-6, 'maxls': 50})
            new_x = result.x
            new_energy = energy_func(new_x)
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
    
    candidates = []
    
    # 1. FCC Truncated Octahedron (primary candidate)
    try:
        fcc = make_fcc_truncated_octahedron(N)
        if check_validity(fcc):
            candidates.append(('fcc', fcc))
    except Exception:
        pass
        
    # 2. FCC with small noise
    for seed in [42, 123, 456]:
        try:
            fcc_noisy = make_fcc_with_noise(N, noise_scale=0.05, seed=seed)
            if check_validity(fcc_noisy):
                candidates.append((f'fcc_noise_{seed}', fcc_noisy))
        except Exception:
            pass
        
    # 3. FCC with larger noise
    for seed in [789, 101]:
        try:
            fcc_noisy = make_fcc_with_noise(N, noise_scale=0.15, seed=seed)
            if check_validity(fcc_noisy):
                candidates.append((f'fcc_noise_large_{seed}', fcc_noisy))
        except Exception:
            pass
    
    # Relax all candidates
    for name, cand in candidates:
        if time.time() - START_TIME > DEADLINE - 20:
            break
            
        try:
            # First do a standard relaxation
            relaxed = relax(cand, maxiter=200, gtol=1e-7)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass
    
    # If we have time, do basin hopping on the best result
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 25:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=20, step_size=0.3, maxiter=80, temperature=0.03)
            if check_validity(bh_result) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_result
        except Exception:
            pass
    
    # Final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions, maxiter=500, gtol=1e-10)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
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
