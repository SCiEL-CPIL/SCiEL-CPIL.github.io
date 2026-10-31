import numpy as np
import time
import sys

N = 38

# EVOLVE-BLOCK-START
def lj_energy_and_grad(pos):
    """
    Compute the Lennard-Jones energy and gradient for a set of positions.
    pos: (N, 3) array of coordinates.
    Returns: energy (float), grad (N, 3) array.
    """
    # Compute pairwise differences
    # pos[:, np.newaxis, :] - pos[np.newaxis, :, :] gives (N, N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # Squared distances
    r2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 ** -3
    r12 = r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) over i < j
    # We compute sum over all i != j and divide by 2
    e_pair = 4.0 * (r12 - r6)
    energy = 0.5 * np.sum(e_pair)
    
    # Gradient
    # d/dr (4(r^-12 - r^-6)) = 4(-12 r^-13 + 6 r^-7)
    # dE/dx_i = sum_{j != i} dE/dx_ij
    # Force on i from j: F_ij = -dE/dx_i
    # Let's compute the force matrix F (N, N, 3) where F[i,j] is force on i due to j?
    # Or simply use the chain rule.
    # E = sum_{i<j} 4(r_ij^-12 - r_ij^-6)
    # dE/dx_i = sum_{j != i} 4(-12 r_ij^-13 + 6 r_ij^-7) * (x_i - x_j) / r_ij
    # Note: (x_i - x_j) / r_ij is the unit vector from j to i.
    # Let's define coeff = 4 * (-12 * r^-13 + 6 * r^-7)
    # This is effectively -24 * r^-13 + 24 * r^-7 = 24 * r^-7 * (1 - 2*r^-6)
    
    # r = sqrt(r2)
    # r^-13 = r2^-6.5
    # r^-7 = r2^-3.5
    
    # Let's compute the factor for the vector (x_i - x_j)
    # factor_ij = 4 * (-12 * r^-13 + 6 * r^-7)
    # We need r.
    r = np.sqrt(r2)
    # Avoid division by zero again, though r2 diag is 1
    r7 = r ** -7
    r13 = r ** -13
    
    # Factor for each pair i,j (i!=j)
    # grad_i = sum_j 4 * (-12 * r^-13 + 6 * r^-7) * (x_i - x_j)
    # Let's compute the scalar part
    scalar = 4.0 * (-12.0 * r13 + 6.0 * r7)
    # scalar is (N, N)
    # We want grad[i] = sum_j scalar[i,j] * diff[i,j]
    # diff[i,j] = pos[i] - pos[j]
    
    # Multiply scalar[i,j] by diff[i,j,k]
    # grad = sum over j of scalar[i,j] * (pos[i] - pos[j])
    # This is equivalent to:
    # grad[i,k] = sum_j scalar[i,j] * diff[i,j,k]
    
    # We can use einsum
    # grad[i,k] = sum_j scalar[i,j] * diff[i,j,k]
    grad = np.einsum('ij,ijk->ik', scalar, diff)
    
    # The diagonal elements of diff are 0, so scalar[i,i] * 0 = 0, which is correct.
    # However, we set r2 diagonal to 1, so r diagonal is 1.
    # scalar[i,i] = 4 * (-12 + 6) = -24.
    # diff[i,i] = 0. So product is 0. Correct.
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate FCC lattice points
    # We need 38 points.
    # FCC points: (i,j,k) with i+j+k even.
    # We will generate a cube of side length L centered at 0.
    # N=38 is small. A 5x5x5 cube contains 125 points.
    # Enough to pick the 38 closest.
    
    side = 5
    coords = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                if (i + j + k) % 2 == 0:
                    coords.append([i, j, k])
    
    coords = np.array(coords, dtype=float)
    
    # Calculate distances from origin
    dists = np.linalg.norm(coords, axis=1)
    
    # Sort by distance
    sorted_indices = np.argsort(dists)
    
    # Take the first N points
    fcc_points = coords[sorted_indices[:N]]
    
    # 2. Scale to optimal LJ bond length
    # Nearest neighbor distance in this integer lattice is sqrt(2).
    # Optimal LJ distance is 2^(1/6).
    # Scale factor s = 2^(1/6) / sqrt(2) = 2^(1/6 - 1/2) = 2^(-1/3)
    scale = 2.0 ** (-1.0/3.0)
    pos = fcc_points * scale
    
    # 3. Local Relaxation
    # Use L-BFGS-B with analytic gradient
    from scipy.optimize import minimize
    
    # Initial guess
    x0 = pos.flatten()
    
    # Define objective for optimizer
    def objective(x):
        p = x.reshape(N, 3)
        e, _ = lj_energy_and_grad(p)
        return e
    
    def gradient(x):
        p = x.reshape(N, 3)
        _, g = lj_energy_and_grad(p)
        return g.flatten()
    
    best_pos = pos
    best_energy = objective(x0)
    
    # Run minimization
    # We can try a few restarts or just one high-precision run.
    # Given the start is very close to the global minimum, one run should suffice.
    # But to be safe against numerical issues, we can do a simple check.
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
        
        if res.success:
            relaxed_pos = res.x.reshape(N, 3)
            relaxed_energy = res.fun
            
            # Verify minimum distance constraint (atoms at least 0.5 apart)
            # In LJ global min, min dist is ~1.12, so this is safe.
            # But let's check just in case of numerical drift.
            diff = relaxed_pos[:, np.newaxis, :] - relaxed_pos[np.newaxis, :, :]
            min_dist = np.min(np.linalg.norm(diff, axis=2)[np.triu_indices(N, k=1)])
            
            if min_dist >= 0.5:
                if relaxed_energy < best_energy:
                    best_pos = relaxed_pos
                    best_energy = relaxed_energy
            else:
                # If constraint violated (unlikely), try a different optimizer or just return initial
                pass
        else:
            # If optimization failed, try a simpler method or just use initial
            pass
            
    except Exception as e:
        # Fallback to initial FCC position if optimization crashes
        pass
    
    # Check time
    if time.time() > deadline:
        return best_pos
    
    # Final polish: run a few more iterations with tighter tolerance if time permits
    if time.time() < deadline - 5.0:
        try:
            x0_polish = best_pos.flatten()
            res2 = minimize(objective, x0_polish, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False})
            if res2.success:
                polished_pos = res2.x.reshape(N, 3)
                diff = polished_pos[:, np.newaxis, :] - polished_pos[np.newaxis, :, :]
                min_dist = np.min(np.linalg.norm(diff, axis=2)[np.triu_indices(N, k=1)])
                if min_dist >= 0.5 and res2.fun < best_energy:
                    best_pos = polished_pos
                    best_energy = res2.fun
        except:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
