"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct the 38-atom FCC truncated octahedron (global minimum of LJ38).
    Uses exact integer lattice indices for the 38 closest FCC points.
    """
    
    # Scale factor: a/2 where a = 2^(2/3)
    # Nearest neighbor distance r_nn = a/sqrt(2) = 2^(1/6)
    # s = a/2 = 2^(2/3)/2 = 2^(-1/3)
    s = 2.0 ** (-1.0 / 3.0)
    
    # Generate the 38 closest FCC lattice points
    # FCC points: (i, j, k) with i+j+k even.
    # We need 1 (0,0,0), 12 (1,1,0), 6 (2,0,0), 19 (2,1,1).
    
    points_int = []
    
    # Layer 0: 1 point
    points_int.append((0, 0, 0))
    
    # Layer 1: 12 points (perms of 1,1,0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i*i + j*j + k*k == 2 and (i+j+k) % 2 == 0:
                    points_int.append((i, j, k))
    
    # Layer 2: 6 points (perms of 2,0,0)
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                if i*i + j*j + k*k == 4 and (i+j+k) % 2 == 0:
                    points_int.append((i, j, k))
    
    # Layer 3: 24 points (perms of 2,1,1). We need 19.
    # Generate all 24 and sort deterministically (e.g., lexicographically) to pick 19.
    layer3 = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                if i*i + j*j + k*k == 6 and (i+j+k) % 2 == 0:
                    layer3.append((i, j, k))
    
    # Sort to ensure deterministic selection
    layer3.sort()
    points_int.extend(layer3[:19])
    
    # Convert to float coordinates
    points = np.array(points_int, dtype=float) * s
    
    # Center the cluster at the origin
    center = points.mean(axis=0)
    points = points - center
    
    # Define LJ energy and gradient
    def lj_energy_and_grad(coords):
        n = len(coords)
        energy = 0.0
        grad = np.zeros_like(coords)
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = coords[i] - coords[j]
                r2 = np.dot(r_vec, r_vec)
                if r2 < 1e-12:
                    continue
                r6 = r2 ** 3
                r12 = r6 ** 2
                # Gradient coefficient: 4 * (12/r^14 - 6/r^8)
                # r^14 = (r^2)^7, r^8 = (r^2)^4
                r14 = r2 ** 7
                r8 = r2 ** 4
                coeff = 4.0 * (12.0 / r14 - 6.0 / r8)
                grad[i] += coeff * r_vec
                grad[j] -= coeff * r_vec
                energy += 4.0 * (1.0 / r12 - 1.0 / r6)
        return energy, grad
    
    # Tight relaxation to remove any FP drift and ensure local minimum
    result = minimize(
        lambda x: lj_energy_and_grad(x.reshape(-1, 3))[0],
        points.flatten(),
        jac=lambda x: lj_energy_and_grad(x.reshape(-1, 3))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-14}
    )
    
    final_positions = result.x.reshape(-1, 3)
    
    return final_positions
# EVOLVE-BLOCK-END
