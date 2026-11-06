"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds


def construct():
    """
    Lennard-Jones style minimization with wall penalties to maximize minimum
    inter-particle distance and wall clearance for 100 spheres in a unit cube.
    """
    best_score = 0.0
    best_result = None
    
    # LJ parameters
    sigma = 1.0  # Reference distance
    epsilon = 1.0
    # Use a high exponent for the repulsive part to focus on close pairs
    # Standard LJ: (1/d^12 - 1/d^6) but we only want repulsion for packing
    # We'll use a pure repulsive potential: 1/d^12 (or similar)
    # But to handle walls, we add a penalty if any coordinate is too close to boundary
    
    def lj_energy(pos):
        pos = pos.reshape(N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        iu = np.triu_indices(N, k=1)
        d_sq = dist_sq[iu]
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Repulsive potential: sum(1/d^12) = sum(1/d_sq^6)
        # This strongly penalizes close pairs
        pair_energy = np.sum(1.0 / (d_sq ** 6))
        
        # Wall penalty: quadratic penalty if any particle is outside [-0.5, 0.5]
        # We use a smooth penalty that is zero inside and grows quadratically outside
        # Actually, since we use bounds in L-BFGS-B, walls are handled by the optimizer.
        # But to be safe, we add a soft penalty for approaching walls too closely
        # to encourage particles to stay away from walls, which helps maximize
        # the minimum wall clearance.
        
        # Wall clearance penalty: for each coordinate, if |x| > 0.5 - margin,
        # add penalty. But actually, we want to MAXIMIZE wall clearance, which
        # means we want particles to be as far from walls as possible.
        # The score is min(min_pair_dist/2, min_wall_clearance).
        # So we need to balance pair separation and wall clearance.
        
        # A simple approach: add a small attractive force toward the center
        # to keep particles away from walls, but this might hurt pair distances.
        # Better: rely on the pair repulsion to push particles apart, and the
        # bounds to keep them inside. The optimizer will find the balance.
        
        return pair_energy
    
    def lj_gradient(pos):
        pos = pos.reshape(N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        dist_sq_safe = dist_sq.copy()
        np.fill_diagonal(dist_sq_safe, 1.0)
        
        # d/dx_i (1/|x_i - x_j|^12) = -12 * (x_i - x_j) / |x_i - x_j|^14
        # = -12 * diff[i,j] / (dist_sq[i,j]^7)
        coeff = -12.0 / (dist_sq_safe ** 7)
        np.fill_diagonal(coeff, 0.0)
        
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        return grad
    
    bounds = Bounds(np.full(N * 3, -0.5), np.full(N * 3, 0.5))
    
    # Multiple random initializations
    np.random.seed(42)
    
    for trial in range(5):
        if trial == 0:
            # Grid initialization with jitter
            nx, ny, nz = 5, 4, 5
            x = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
            y = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
            z = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
            positions = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
            np.random.seed(123 + trial)
            positions += np.random.uniform(-0.005, 0.005, size=positions.shape)
        else:
            # Random initialization
            np.random.seed(42 + trial * 7)
            positions = np.random.uniform(-0.4, 0.4, size=(N, 3))
        
        positions = np.clip(positions, -0.499, 0.499)
        
        # Run L-BFGS-B optimization
        result = minimize(
            lj_energy,
            positions.ravel(),
            jac=lj_gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
        )
        
        pos = result.x.reshape(N, 3)
        
        # Calculate score
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        iu = np.triu_indices(N, k=1)
        min_dist = np.sqrt(dist_sq[iu].min())
        
        wall_clearance = 0.5 - np.max(np.abs(pos))
        
        current_score = min(min_dist / 2.0, wall_clearance)
        
        if current_score > best_score:
            best_score = current_score
            best_result = pos.copy()
    
    # Final polish with tighter tolerances
    if best_result is not None:
        result = minimize(
            lj_energy,
            best_result.ravel(),
            jac=lj_gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
        )
        best_result = result.x.reshape(N, 3)
    
    # Ensure validity
    best_result = np.clip(best_result, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_result
# EVOLVE-BLOCK-END
