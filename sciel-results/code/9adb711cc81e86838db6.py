"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds


def construct():
    # 3D Lennard-Jones-style repulsion minimization with wall constraints
    # to maximize the minimum inter-particle distance.
    
    np.random.seed(42)
    
    # Initial random positions within the cube [-0.5, 0.5]^3
    # Start with a small margin to avoid initial overlaps
    margin = 0.02
    positions = np.random.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))
    
    # We want to maximize the minimum distance, so we minimize the negative
    # of the minimum distance. But that's non-smooth. Instead, we use a
    # smooth proxy: minimize the sum of inverse powers of distances (like
    # Lennard-Jones repulsion) which pushes all pairs apart.
    
    # Use a potential that penalizes close pairs strongly:
    # E = sum_{i<j} 1/d_{ij}^alpha for some alpha, plus wall penalties.
    # Minimizing this pushes particles apart and toward the center.
    
    alpha = 12  # LJ-like exponent for strong repulsion at short range
    
    def energy(pos):
        pos = pos.reshape(N, 3)
        
        # Compute all pairwise distances
        # Use broadcasting: diff[i,j,:] = pos[i] - pos[j]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Upper triangle only
        iu = np.triu_indices(N, k=1)
        d_sq = dist_sq[iu]
        
        # Repulsive energy: sum of 1/d^alpha
        # Use d^alpha = (d_sq)^(alpha/2)
        repulsion = np.sum(d_sq ** (-alpha / 2.0))
        
        # Wall penalty: for each coordinate, penalize if |c_ik| > 0.5
        # Soft penalty: sum over all coords of max(0, |c_ik| - 0.5 + eps)^2
        # Actually, we want to keep them inside, so penalize approaching walls.
        # Use a smooth barrier: for each coord, if abs(c) > 0.45, add penalty
        # Better: use a smooth function that goes to infinity at the wall.
        
        # Let's use: for each coordinate value v, penalty = max(0, v - 0.5)^2 + max(0, -0.5 - v)^2
        # But this is zero inside, which is fine. We want to encourage staying away from walls.
        # Actually, for maximizing radius, we want to push toward center, not just stay inside.
        # The repulsion term already does that. The wall constraint is just a bound.
        
        # For the wall penalty, use a soft constraint:
        # penalty_wall = sum over all coords of (max(0, |v| - 0.5 + 0.01))^2
        # This adds a small buffer zone.
        
        abs_pos = np.abs(pos)
        # Penalty for being too close to wall (within 0.01 of boundary)
        wall_excess = np.maximum(abs_pos - (0.5 - 0.01), 0.0)
        wall_penalty = np.sum(wall_excess ** 2)
        
        return repulsion + 100.0 * wall_penalty
    
    def gradient(pos):
        pos = pos.reshape(N, 3)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Gradient of sum_{i<j} 1/d^alpha with respect to pos
        # d/dx_i (1/|x_i - x_j|^alpha) = -alpha * (x_i - x_j) / |x_i - x_j|^(alpha+2)
        # For the sum over j != i:
        # grad_i = sum_{j != i} -alpha * (pos_i - pos_j) / |pos_i - pos_j|^(alpha+2)
        
        # Compute for all pairs
        # dist = sqrt(dist_sq)
        dist = np.sqrt(dist_sq)
        dist_cubed = dist ** (alpha + 2)  # |x_i - x_j|^(alpha+2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist_cubed, 1.0)
        
        # coeff[i,j] = -alpha / dist_cubed[i,j] for i != j, 0 for i == j
        coeff = -alpha / dist_cubed
        np.fill_diagonal(coeff, 0.0)
        
        # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
        # = sum_j coeff[i,j] * diff[i,j,:]
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Wall gradient
        abs_pos = np.abs(pos)
        wall_excess = np.maximum(abs_pos - (0.5 - 0.01), 0.0)
        # d/dv max(0, |v| - 0.49)^2 = 2 * max(0, |v| - 0.49) * sign(v) if |v| > 0.49 else 0
        wall_grad = np.zeros_like(pos)
        mask = abs_pos > (0.5 - 0.01)
        wall_grad[mask] = 2.0 * wall_excess[mask] * np.sign(pos[mask])
        
        grad += 100.0 * wall_grad
        
        return grad
    
    # Use L-BFGS-B with bounds to strictly keep positions in the cube
    bounds = Bounds(np.full(N * 3, -0.5), np.full(N * 3, 0.5))
    
    # Run multiple restarts and keep the best
    best_result = None
    best_min_dist = 0.0
    
    for restart in range(5):
        np.random.seed(42 + restart)
        margin = 0.02
        positions = np.random.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))
        
        result = minimize(
            energy,
            positions.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        pos = result.x.reshape(N, 3)
        
        # Compute the actual minimum distance for this configuration
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        iu = np.triu_indices(N, k=1)
        min_dist_sq = dist_sq[iu].min()
        min_dist = np.sqrt(min_dist_sq)
        
        # Also check wall clearance
        wall_clearance = 0.5 - np.max(np.abs(pos))
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_result = pos.copy()
    
    # Final polish: run one more optimization from the best result
    if best_result is not None:
        result = minimize(
            energy,
            best_result.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
        )
        best_result = result.x.reshape(N, 3)
    
    # Ensure all positions are strictly inside the cube
    best_result = np.clip(best_result, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_result
# EVOLVE-BLOCK-END
