"""Program for packing 50 equal spheres in a unit cube.
Uses BCC-inspired initialization with repulsive potential energy minimization."""
import numpy as np
from scipy.optimize import minimize

N = 50


def _repulsive_potential(flat_x):
    """Compute a repulsive potential energy that we want to MINIMIZE.
    
    We use an inverse-power repulsion between all pairs and walls.
    Minimizing this maximizes the minimum distance.
    """
    x = flat_x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Pair repulsion: sum of 1/d^4 (stronger repulsion when close, smoother gradient)
    pair_energy = np.sum(1.0 / (dist_sq ** 2))
    
    # Wall distances: distance from each center to the nearest wall
    # Cube is [-0.5, 0.5]^3
    wall_dists = np.minimum(
        np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        ),
        np.minimum(0.5 - x[:, 2], x[:, 2] + 0.5)
    )
    
    # Wall repulsion: sum of 1/wall_dist^4
    wall_dists_clipped = np.maximum(wall_dists, 1e-8)
    wall_energy = np.sum(1.0 / (wall_dists_clipped ** 4))
    
    return pair_energy + wall_energy


def _score(x):
    """Compute the actual score: minimum radius."""
    x = x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        ),
        np.minimum(0.5 - x[:, 2], x[:, 2] + 0.5)
    )
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _bcc_initialization(scale=1.0, seed=42):
    """Initialize 50 centers on a distorted BCC superlattice.
    
    BCC lattice: For a cube of side L, BCC has 2 atoms per conventional cell.
    We create a 3x3x3 grid of BCC unit cells (27 cells * 2 atoms = 54),
    then select 50 of them with slight perturbation.
    """
    np.random.seed(seed)
    
    n_cells = 3  # 3x3x3 = 27 BCC cells
    cell_size = 1.0 / n_cells
    
    centers = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                # Base position of this sub-cube
                cx = -0.5 + (i + 0.5) * cell_size
                cy = -0.5 + (j + 0.5) * cell_size
                cz = -0.5 + (k + 0.5) * cell_size
                
                # BCC: two atoms - one at corner offset, one at body center
                # Corner-like atom (offset toward one corner)
                offset = 0.25 * cell_size
                centers.append([cx - offset, cy - offset, cz - offset])
                # Body center atom
                centers.append([cx + offset, cy + offset, cz + offset])
    
    centers = np.array(centers[:N])
    
    # Add small random perturbation to break symmetry
    perturbation = 0.02 * scale
    centers += np.random.randn(N, 3) * perturbation
    
    # Clip to stay within cube with margin
    margin = 0.48
    centers = np.clip(centers, -margin, margin)
    
    return centers


def _local_refinement(x, max_iters=100, step_size=0.01):
    """Local refinement using coordinate descent on the minimum distance.
    
    For each point, try small perturbations to increase the minimum distance.
    """
    x = x.copy()
    best_score = _score(x)
    
    for iteration in range(max_iters):
        improved = False
        for i in range(N):
            for coord in range(3):
                for direction in [-1, 1]:
                    x_trial = x.copy()
                    x_trial[i, coord] += direction * step_size
                    # Keep within bounds
                    x_trial[i, coord] = np.clip(x_trial[i, coord], -0.499, 0.499)
                    
                    s = _score(x_trial)
                    if s > best_score + 1e-10:
                        x = x_trial
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return x


def construct():
    """Construct centers for 50 equal spheres in a unit cube."""
    best_x = None
    best_score = 0.0
    
    # Try multiple BCC initializations with different seeds and scales
    seeds = [42, 123, 456, 789, 1011, 2023]
    scales = [0.5, 1.0, 1.5]
    
    for seed in seeds:
        for scale in scales:
            x0 = _bcc_initialization(scale=scale, seed=seed)
            x0_flat = x0.flatten()
            
            # Define bounds: each coordinate in [-0.5, 0.5]
            bounds = [(-0.499, 0.499)] * (N * 3)
            
            # Optimize using L-BFGS-B
            result = minimize(
                _repulsive_potential,
                x0_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 50000}
            )
            
            x_opt = result.x.reshape(N, 3)
            x_opt = np.clip(x_opt, -0.499, 0.499)
            
            # Local refinement
            x_refined = _local_refinement(x_opt, max_iters=50, step_size=0.005)
            
            s = _score(x_refined)
            if s > best_score:
                best_score = s
                best_x = x_refined
    
    # Final verification: ensure validity
    best_x = np.clip(best_x, -0.499, 0.499)
    
    return best_x
