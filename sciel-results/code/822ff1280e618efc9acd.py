"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _compute_r(centres):
    """Compute the maximum valid radius for given centres in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centres)):
        return 0.0
    if centres.shape != (N, 3):
        return 0.0
    # Wall constraint: distance from each centre to the nearest wall
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        return min_wall
    diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _force_refine(centres, max_iters=500, dt_init=0.01):
    """Force-directed repulsion with wall constraints."""
    centres = centres.copy().astype(np.float64)
    best_centres = centres.copy()
    best_r = _compute_r(best_centres)
    
    for it in range(max_iters):
        # Compute forces
        tree = cKDTree(centres)
        # Find all pairs within some cutoff
        pairs = tree.query_pairs(0.5, output_type='ndarray')
        
        forces = np.zeros_like(centres)
        
        if len(pairs) > 0:
            for i, j in pairs:
                diff = centres[i] - centres[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 1e-12:
                    # Random nudge for coincident points
                    diff = np.random.randn(3) * 1e-6
                    dist = np.sqrt(np.sum(diff**2))
                # Repulsive force: strong when close
                force_mag = 1.0 / (dist**2 + 1e-8)
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
        
        # Wall forces: push away from walls
        for axis in range(3):
            # Positive wall (x > 0.5)
            over = centres[:, axis] - 0.5
            mask_pos = over > -0.05
            if np.any(mask_pos):
                forces[mask_pos, axis] -= 50.0 * np.clip(over[mask_pos], 0, None) + 1.0 * np.clip(0.05 + over[mask_pos], 0, None)
            # Negative wall (x < -0.5)
            under = centres[:, axis] + 0.5
            mask_neg = under < 0.05
            if np.any(mask_neg):
                forces[mask_neg, axis] += 50.0 * np.clip(-under[mask_neg], 0, None) + 1.0 * np.clip(0.05 - under[mask_neg], 0, None)
        
        # Update positions with adaptive step size
        dt = dt_init * (0.995 ** it)
        centres += dt * forces
        
        # Clamp to valid region (with small margin to allow force to work)
        centres = np.clip(centres, -0.499, 0.499)
        
        # Evaluate
        r = _compute_r(centres)
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
    
    return best_centres, best_r


def _local_optimize(centres, max_iters=200):
    """Use scipy minimize to maximize the minimum radius."""
    def neg_radius(x):
        c = x.reshape(N, 3)
        r = _compute_r(c)
        return -r
    
    x0 = centres.flatten()
    result = minimize(neg_radius, x0, method='Nelder-Mead',
                      options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    
    c_opt = result.x.reshape(N, 3)
    r_opt = -result.fun
    return c_opt, r_opt


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Start with a simple grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_r = _compute_r(best_centres)
    
    # Multiple restarts with different initializations and refinement strategies
    np.random.seed(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Create perturbed initialization
        if restart == 0:
            init = centres.copy()
        elif restart == 1:
            # Slightly perturbed grid
            init = centres.copy() + np.random.randn(N, 3) * 0.02
            init = np.clip(init, -0.49, 0.49)
        elif restart == 2:
            # Random initialization
            init = np.random.uniform(-0.45, 0.45, (N, 3))
        elif restart == 3:
            # FCC-like structure
            a = 1.0 / (N ** (1/3.0))
            fcc_points = []
            for i in range(int(np.ceil(N**0.5))):
                for j in range(int(np.ceil(N**0.5))):
                    for k in range(int(np.ceil(N**0.5))):
                        x = (i + 0.5) * a - 0.5
                        y = (j + 0.5) * a - 0.5
                        z = (k + 0.5) * a - 0.5
                        fcc_points.append([x, y, z])
                        if len(fcc_points) >= N:
                            break
                    if len(fcc_points) >= N:
                        break
                if len(fcc_points) >= N:
                    break
            init = np.array(fcc_points[:N])
            # Add small perturbation
            init = init + np.random.randn(N, 3) * 0.01
            init = np.clip(init, -0.49, 0.49)
        else:
            # More aggressive perturbation
            init = best_centres.copy() + np.random.randn(N, 3) * 0.05
            init = np.clip(init, -0.49, 0.49)
        
        # Force-directed refinement
        c_refined, r_refined = _force_refine(init, max_iters=300)
        if r_refined > best_r:
            best_r = r_refined
            best_centres = c_refined.copy()
        
        if time.time() > deadline - 10.0:
            break
        
        # Local optimization with scipy
        c_opt, r_opt = _local_optimize(c_refined, max_iters=100)
        if r_opt > best_r:
            best_r = r_opt
            best_centres = c_opt.copy()
    
    # Final validation
    final_r = _compute_r(best_centres)
    if final_r > best_r:
        best_r = final_r
    
    return best_centres

# EVOLVE-BLOCK-END
