"""Improved packing of 50 equal circles in a unit square via repulsive forces and local search."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def _radius_score(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^2."""
    # Distance to boundaries
    dist_bound = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    
    # Pairwise distances
    if len(centres) < 2:
        min_pair = np.inf
    else:
        min_pair = pdist(centres, metric='euclidean').min()
    
    r = min(min_pair / 2.0, dist_bound.min())
    return r

def _repulsive_force(centres, k=1.0):
    """Calculate the repulsive force on each centre."""
    n = len(centres)
    forces = np.zeros_like(centres)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = centres[i] - centres[j]
            dist_sq = np.dot(diff, diff)
            # Avoid division by zero
            if dist_sq < 1e-12:
                dist_sq = 1e-12
            dist = np.sqrt(dist_sq)
            # Force magnitude k / dist^2, direction diff / dist
            # F = k * diff / dist^3
            force = k * diff / (dist_sq * dist)
            forces[i] += force
            
    # Boundary repulsion
    # Push away from walls if close
    # We can treat walls as infinite masses or just use a potential.
    # Simpler: if a point is close to a wall, push it inward.
    # Let's add a force based on distance to the nearest wall.
    # dist to left: x + 0.5
    # dist to right: 0.5 - x
    # dist to bottom: y + 0.5
    # dist to top: 0.5 - y
    
    # A smooth potential for walls: V = 1/d^2
    # Force = -grad V = 2/d^3 * n_hat
    
    # Left wall (x = -0.5)
    d_left = centres[:, 0] + 0.5
    d_left = np.maximum(d_left, 1e-6)
    f_left = k * (1.0 / (d_left**2)) # Magnitude
    forces[:, 0] += f_left # Push right (+x)
    
    # Right wall (x = 0.5)
    d_right = 0.5 - centres[:, 0]
    d_right = np.maximum(d_right, 1e-6)
    f_right = k * (1.0 / (d_right**2))
    forces[:, 0] -= f_right # Push left (-x)
    
    # Bottom wall (y = -0.5)
    d_bottom = centres[:, 1] + 0.5
    d_bottom = np.maximum(d_bottom, 1e-6)
    f_bottom = k * (1.0 / (d_bottom**2))
    forces[:, 1] += f_bottom # Push up (+y)
    
    # Top wall (y = 0.5)
    d_top = 0.5 - centres[:, 1]
    d_top = np.maximum(d_top, 1e-6)
    f_top = k * (1.0 / (d_top**2))
    forces[:, 1] -= f_top # Push down (-y)
    
    return forces

def _local_hill_climb(centres, max_iter=1000, step_size=0.01):
    """Perform local hill climbing to refine the packing."""
    current_score = _radius_score(centres)
    
    for _ in range(max_iter):
        improved = False
        # Try moving each point slightly
        for i in range(N):
            original = centres[i].copy()
            # Try 4 directions + original
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_centres = centres.copy()
                    new_centres[i] = original + np.array([dx, dy])
                    # Clamp
                    new_centres[i] = np.clip(new_centres[i], -0.5, 0.5)
                    new_score = _radius_score(new_centres)
                    if new_score > current_score:
                        centres = new_centres
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            # Try smaller step size if stuck? Or break.
            step_size *= 0.5
            if step_size < 1e-5:
                break
    return centres, current_score

def _optimize_repulsive(centres, iterations=500, k=0.01, alpha_start=0.05, alpha_end=0.001):
    """Optimize using repulsive forces."""
    centres = centres.copy()
    alpha = alpha_start
    
    for i in range(iterations):
        # Linear cooling of alpha
        progress = i / iterations
        alpha = alpha_start * (1 - progress) + alpha_end * progress
        
        forces = _repulsive_force(centres, k=k)
        centres += alpha * forces
        centres = np.clip(centres, -0.5, 0.5)
        
    # Refine with local search
    centres, score = _local_hill_climb(centres, max_iter=200, step_size=0.005)
    return centres, score

def construct():
    """Construct 50 circle centres in [-0.5, 0.5]^2 maximizing the common radius."""
    best_centres = None
    best_score = -1.0
    
    # Multi-start
    for seed in range(20):
        np.random.seed(seed)
        # Random initialization
        centres = np.random.uniform(-0.5, 0.5, size=(N, 2))
        
        # Optimization
        opt_centres, opt_score = _optimize_repulsive(centres, iterations=300, k=0.05, alpha_start=0.02, alpha_end=0.0005)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centres = opt_centres
    
    return best_centres
# EVOLVE-BLOCK-END
