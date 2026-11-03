"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centres):
    """Compute the largest common radius for the given centres."""
    # Check validity
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(np.abs(centres) > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    # Distance to boundary
    min_boundary = np.min(0.5 - np.abs(centres))
    
    r = min(min_dist / 2.0, min_boundary)
    return r

def _initial_grid():
    """Create an initial grid of N points in the cube."""
    m = int(np.ceil(N ** (1.0/3.0)))
    # Use a slightly perturbed grid to break symmetry
    np.random.seed(42)
    ticks = (np.arange(m) + 0.5) / m - 0.5 + np.random.uniform(-0.01, 0.01, m) * 0.1
    ticks = np.clip(ticks, -0.49, 0.49)
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _repulsive_forces(centres, softness=0.01):
    """Compute smooth repulsive forces using a soft-min surrogate."""
    forces = np.zeros_like(centres)
    n = len(centres)
    
    for i in range(n):
        for j in range(i+1, n):
            diff = centres[i] - centres[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < 1e-10:
                # Add small random perturbation
                dist = 1e-10
                diff = np.random.uniform(-1e-5, 1e-5, 3)
            
            # Soft repulsion: force ~ 1/dist^2 but smoothed
            # Use a smooth approximation to avoid singularity
            force_mag = 1.0 / (dist**2 + softness**2)
            forces[i] += force_mag * diff / dist
            forces[j] -= force_mag * diff / dist
    
    # Boundary forces: push inward if close to boundary
    for i in range(n):
        for k in range(3):
            if centres[i, k] > 0:
                dist_to_wall = 0.5 - centres[i, k]
                if dist_to_wall < 0.1:
                    forces[i, k] -= 1.0 / (dist_to_wall**2 + softness**2)
            else:
                dist_to_wall = 0.5 + centres[i, k]
                if dist_to_wall < 0.1:
                    forces[i, k] += 1.0 / (dist_to_wall**2 + softness**2)
    
    return forces

def _optimize(centres, max_iter=5000, lr_init=0.001, lr_decay=0.9995, deadline=None):
    """Optimize using gradient descent with repulsive forces and annealing."""
    best_centres = centres.copy()
    best_score = _score(centres)
    lr = lr_init
    damping = 0.5
    
    for it in range(max_iter):
        # Check deadline
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        forces = _repulsive_forces(centres)
        
        # Update positions
        centres_new = centres + lr * forces
        centres_new = centres_new * damping + centres * (1 - damping)
        
        # Clip to cube
        centres_new = np.clip(centres_new, -0.499, 0.499)
        
        # Check score
        new_score = _score(centres_new)
        if new_score > best_score:
            best_score = new_score
            best_centres = centres_new.copy()
            centres = centres_new
        else:
            # If score decreased, try smaller step
            lr *= 0.9
        
        lr *= lr_decay
        damping = min(0.9, damping + 0.0001)
    
    return best_centres, best_score

def _polish(centres, deadline=None):
    """High-precision polish pass using local optimization."""
    best_centres = centres.copy()
    best_score = _score(centres)
    lr = 1e-5
    
    for it in range(1000):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        forces = _repulsive_forces(centres, softness=0.001)
        centres_new = centres + lr * forces
        centres_new = np.clip(centres_new, -0.4999, 0.4999)
        
        new_score = _score(centres_new)
        if new_score > best_score:
            best_score = new_score
            best_centres = centres_new.copy()
            centres = centres_new
            lr *= 1.01
        else:
            lr *= 0.95
        
        if lr < 1e-10:
            break
    
    return best_centres, best_score

def construct():
    """Construct the centres of 50 equal spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with grid
    centres = _initial_grid()
    best_centres = centres.copy()
    best_score = _score(centres)
    
    # Main optimization
    best_centres, best_score = _optimize(centres, max_iter=3000, deadline=deadline)
    
    # Polish
    best_centres, best_score = _polish(best_centres, deadline=deadline)
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres
# EVOLVE-BLOCK-END
