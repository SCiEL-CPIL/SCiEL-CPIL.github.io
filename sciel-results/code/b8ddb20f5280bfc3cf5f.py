"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^3."""
    # Distance to nearest face
    face_dist = 0.5 - np.max(np.abs(centres), axis=1)
    min_face = np.min(face_dist)
    
    # Minimum pairwise distance
    n = len(centres)
    if n < 2:
        return min_face
    
    # Compute all pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_face, min_pair / 2.0)


def _local_optimize(centres, max_iter=5000, tol=1e-10):
    """Local optimization using gradient ascent on the radius objective."""
    centres = centres.copy()
    n = len(centres)
    
    best_radius = _compute_radius(centres)
    best_centres = centres.copy()
    
    # Use Nelder-Mead or L-BFGS-B for local optimization
    # We'll use a simple approach: iteratively move each sphere to maximize
    # the minimum distance to other spheres and walls
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            # Compute forces on sphere i
            force = np.zeros(3)
            
            # Repulsion from other spheres
            for j in range(n):
                if i == j:
                    continue
                d = centres[i] - centres[j]
                dist = np.linalg.norm(d)
                if dist > 1e-12:
                    # Only push if closer than some threshold
                    force += d / dist
            
            # Attraction to center (to keep inside cube)
            force -= centres[i] * 0.1
            
            # Normalize and take small step
            norm = np.linalg.norm(force)
            if norm > 1e-15:
                step = force / norm * 0.001
                new_pos = centres[i] + step
                # Clamp to cube
                new_pos = np.clip(new_pos, -0.5, 0.5)
                
                # Check if this improves the radius
                old_centres = centres.copy()
                centres[i] = new_pos
                new_radius = _compute_radius(centres)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centres = centres.copy()
                    improved = True
                else:
                    centres[i] = old_centres[i]
        
        if not improved:
            break
    
    return best_centres, best_radius


def construct():
    """Construct 50 sphere centres in [-0.5, 0.5]^3 to maximize minimum radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_centres = None
    best_radius = 0.0
    
    # Strategy 1: Start with a good initial configuration
    # Use a face-centered cubic or similar dense packing pattern
    
    # Approach: Use a grid-based initialization with perturbation
    # For 50 spheres, try various lattice structures
    
    # Try body-centered cubic (BCC) arrangement
    # In a unit cube, BCC has 8 corners + 1 center = 9 per sub-cube
    # Divide cube into sub-cubes and place spheres accordingly
    
    # Let's try a more systematic approach:
    # 1. Start with a regular grid
    # 2. Apply local optimization
    # 3. Try multiple random restarts
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    initial += np.random.randn(N, 3) * 0.01
    initial = np.clip(initial, -0.49, 0.49)
    
    # Local optimization
    opt_centres, opt_radius = _local_optimize(initial, max_iter=2000)
    best_centres = opt_centres
    best_radius = opt_radius
    
    # Try multiple random restarts if time permits
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Random initialization
        random_centres = np.random.randn(N, 3) * 0.3
        random_centres = np.clip(random_centres, -0.45, 0.45)
        
        opt_centres, opt_radius = _local_optimize(random_centres, max_iter=1000)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centres = opt_centres
    
    # Final refinement with more iterations if time permits
    if time.time() < deadline - 20:
        refined_centres, refined_radius = _local_optimize(best_centres, max_iter=3000)
        if refined_radius > best_radius:
            best_radius = refined_radius
            best_centres = refined_centres
    
    return best_centres
# EVOLVE-BLOCK-END
