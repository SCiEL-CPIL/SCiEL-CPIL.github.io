"""Improved packing program using simulated annealing / local optimization."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _compute_radius(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^3."""
    if len(centers) < 2:
        return 0.5
    # Minimum distance between any two centers
    from scipy.spatial.distance import pdist
    dists = pdist(centers, metric='euclidean')
    min_dist = np.min(dists)
    # Minimum distance from any center to any face
    # Distance to each face: 0.5 - |coord| for each coordinate
    dists_to_faces = 0.5 - np.abs(centers)
    min_face_dist = np.min(dists_to_faces)
    return min(min_dist / 2.0, min_face_dist)


def _objective(x, centers, weights):
    """Objective to minimize: negative of the radius (so minimizing maximizes radius)."""
    return -_compute_radius(centers)


def construct():
    """
    Place 100 equal spheres in unit cube to maximize common radius.
    Uses a multi-start local optimization approach.
    """
    best_radius = 0.0
    best_centers = None
    
    # Try multiple random initializations and optimize
    np.random.seed(42)
    
    for trial in range(30):
        # Random initialization inside the cube
        centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Run a simple gradient-free optimization using Nelder-Mead on subsets
        # Actually, let's use a different approach: iterative relaxation
        
        # Start with random positions
        pos = centers.copy()
        
        # Iterative improvement: push apart overlapping spheres and pull toward center
        for it in range(2000):
            # Compute pairwise distances
            from scipy.spatial.distance import cdist
            D = cdist(pos, pos)
            np.fill_diagonal(D, np.inf)
            
            # For each sphere, compute the minimum distance to others
            min_dists = np.min(D, axis=1)
            
            # Target: all spheres should be at least 2*r apart, but we don't know r yet
            # Instead, let's use a repulsion force approach
            
            # Force from other spheres: push away from nearest neighbors
            forces = np.zeros_like(pos)
            for i in range(N):
                # Find nearest neighbor
                j = np.argmin(D[i])
                if D[i, j] < 0.01:  # Very close
                    # Strong repulsion
                    direction = pos[i] - pos[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    else:
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                    forces[i] += 0.01 * direction
                else:
                    # Gentle repulsion for all neighbors within some range
                    for j in range(N):
                        if i == j:
                            continue
                        d = D[i, j]
                        if d < 0.3:  # Interaction range
                            direction = pos[i] - pos[j]
                            norm = np.linalg.norm(direction)
                            if norm > 1e-10:
                                direction /= norm
                            else:
                                continue
                            # Repulsion strength decreases with distance
                            strength = 0.005 * (1.0 - d / 0.3)
                            forces[i] += strength * direction
            
            # Also add a small force to keep spheres inside the cube
            for i in range(3):
                # If too close to boundary, push inward
                mask_pos = pos[:, i] > 0.4
                mask_neg = pos[:, i] < -0.4
                forces[mask_pos, i] -= 0.01
                forces[mask_neg, i] += 0.01
            
            # Update positions with a decaying step size
            step = 0.05 * (0.995 ** it)
            pos += step * forces
            
            # Clamp to cube
            pos = np.clip(pos, -0.49, 0.49)
            
            # Check if converged
            if np.max(np.abs(forces)) < 1e-6:
                break
        
        # Compute radius for this configuration
        r = _compute_radius(pos)
        if r > best_radius:
            best_radius = r
            best_centers = pos.copy()
    
    return best_centers if best_centers is not None else np.random.uniform(-0.4, 0.4, size=(N, 3))
# EVOLVE-BLOCK-END
