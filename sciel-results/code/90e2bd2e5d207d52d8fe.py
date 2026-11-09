"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_lattice_init(n):
    """Generate a hexagonal lattice configuration clipped to the unit disk."""
    # Estimate spacing: area per circle ~ pi/(n*sqrt(3)/2) for hex packing density
    # Actually, hexagonal packing density is pi/(2*sqrt(3)) ~ 0.9069
    # For n circles of radius r in unit disk: n * pi * r^2 ~ pi * 1^2 * 0.9069
    # So r ~ sqrt(0.9069/n)
    # Spacing between centers ~ 2*r
    spacing = 2.0 * np.sqrt(0.9069 / n)
    
    # Generate hexagonal lattice points
    points = []
    # Range in x and y
    half_range = 1.2 / spacing + 2
    kx = int(np.ceil(half_range))
    ky = int(np.ceil(half_range))
    
    for i in range(-kx, kx + 1):
        for j in range(-ky, ky + 1):
            x = i * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if j % 2 != 0:
                x += spacing / 2.0
            r = np.sqrt(x*x + y*y)
            if r <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n, 2))
    
    # If too many points, select n closest to origin
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # Pad with points near the edge
        while len(points) < n:
            angle = np.random.uniform(0, 2*np.pi)
            r = 0.95
            points = np.vstack([points, [r*np.cos(angle), r*np.sin(angle)]])
    
    return points


def _compute_score(points):
    """Compute the packing score: min of half-min-pair-distance and min-boundary-distance."""
    n = len(points)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances (use chunking for memory)
    min_pair_dist = np.inf
    # For 300 points, full pairwise is 300*299/2 = 44850 pairs, manageable
    for i in range(n - 1):
        diffs = points[i+1:] - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = min(min_pair_dist, np.min(dists))
    
    # Boundary constraint: distance from each point to unit circle boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _repulsion_forces(points, target_r, sigma):
    """Compute forces from repulsion between overlapping circles and boundary."""
    n = len(points)
    forces = np.zeros_like(points)
    
    # Pairwise repulsion
    for i in range(n - 1):
        diff = points[i] - points[i+1:]
        dists = np.linalg.norm(diff, axis=1)
        # Avoid division by zero
        mask = dists > 1e-10
        if np.any(mask):
            # Repulsive force when dist < 2*target_r
            overlap = 2*target_r - dists[mask]
            positive_overlap = np.maximum(overlap, 0)
            # Force magnitude proportional to overlap (linear)
            force_mag = positive_overlap * (positive_overlap > 0)
            # Normalize direction
            dirs = diff[mask] / dists[mask][:, np.newaxis]
            forces[i] += np.sum(force_mag[:, np.newaxis] * dirs, axis=0)
            forces[i+1:][mask] -= force_mag[:, np.newaxis] * dirs
    
    # Boundary repulsion: push points inward if they go outside
    norms = np.linalg.norm(points, axis=1)
    outside = norms > (1.0 - target_r)
    if np.any(outside):
        boundary_force = (norms[outside] - (1.0 - target_r))[:, np.newaxis]
        # Direction is inward (toward origin)
        inward_dirs = -points[outside] / norms[outside][:, np.newaxis]
        forces[outside] += boundary_force * inward_dirs
    
    return forces


def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Initial configuration from hexagonal lattice
    np.random.seed(42)
    points = _hex_lattice_init(N)
    
    # Compute initial score
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Start with a slightly smaller radius and gradually increase
    # Target radius starts below the current achievable
    r = max(0.001, current_score * 0.9)
    
    # Adaptive step size
    step_size = 0.01
    
    # Iterative relaxation
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Compute forces with current target radius
        forces = _repulsion_forces(points, r, sigma=1.0)
        
        # Add small random perturbation to escape local minima
        noise = np.random.normal(0, 0.001 * step_size, size=points.shape)
        
        # Update positions
        new_points = points + step_size * forces + noise
        
        # Check if new configuration is valid (all points in unit disk)
        new_norms = np.linalg.norm(new_points, axis=1)
        if np.all(new_norms <= 1.0):
            new_score = _compute_score(new_points)
            
            if new_score >= r * 0.99:  # Accept if close to target
                points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                
                # Gradually increase target radius
                r = min(r * 1.001, current_score * 1.01)
                # Decay step size slowly
                step_size *= 0.999
            else:
                # Rejected: increase noise
                step_size *= 1.05
                # Try slightly lower target
                r = max(0.001, r * 0.995)
        else:
            # Points went outside, reduce step size
            step_size *= 0.9
            # Revert
            new_points = points
        
        # Periodically re-initialize noise magnitude
        if iteration % 1000 == 0:
            step_size = max(0.001, min(step_size, 0.02))
    
    return best_points
# EVOLVE-BLOCK-END
