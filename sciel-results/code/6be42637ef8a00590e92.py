"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _compute_radius(points):
    """Compute the maximum common radius for given centers."""
    if len(points) < 2:
        return 0.0
    # Distance to boundary
    dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    return min(min_boundary, min_pair)

def _radial_ring_init():
    """
    Initialize points using a radial-ring decomposition.
    Place circles in concentric rings, optimizing the number of circles per ring
    and the radial positions to maximize the minimum distance.
    """
    # We'll try to distribute 300 circles across several rings.
    # The center circle is at the origin.
    # Then rings of increasing radius.
    
    # Let's start with a heuristic: 
    # Ring 0: 1 circle at center
    # Ring 1: 6 circles
    # Ring 2: 12 circles
    # Ring 3: 18 circles
    # etc.
    # But we need to fit them in a unit circle.
    
    # A better approach: 
    # Try different numbers of rings and distribute circles accordingly.
    # For each configuration, compute the optimal radius and positions.
    
    best_points = None
    best_radius = 0.0
    
    # Try configurations with 5-15 rings
    for num_rings in range(5, 16):
        # Distribute circles among rings
        # First ring (innermost) gets fewer circles, outer rings get more
        # Use a distribution proportional to the circumference
        # r_i ≈ i / num_rings
        # circumference ∝ r_i
        # So circles per ring ∝ i
        
        # But the center is special
        # Let's say ring 0 is the center (1 circle)
        # Rings 1 to num_rings-1 are the other rings
        
        # Total circles = 1 + sum_{i=1}^{num_rings-1} n_i = 300
        # So sum_{i=1}^{num_rings-1} n_i = 299
        
        # Let's distribute 299 circles among num_rings-1 rings
        # proportional to i (circumference)
        weights = np.arange(1, num_rings)
        total_weight = np.sum(weights)
        n_per_ring = np.round(weights / total_weight * 299).astype(int)
        
        # Adjust to make sure the sum is exactly 299
        diff = 299 - np.sum(n_per_ring)
        # Add or remove from the largest rings
        indices = np.argsort(n_per_ring)[::-1]
        idx = 0
        while diff != 0:
            if diff > 0:
                n_per_ring[indices[idx % len(indices)]] += 1
                diff -= 1
            else:
                if n_per_ring[indices[idx % len(indices)]] > 1:
                    n_per_ring[indices[idx % len(indices)]] -= 1
                    diff += 1
            idx += 1
            if idx > 1000:
                break
        
        # Now we have n_per_ring for rings 1 to num_rings-1
        # Ring 0 has 1 circle at center
        
        # Optimize the radial positions
        # Let r_i be the radius of ring i
        # We want to maximize the minimum distance between circles
        
        # For a ring with n circles at radius r, the distance between adjacent circles is 2*r*sin(pi/n)
        # The distance between rings i and i+1 is approximately r_{i+1} - r_i (if aligned) or sqrt(r_i^2 + r_{i+1}^2 - 2*r_i*r_{i+1}*cos(angle_diff))
        
        # For simplicity, let's assume the rings are aligned (angles are multiples of 2*pi/n_i)
        # The minimum distance will be the minimum of:
        # 1. 2*r_i*sin(pi/n_i) for each ring
        # 2. r_{i+1} - r_i for adjacent rings (if the angular offset allows)
        # 3. 1 - r_{max} for the outermost ring
        
        # Let's set up an optimization problem to find r_i that maximizes the minimum distance
        
        # Variables: r_1, r_2, ..., r_{num_rings-1}
        # Objective: maximize min(2*r_i*sin(pi/n_i), r_{i+1}-r_i, 1-r_{num_rings-1})
        
        # This is a max-min problem. We can solve it by binary search on the radius.
        # Or use scipy.optimize to maximize the minimum.
        
        # Let's use a simple approach: 
        # Start with evenly spaced radii, then optimize
        
        r_values = np.linspace(0.1, 0.9, num_rings - 1)
        
        # Define the objective function
        def neg_min_radius(r_vec):
            r_vec = np.abs(r_vec)  # Ensure positive
            if np.any(r_vec < 0.01) or np.any(r_vec > 0.99):
                return 100.0
            
            min_dist = np.inf
            
            # Within each ring
            for i, r in enumerate(r_vec):
                n = n_per_ring[i]
                if n >= 2:
                    dist = 2 * r * np.sin(np.pi / n)
                    min_dist = min(min_dist, dist)
            
            # Between adjacent rings
            for i in range(len(r_vec) - 1):
                min_dist = min(min_dist, r_vec[i+1] - r_vec[i])
            
            # From center to first ring
            min_dist = min(min_dist, r_vec[0])
            
            # From outermost ring to boundary
            min_dist = min(min_dist, 1.0 - r_vec[-1])
            
            return -min_dist
        
        # Optimize
        result = minimize(neg_min_radius, r_values, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.success or result.fun < 0:
            r_opt = np.abs(result.x)
            
            # Check if this configuration is valid
            if np.all(r_opt > 0.01) and np.all(r_opt < 0.99):
                # Generate the points
                points = [np.array([0.0, 0.0])]  # Center
                
                for i, r in enumerate(r_opt):
                    n = n_per_ring[i]
                    # Add angular offset to stagger the rings
                    offset = (i % 2) * np.pi / n
                    angles = offset + np.arange(n) * 2 * np.pi / n
                    ring_points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
                    points.append(ring_points)
                
                points = np.vstack(points)
                
                # Compute the actual radius
                radius = _compute_radius(points)
                
                if radius > best_radius:
                    best_radius = radius
                    best_points = points
        
        # Check time
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
    
    if best_points is None:
        # Fallback to hexagonal lattice
        return _hex_lattice_init()
    
    return best_points

def _hex_lattice_init():
    """Fallback: hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    return points

def _local_optimize(points, max_iter=2000):
    """
    Local optimization using gradient ascent on the soft-min of distances.
    """
    n = len(points)
    points = points.copy()
    
    t = 100.0  # Temperature parameter
    lr = 0.001
    start_time = time.time()
    
    iu = np.triu_indices(n, k=1)
    
    def compute_grad(pos):
        pos = pos.reshape(n, 2)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        boundary_dists = 1.0 - np.linalg.norm(pos, axis=1)
        
        pair_dists = dist_matrix[iu]
        all_dists = np.concatenate([pair_dists, boundary_dists])
        
        d_min = np.min(all_dists)
        exp_terms = np.exp(-t * (all_dists - d_min))
        S = np.sum(exp_terms)
        
        grad = np.zeros_like(pos)
        
        pair_weights = exp_terms[:len(pair_dists)] / S
        diff_pairs = diff[iu]
        norm_pairs = pair_dists[:, np.newaxis]
        norm_pairs[norm_pairs == 0] = 1e-12
        unit_vecs = diff_pairs / norm_pairs
        
        i_idx, j_idx = iu
        np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * unit_vecs)
        
        boundary_weights = exp_terms[len(pair_dists):] / S
        norms = np.linalg.norm(pos, axis=1)[:, np.newaxis]
        norms[norms == 0] = 1e-12
        unit_pos = pos / norms
        grad += boundary_weights[:, np.newaxis] * unit_pos
        
        return grad, d_min
    
    for step in range(max_iter):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        grad, d_min = compute_grad(points)
        
        points_new = points + lr * grad
        
        # Simple line search: check if the new point is better
        new_radius = _compute_radius(points_new)
        old_radius = _compute_radius(points)
        
        if new_radius > old_radius:
            points = points_new
            lr *= 1.01
        else:
            lr *= 0.95
        
        if lr < 1e-10:
            break
    
    return points

def construct():
    start_time = time.time()
    
    # Try radial-ring initialization
    points = _radial_ring_init()
    
    # Local optimization
    if time.time() < DEADLINE - SAFETY_MARGIN:
        points = _local_optimize(points)
    
    # Ensure we have exactly N points
    if len(points) != N:
        # Fallback to hexagonal lattice
        points = _hex_lattice_init()
        if time.time() < DEADLINE - SAFETY_MARGIN:
            points = _local_optimize(points)
    
    # Final check: ensure all points are within the unit circle
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms > 1.0):
        # Scale down if necessary
        max_norm = np.max(norms)
        if max_norm > 1.0:
            points = points * (0.999 / max_norm)
    
    return points
# EVOLVE-BLOCK-END
