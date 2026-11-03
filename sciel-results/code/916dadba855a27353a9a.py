"""Packing 300 equal circles in the unit disk.
Strategy:
1. Generate initial positions using a hexagonal lattice with radial compression.
2. Refine with a vectorized force-based optimization (repulsion + boundary constraint).
3. Use a deadline guard to ensure termination within 120s.
"""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill

    # --- Step 1: Generate good initial positions ---
    # Use a hexagonal lattice with a slight radial compression to fit N points
    # inside the unit disk.
    
    def hexagonal_positions(n_target):
        """Generate n_target points in a hexagonal pattern within the unit disk."""
        # Estimate spacing based on area: N * pi * r^2 <= pi * 1^2 => r <= 1/sqrt(N)
        # Hexagonal packing density is pi/(2*sqrt(3)) ≈ 0.9069
        # So effective area per point in hex packing: (2*sqrt(3)/pi) * (1/N)
        # Linear dimension ~ sqrt(1/N)
        base_spacing = 1.0 / np.sqrt(n_target * 2.0 / np.pi)
        
        best_points = None
        best_count_diff = float('inf')
        
        # Try a range of spacings around the theoretical base
        for spacing_mult in np.linspace(0.85, 1.15, 20):
            spacing = base_spacing * spacing_mult
            
            # Generate hex lattice points
            a = spacing
            b = a * np.sqrt(3) / 2
            k = int(1.0 / spacing) + 5
            points = []
            for j in range(-k, k + 1):
                y = j * b
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * a
                    if x*x + y*y <= 0.98**2:  # Keep well inside for now
                        points.append((x, y))
            
            points = np.array(points)
            if len(points) == 0:
                continue
            
            # Sort by distance from center
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
            points = points[idx]
            
            if len(points) >= n_target:
                points = points[:n_target]
                diff = abs(len(points) - n_target)
                if diff < best_count_diff:
                    best_count_diff = diff
                    best_points = points
            
            # If we have exactly n_target or very close, use it
            if len(points) == n_target:
                best_points = points
                break
        
        if best_points is None or len(best_points) != n_target:
            # Fallback: create a simple radial pattern
            angles = np.linspace(0, 2*np.pi, n_target, endpoint=False)
            radii = np.linspace(0.05, 0.95, n_target)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        return best_points
    
    positions = hexagonal_positions(N)
    
    # --- Step 2: Force-based optimization ---
    # Objective: maximize r where r = min(min_{i<j} ||p_i - p_j||/2, min_i(1-||p_i||))
    
    def compute_radius(pos):
        """Compute the current minimum radius."""
        # Distance to boundary
        boundary = 1.0 - np.linalg.norm(pos, axis=1)
        min_boundary = np.min(boundary)
        
        # Pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Set diagonal to infinity
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist) / 2.0
        
        return min(min_boundary, min_pair_dist)
    
    # Start with a reasonable target radius
    current_r = compute_radius(positions)
    if current_r < 0.04:
        # Scale positions to ensure we have a valid starting point
        max_norm = np.max(np.linalg.norm(positions, axis=1))
        if max_norm > 0:
            positions = positions * (0.95 / max_norm)
        current_r = compute_radius(positions)
    
    best_positions = positions.copy()
    best_r = current_r
    
    # Optimization loop
    lr = 0.0005  # Learning rate
    patience = 0
    max_patience = 2000  # Increased patience for better convergence
    
    # Pre-compute indices for pairwise operations if needed, but vectorization is fast enough
    for step in range(200000):
        if time.time() > deadline:
            break
        
        pos = positions
        
        # Compute gradients
        # For each point, the gradient of the objective w.r.t. position
        # The objective is the minimum of many terms, so we focus on the most constraining ones
        
        # Boundary constraint gradient
        norms = np.linalg.norm(pos, axis=1)
        boundary_terms = 1.0 - norms
        
        # Pairwise distance terms
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        pair_terms = dist / 2.0
        
        # Find the minimum
        min_boundary = np.min(boundary_terms)
        min_pair = np.min(pair_terms)
        
        current_r = min(min_boundary, min_pair)
        
        # Update best
        if current_r > best_r:
            best_r = current_r
            best_positions = pos.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary force: push inward if close to boundary
        # Gradient of (1 - ||p_i||) w.r.t. p_i is -p_i/||p_i||
        # We want to maximize this, so we add to forces in the direction of the gradient
        # But we are minimizing a loss, or maximizing the radius.
        # Let's use a simple repulsion model.
        
        # 1. Boundary repulsion
        # If a point is close to the boundary (norm > 1 - r_target), push it in.
        # Here we just push all points slightly inward proportional to their norm?
        # No, better to only push those that are limiting.
        # But a global inward force helps compact the center.
        
        # Let's use a soft potential approach.
        # Potential U = sum_{i<j} 1/d_ij^2 + sum_i (1/2) * max(0, ||p_i|| - (1-r))^2 ?
        # Simpler: Repulsive force between all pairs + Boundary force.
        
        # Pairwise repulsion
        # Force on i from j: (p_i - p_j) / ||p_i - p_j||^3 * strength
        # Vectorized:
        # diff is (N, N, 2)
        # dist is (N, N)
        
        # Avoid division by zero
        dist_safe = np.where(dist < 1e-6, 1e-6, dist)
        
        # Force magnitude: 1/d^2
        # Direction: diff / d
        # So force vector = diff / d^3
        # But we only want to consider pairs that are close?
        # For N=300, full matrix is 90k elements, which is small.
        
        # force_mag[i,j] = 1 / dist[i,j]^2
        # force_vec[i,j] = diff[i,j] * force_mag[i,j] / dist[i,j] = diff[i,j] / dist[i,j]^3
        
        # Let's compute forces for all pairs
        # This might be heavy if done naively, but N=300 is fine.
        
        # dist_safe is (N, N)
        # diff is (N, N, 2)
        
        # We want F_i = sum_j (p_i - p_j) / ||p_i - p_j||^3
        # This is equivalent to:
        # F = sum_j (diff[i,j] / dist_safe[i,j]^3)
        
        # Create a weight matrix W where W[i,j] = 1/dist_safe[i,j]^3
        # But we need to be careful with the diagonal (i=j) which is inf.
        # We set diagonal to 0 in W.
        
        # Actually, computing 1/d^3 for all pairs might be unstable if d is very small.
        # We already clipped dist to 1e-6.
        
        # Let's try a different approach:
        # Only compute forces for pairs closer than a certain threshold?
        # Or just use a global repulsion that is strong for close pairs.
        
        # Let's use the gradient of the objective function directly.
        # Objective: r = min(terms)
        # If we assume the minimum is determined by a specific pair or boundary,
        # the gradient is the gradient of that term.
        # But we don't know which one.
        
        # Alternative: Use a smooth approximation.
        # Minimize sum of exp(-k * term) or similar?
        
        # Let's stick to the force-based method but make it more robust.
        
        # 1. Boundary Force
        # Push points inward if they are outside a safe radius.
        # Safe radius = 1 - current_r - epsilon?
        # Or just always push inward with a small force to keep them compact.
        
        # Let's use: F_boundary = -pos / ||pos|| * (||pos|| - 0.9)^2 if ||pos|| > 0.9
        # This creates a soft wall at 0.9.
        
        # 2. Pairwise Repulsion
        # F_pair = sum_j (p_i - p_j) / ||p_i - p_j||^3
        
        # Let's compute F_pair vectorized.
        # diff: (N, N, 2)
        # dist: (N, N)
        
        # We want to sum over j for each i.
        # F_i = sum_j (diff[i,j] / dist[i,j]^3)
        
        # dist_inv3 = 1 / dist_safe^3
        # dist_inv3[diag] = 0
        dist_inv3 = 1.0 / (dist_safe ** 3)
        np.fill_diagonal(dist_inv3, 0.0)
        
        # F_pair = diff * dist_inv3[:, :, np.newaxis]
        # Sum over axis 1 (j)
        F_pair = np.sum(diff * dist_inv3[:, :, np.newaxis], axis=1)
        
        # Boundary Force
        # F_boundary = -pos * (norms - 0.95)^2 / norms  (if norms > 0.95)
        # Let's use a smoother potential.
        # U_boundary = 0.5 * (max(0, ||p|| - 0.95))^2
        # F_boundary = -dU/dp = - (||p|| - 0.95) * p/||p||  (if ||p|| > 0.95)
        
        F_boundary = np.zeros_like(pos)
        mask = norms > 0.95
        if np.any(mask):
            # Gradient of 0.5 * (||p|| - 0.95)^2 is (||p|| - 0.95) * p/||p||
            # We want to minimize this, so force is negative gradient.
            # F = - (||p|| - 0.95) * p/||p||
            factor = (norms[mask] - 0.95) / norms[mask]
            F_boundary[mask] = -pos[mask] * factor[:, np.newaxis]
        
        # Total Force
        forces = F_pair * 0.001 + F_boundary * 0.1
        
        # Update positions
        positions = pos + lr * forces
        
        # Keep positions inside the unit disk
        norms = np.linalg.norm(positions, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            positions[mask] = positions[mask] * (0.999 / norms[mask])
        
        # Reduce learning rate over time
        lr = max(0.00005, lr * 0.9999)
    
    # Final check
    final_r = compute_radius(best_positions)
    if final_r > compute_radius(positions):
        return best_positions
    else:
        return positions

# Ensure the function returns a valid array
def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        # Fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Check for non-finite values
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    # Check inside unit disk
    norms = np.linalg.norm(pos, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
        
    # Check for coincident points
    # If any pair is too close, jitter them
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-4:
        # Find indices of close pairs and jitter
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-4:
                    pos[i] += np.random.randn(2) * 1e-3
                    pos[j] -= np.random.randn(2) * 1e-3
                    # Re-check boundary
                    n = np.linalg.norm(pos[i])
                    if n > 0.999:
                        pos[i] = pos[i] * (0.999 / n)
                    n = np.linalg.norm(pos[j])
                    if n > 0.999:
                        pos[j] = pos[j] * (0.999 / n)
                        
    return pos

# Wrap the construct function to ensure validity
_original_construct = construct

def construct():
    try:
        pos = _original_construct()
        pos = _validate_and_fix(pos)
        return pos
    except Exception:
        # Fallback to a simple valid configuration
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.05, 0.95, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pos
# EVOLVE-BLOCK-END
