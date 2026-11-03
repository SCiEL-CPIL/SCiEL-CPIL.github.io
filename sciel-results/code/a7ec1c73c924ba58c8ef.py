"""Packing 300 equal circles in the unit disk.
Strategy:
1. Generate initial positions using a hexagonal lattice.
2. Use scipy.optimize.minimize with SLSQP to maximize minimum distance with explicit constraints.
3. Polish with a soft-min gradient ascent for fine-tuning.
4. Use a deadline guard to ensure termination within 120s.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin before 120s limit

    # --- Step 1: Generate good initial positions ---
    def hexagonal_positions(n_target):
        """Generate n_target points in a hexagonal pattern within the unit disk."""
        base_spacing = 1.0 / np.sqrt(n_target * 2.0 / np.pi)
        
        best_points = None
        best_count_diff = float('inf')
        
        for spacing_mult in np.linspace(0.85, 1.15, 20):
            spacing = base_spacing * spacing_mult
            
            a = spacing
            b = a * np.sqrt(3) / 2
            k = int(1.0 / spacing) + 5
            points = []
            for j in range(-k, k + 1):
                y = j * b
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * a
                    if x*x + y*y <= 0.98**2:
                        points.append((x, y))
            
            points = np.array(points)
            if len(points) == 0:
                continue
            
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
            points = points[idx]
            
            if len(points) >= n_target:
                points = points[:n_target]
                diff = abs(len(points) - n_target)
                if diff < best_count_diff:
                    best_count_diff = diff
                    best_points = points
            
            if len(points) == n_target:
                best_points = points
                break
        
        if best_points is None or len(best_points) != n_target:
            angles = np.linspace(0, 2*np.pi, n_target, endpoint=False)
            radii = np.linspace(0.05, 0.95, n_target)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        return best_points
    
    positions = hexagonal_positions(N)
    
    def compute_radius(pos):
        """Compute the current minimum radius."""
        boundary = 1.0 - np.linalg.norm(pos, axis=1)
        min_boundary = np.min(boundary)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist) / 2.0
        
        return min(min_boundary, min_pair_dist)
    
    current_r = compute_radius(positions)
    if current_r < 0.04:
        max_norm = np.max(np.linalg.norm(positions, axis=1))
        if max_norm > 0:
            positions = positions * (0.95 / max_norm)
        current_r = compute_radius(positions)
    
    best_positions = positions.copy()
    best_r = current_r

    # --- Step 2: SLSQP optimization with explicit constraints ---
    # We'll maximize t such that:
    #   ||p_i - p_j|| >= 2*t for all i < j
    #   1 - ||p_i|| >= t for all i
    # This is equivalent to maximizing the minimum of {||p_i-p_j||/2, 1-||p_i||}
    
    # For SLSQP, we minimize -t with constraints:
    #   g_ij = ||p_i - p_j||^2 - (2*t)^2 >= 0
    #   h_i = 1 - ||p_i||^2 - t^2 >= 0 (approximation, or use 1-||p_i|| - t >= 0)
    
    # Actually, let's use a simpler formulation:
    # Minimize -t
    # Subject to: ||p_i - p_j|| >= 2*t, 1 - ||p_i|| >= t
    
    # This is non-convex but SLSQP can handle it locally.
    
    def slsqp_objective(x):
        """Objective: -t (we want to maximize t, so minimize -t)"""
        t = x[-1]
        return -t
    
    def slsqp_grad_objective(x):
        """Gradient of objective"""
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad
    
    # Build constraint functions
    # To keep the number of constraints manageable, we'll use a neighbor-based approach
    # and periodically update the neighbor list
    
    def build_neighbor_list(pos, factor=3.0):
        """Build a list of nearby pairs for constraint generation."""
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        threshold = min_d * factor
        i_idx, j_idx = np.where(dists < threshold)
        mask = i_idx < j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        if len(i_idx) == 0:
            i_idx, j_idx = np.triu_indices(N, k=1)
        return i_idx, j_idx
    
    i_idx, j_idx = build_neighbor_list(positions)
    
    def slsqp_constraints(x):
        """Return list of constraint dictionaries."""
        pos = x[:-1].reshape(N, 2)
        t = x[-1]
        
        constraints = []
        
        # Pairwise distance constraints: ||p_i - p_j||^2 - (2*t)^2 >= 0
        # Only for neighbor pairs to keep constraint count manageable
        diff = pos[i_idx] - pos[j_idx]
        dist_sq = np.sum(diff**2, axis=1)
        min_dist_sq = (2*t)**2
        
        # Only include constraints that are close to being violated
        # to keep the number of constraints reasonable
        violated = dist_sq - min_dist_sq < 0.1 * min_dist_sq
        if np.any(violated):
            for idx in np.where(violated)[0]:
                i, j = i_idx[idx], j_idx[idx]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sum((x[2*i:2*i+2] - x[2*j:2*j+2])**2) - (2*x[-1])**2,
                    'jac': lambda x, i=i, j=j: _pair_jac(x, i, j)
                })
        
        # Boundary constraints: 1 - ||p_i|| - t >= 0
        rad = np.linalg.norm(pos, axis=1)
        boundary_violated = rad + t > 0.95
        if np.any(boundary_violated):
            for idx in np.where(boundary_violated)[0]:
                i = idx
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: 1.0 - np.linalg.norm(x[2*i:2*i+2]) - x[-1],
                    'jac': lambda x, i=i: _boundary_jac(x, i)
                })
        
        # If no constraints are close to being violated, add a few to guide the optimizer
        if len(constraints) == 0:
            # Add the tightest constraints
            diff = pos[i_idx] - pos[j_idx]
            dist_sq = np.sum(diff**2, axis=1)
            # Find the 10 tightest pairwise constraints
            n_tight = min(10, len(dist_sq))
            tight_idx = np.argsort(dist_sq)[:n_tight]
            for idx in tight_idx:
                i, j = i_idx[idx], j_idx[idx]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sum((x[2*i:2*i+2] - x[2*j:2*j+2])**2) - (2*x[-1])**2,
                    'jac': lambda x, i=i, j=j: _pair_jac(x, i, j)
                })
            
            # Add the tightest boundary constraints
            rad = np.linalg.norm(pos, axis=1)
            tight_boundary = np.argsort(rad)[:5]
            for idx in tight_boundary:
                i = idx
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: 1.0 - np.linalg.norm(x[2*i:2*i+2]) - x[-1],
                    'jac': lambda x, i=i: _boundary_jac(x, i)
                })
        
        return constraints
    
    def _pair_jac(x, i, j):
        """Jacobian of pairwise constraint ||p_i - p_j||^2 - (2*t)^2 >= 0"""
        pos = x[:-1].reshape(N, 2)
        t = x[-1]
        grad = np.zeros_like(x)
        diff = pos[i] - pos[j]
        grad[2*i:2*i+2] = 2*diff
        grad[2*j:2*j+2] = -2*diff
        grad[-1] = -8*t
        return grad
    
    def _boundary_jac(x, i):
        """Jacobian of boundary constraint 1 - ||p_i|| - t >= 0"""
        pos = x[:-1].reshape(N, 2)
        t = x[-1]
        grad = np.zeros_like(x)
        rad = np.linalg.norm(pos[i])
        if rad > 1e-9:
            grad[2*i:2*i+2] = -pos[i] / rad
        grad[-1] = -1.0
        return grad
    
    # Run SLSQP optimization
    x0 = np.concatenate([positions.flatten(), [current_r]])
    
    # Set bounds: positions in [-1, 1], t in [0, 1]
    bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 1.0)]
    
    slsqp_success = False
    slsqp_result = None
    
    try:
        # Use SLSQP with a reasonable number of iterations
        slsqp_result = minimize(
            slsqp_objective,
            x0,
            method='SLSQP',
            jac=slsqp_grad_objective,
            constraints=slsqp_constraints(x0),
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if slsqp_result.success:
            new_pos = slsqp_result.x[:-1].reshape(N, 2)
            new_r = compute_radius(new_pos)
            if new_r > best_r:
                best_positions = new_pos.copy()
                best_r = new_r
                slsqp_success = True
    except Exception:
        pass
    
    # --- Step 3: Soft-min gradient polish ---
    # Use the best positions so far as starting point
    positions = best_positions.copy()
    
    # Rebuild neighbor list
    i_idx, j_idx = build_neighbor_list(positions)
    
    alpha = 0.03
    cooling = 0.999
    k_soft = 80.0
    k_polish = 300.0
    steps_per_k = 500
    
    step_count = 0
    patience = 0
    max_patience = 3000
    
    while time.time() - start_time < deadline:
        if step_count > 0 and step_count % steps_per_k == 0:
            k_soft = min(k_soft * 1.15, k_polish)
        
        diff = positions[i_idx] - positions[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        safe_dists = np.maximum(dists, 1e-9)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        
        rad = np.linalg.norm(positions, axis=1)
        b_vals = 1.0 - rad
        
        pair_vals = dists / 2.0
        all_vals = np.concatenate([pair_vals, b_vals])
        
        v_min = np.min(all_vals)
        exponents = -k_soft * (all_vals - v_min)
        weights = np.exp(exponents)
        sum_weights = np.sum(weights)
        
        if sum_weights < 1e-9:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= sum_weights
            
        pair_weights = weights[:len(dists)]
        bound_weights = weights[len(dists):]
        
        grads = np.zeros_like(positions)
        
        half_unit_vecs = 0.5 * unit_vecs
        np.add.at(grads, i_idx, half_unit_vecs * pair_weights[:, np.newaxis])
        np.add.at(grads, j_idx, -half_unit_vecs * pair_weights[:, np.newaxis])
        
        safe_rad = np.maximum(rad, 1e-9)
        bound_vecs = -positions / safe_rad[:, np.newaxis]
        grads += bound_vecs * bound_weights[:, np.newaxis]
        
        positions += alpha * grads
        
        rad_new = np.linalg.norm(positions, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            positions[mask] /= rad_new[mask, np.newaxis]
        
        current_r = compute_radius(positions)
        
        if current_r > best_r:
            best_r = current_r
            best_positions = positions.copy()
            patience = 0
        else:
            patience += 1
            
        if patience > max_patience:
            break
            
        alpha *= cooling
        step_count += 1
        
        if step_count % 5000 == 0 and time.time() - start_time < deadline - 15:
            i_idx, j_idx = build_neighbor_list(positions)

    final_r = compute_radius(best_positions)
    if final_r > compute_radius(positions):
        return best_positions
    else:
        return positions

def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    norms = np.linalg.norm(pos, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
        
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-4:
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-4:
                    pos[i] += np.random.randn(2) * 1e-3
                    pos[j] -= np.random.randn(2) * 1e-3
                    n = np.linalg.norm(pos[i])
                    if n > 0.999:
                        pos[i] = pos[i] * (0.999 / n)
                    n = np.linalg.norm(pos[j])
                    if n > 0.999:
                        pos[j] = pos[j] * (0.999 / n)
                        
    return pos

_original_construct = construct

def construct():
    try:
        pos = _original_construct()
        pos = _validate_and_fix(pos)
        return pos
    except Exception:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.05, 0.95, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pos
# EVOLVE-BLOCK-END
