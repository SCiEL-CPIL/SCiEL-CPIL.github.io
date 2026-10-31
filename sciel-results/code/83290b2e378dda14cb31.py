"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, penalty_weight=5000.0):
    """
    Objective: minimize negative sum of radii + smooth penalty for violations.
    Uses a high penalty weight to strictly enforce constraints.
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-4)
    
    penalty = 0.0
    # Boundary penalties (smooth)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    penalty += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Pairwise overlap penalties (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    # Only consider upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    # Use a smoother penalty: scale by min_dist to normalize
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * np.maximum(min_dist_sq[mask], 1e-8)))
    
    return -np.sum(radii) + penalty_weight * penalty


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.098
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    
    pos[25, 0] = 0.90
    pos[25, 1] = 0.90
    radii[25] = 0.08
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern with optimized spacing."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    y_spacing = r * np.sqrt(3)
    
    total_height = 4 * y_spacing + 2 * r
    y_start = (1.0 - total_height) / 2.0
    
    x_start_5 = (1.0 - 4 * 2 * r) / 2.0
    for i in range(5):
        idx = i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r
        radii[idx] = r
    
    x_start_4 = (1.0 - 3 * 2 * r) / 2.0
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = x_start_4 + i * 2 * r
        pos[idx, 1] = y_start + r + y_spacing
        radii[idx] = r
    
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r + 2 * y_spacing
        radii[idx] = r
    
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = x_start_4 + i * 2 * r
        pos[idx, 1] = y_start + r + 3 * y_spacing
        radii[idx] = r
    
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r + 4 * y_spacing
        radii[idx] = r
    
    # 26th circle: find best position in gaps
    best_score = -1
    best_pos = (0.5, 0.5)
    for cx in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]:
        for cy in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]:
            min_clearance = 1.0
            for i in range(25):
                dist = np.sqrt((cx - pos[i, 0])**2 + (cy - pos[i, 1])**2)
                clearance = dist - radii[i]
                min_clearance = min(min_clearance, clearance)
            min_clearance = min(min_clearance, cx, 1-cx, cy, 1-cy)
            if min_clearance > best_score:
                best_score = min_clearance
                best_pos = (cx, cy)
    
    pos[25, 0] = best_pos[0]
    pos[25, 1] = best_pos[1]
    radii[25] = max(best_score * 0.9, 0.02)
    
    return pos, radii


def _initial_center_ring():
    """Center circle surrounded by a ring of circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.12
    
    ring_r = 0.08
    center_r = 0.12
    ring_dist = center_r + ring_r
    
    for i in range(12):
        idx = 1 + i
        angle = 2 * np.pi * i / 12
        pos[idx, 0] = 0.5 + ring_dist * np.cos(angle)
        pos[idx, 1] = 0.5 + ring_dist * np.sin(angle)
        radii[idx] = ring_r
    
    corner_r = 0.06
    corners = [
        (0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
        (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5),
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
        (0.5, 0.5)
    ]
    for i in range(13):
        idx = 13 + i
        pos[idx, 0] = corners[i][0]
        pos[idx, 1] = corners[i][1]
        radii[idx] = corner_r
    
    return pos, radii


def _initial_corner_heavy():
    """Configuration emphasizing corners and edges for better packing density."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Four large corner circles
    corner_r = 0.14
    corners = [(corner_r, corner_r), (1-corner_r, corner_r),
               (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]
    for i, (cx, cy) in enumerate(corners):
        pos[i, 0] = cx
        pos[i, 1] = cy
        radii[i] = corner_r
    
    # Edge circles (top, bottom, left, right)
    edge_r = 0.10
    edges = [
        (0.5, edge_r), (0.5, 1-edge_r),
        (edge_r, 0.5), (1-edge_r, 0.5),
        (0.3, edge_r), (0.7, edge_r),
        (0.3, 1-edge_r), (0.7, 1-edge_r),
        (edge_r, 0.3), (edge_r, 0.7),
        (1-edge_r, 0.3), (1-edge_r, 0.7),
    ]
    for i, (cx, cy) in enumerate(edges):
        idx = 4 + i
        pos[idx, 0] = cx
        pos[idx, 1] = cy
        radii[idx] = edge_r
    
    # Center region circles
    center_r = 0.09
    center_positions = [
        (0.5, 0.5),
        (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),
        (0.5, 0.3), (0.5, 0.7), (0.3, 0.5), (0.7, 0.5),
    ]
    for i, (cx, cy) in enumerate(center_positions):
        idx = 16 + i
        pos[idx, 0] = cx
        pos[idx, 1] = cy
        radii[idx] = center_r
    
    # Fill remaining with smaller circles in gaps
    gap_r = 0.06
    gap_positions = [
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
    ]
    for i, (cx, cy) in enumerate(gap_positions):
        idx = 25
        if idx < N:
            pos[idx, 0] = cx
            pos[idx, 1] = cy
            radii[idx] = gap_r
    
    # Ensure we have exactly N circles
    for i in range(25, N):
        if radii[i] == 0:
            pos[i, 0] = 0.5 + 0.1 * np.cos(2*np.pi*i/N)
            pos[i, 1] = 0.5 + 0.1 * np.sin(2*np.pi*i/N)
            radii[i] = 0.05
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(300):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-4)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _grow_and_push(pos, radii, deadline, n_iter=20000):
    """
    Grow-and-Push local search.
    Iteratively tries to grow each circle to its maximum possible radius.
    If blocked, it pushes the blocking objects to create space.
    Improved: better push strategy with multiple neighbors and adaptive step size.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(12345)
    
    # Track recent moves to avoid oscillation
    recent_moves = {}
    
    for iteration in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        # Pick a random circle to grow, with some bias toward smaller circles
        # that have more room to grow
        i = rng.randint(N)
        
        # Calculate max radius for i given current positions of others
        # 1. Boundaries
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        
        # 2. Other circles
        # dist(i, j) >= r_i + r_j  => r_i <= dist(i, j) - r_j
        # We need min over j != i of (dist(i, j) - r_j)
        dx = pos[:, 0] - pos[i, 0]
        dy = pos[:, 1] - pos[i, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        limits = dist - radii
        limits[i] = np.inf # Ignore self
        
        min_pair_limit = np.min(limits)
        max_r = min(max_r, min_pair_limit)
        max_r = max(max_r, 1e-4) # Minimum radius
        
        # If we can grow, grow it
        if max_r > radii[i] + 1e-7:
            radii[i] = max_r
        else:
            # We are blocked. We need to push something to allow growth.
            # Find all blocking neighbors (within tolerance of the limit)
            blocking_indices = np.where(limits <= radii[i] + 1e-4)[0]
            blocking_indices = blocking_indices[blocking_indices != i]
            
            if len(blocking_indices) > 0:
                # Sort blockers by how much they block (tightest first)
                block_amounts = limits[blocking_indices]
                sorted_idx = blocking_indices[np.argsort(block_amounts)]
                
                # Push the top 3 blocking neighbors away from i
                for j in sorted_idx[:3]:
                    dx_ij = pos[j, 0] - pos[i, 0]
                    dy_ij = pos[j, 1] - pos[i, 1]
                    dist_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                    
                    if dist_ij < 1e-9:
                        # Random nudge
                        dx_ij = rng.normal(0, 0.01)
                        dy_ij = rng.normal(0, 0.01)
                        dist_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                    
                    # Adaptive push magnitude based on how tight the constraint is
                    tightness = max(0, (radii[i] + radii[j]) - dist_ij)
                    # Use a smoother function for push magnitude
                    push_mag = 0.002 + 0.008 * min(tightness / 0.02, 1.0) + rng.uniform(0, 0.003)
                    
                    nx = dx_ij / dist_ij
                    ny = dy_ij / dist_ij
                    
                    # Check if we've recently moved j in the same direction
                    move_key = (i, j)
                    if move_key in recent_moves and recent_moves[move_key] > 0:
                        push_mag *= 0.5  # Dampen if oscillating
                    recent_moves[move_key] = recent_moves.get(move_key, 0) + 1
                    if recent_moves[move_key] > 5:
                        recent_moves[move_key] = 0  # Reset counter
                    
                    # Move j away
                    pos[j, 0] += nx * push_mag
                    pos[j, 1] += ny * push_mag
                    
                    # Move i slightly back (less aggressive)
                    pos[i, 0] -= nx * push_mag * 0.2
                    pos[i, 1] -= ny * push_mag * 0.2
                    
                    # Clip positions to [0, 1]
                    pos[j] = np.clip(pos[j], 0, 1)
                    pos[i] = np.clip(pos[i], 0, 1)
            else:
                # Blocked by wall?
                # Check boundaries
                wall_pushed = False
                if pos[i, 0] - radii[i] < 1e-6:
                    pos[i, 0] += 0.004
                    wall_pushed = True
                elif 1.0 - pos[i, 0] - radii[i] < 1e-6:
                    pos[i, 0] -= 0.004
                    wall_pushed = True
                elif pos[i, 1] - radii[i] < 1e-6:
                    pos[i, 1] += 0.004
                    wall_pushed = True
                elif 1.0 - pos[i, 1] - radii[i] < 1e-6:
                    pos[i, 1] -= 0.004
                    wall_pushed = True
                
                if not wall_pushed:
                    # Numerical error or tight constraint, nudge randomly
                    pos[i, 0] += rng.uniform(-0.004, 0.004)
                    pos[i, 1] += rng.uniform(-0.004, 0.004)
            
            # Ensure positions are within [0, 1]
            pos = np.clip(pos, 0, 1)
            
        # Update best if valid
        if _is_valid(pos, radii, tol=1e-8):
            current_score = np.sum(radii)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            # If invalid, shrink radii slightly to restore validity and try again
            pos_pp, radii_pp = _post_process(pos, radii)
            pos = pos_pp
            radii = radii_pp
            current_score = np.sum(radii)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_radii = radii.copy()
                
    return best_pos, best_radii, best_score


def _gradient_objective(x):
    """Objective: minimize negative sum of radii (smooth, differentiable)."""
    radii = x[2*N:]
    return -np.sum(radii)


def _constraints_func(x):
    """All inequality constraints: pairwise distances and boundaries >= 0."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append(pos[i, 0] - radii[i])
        constraints.append(1.0 - pos[i, 0] - radii[i])
        constraints.append(pos[i, 1] - radii[i])
        constraints.append(1.0 - pos[i, 1] - radii[i])
    
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            constraints.append(dist - radii[i] - radii[j])
    
    return np.array(constraints)


def _jacobian_constraints(x):
    """Jacobian of all inequality constraints."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    n_constraints = 4*N + N*(N-1)//2
    jac = np.zeros((n_constraints, 2*N + N))
    
    # Boundary constraint Jacobians
    row = 0
    for i in range(N):
        # x_i - r_i >= 0
        jac[row, 2*i] = 1.0
        jac[row, 2*N + i] = -1.0
        row += 1
        # 1 - x_i - r_i >= 0
        jac[row, 2*i] = -1.0
        jac[row, 2*N + i] = -1.0
        row += 1
        # y_i - r_i >= 0
        jac[row, 2*i+1] = 1.0
        jac[row, 2*N + i] = -1.0
        row += 1
        # 1 - y_i - r_i >= 0
        jac[row, 2*i+1] = -1.0
        jac[row, 2*N + i] = -1.0
        row += 1
    
    # Pairwise distance constraint Jacobians
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
                nx, ny = 1.0, 0.0
            else:
                nx = dx / dist
                ny = dy / dist
            
            jac[row, 2*i] = nx
            jac[row, 2*i+1] = ny
            jac[row, 2*j] = -nx
            jac[row, 2*j+1] = -ny
            jac[row, 2*N + i] = -1.0
            jac[row, 2*N + j] = -1.0
            row += 1
    
    return jac


def _polish(pos, radii, deadline):
    """High-precision local search with multiple methods to squeeze out more sum of radii."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Multiple L-BFGS-B passes with decreasing tolerances
    tolerances = [(1e-12, 1e-8), (1e-14, 1e-10), (1e-16, 1e-12), (1e-18, 1e-14)]
    for ftol, gtol in tolerances:
        if time.time() > deadline - 3:
            break
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 5000, 'ftol': ftol, 'gtol': gtol})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Trust-constr NLP with explicit constraints and gradients
    if time.time() < deadline - 4:
        try:
            from scipy.optimize import LinearConstraint
            from scipy.optimize import NonlinearConstraint
            
            # Build constraint bounds: all constraints >= 0
            n_con = 4*N + N*(N-1)//2
            cons = NonlinearConstraint(
                fun=_constraints_func,
                lb=np.zeros(n_con),
                ub=np.inf,
                jac=_jacobian_constraints
            )
            
            result = minimize(
                fun=_gradient_objective,
                x0=x0,
                method='trust-constr',
                bounds=bounds,
                constraints=[cons],
                jac=np.zeros((2*N+N,)),  # gradient of objective is [0,...,0,-1,...,-1]
                options={'maxiter': 5000, 'gtol': 1e-10, 'xtol': 1e-12}
            )
            
            if result.success or (result.fun < _gradient_objective(x0) - 1e-15):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # SLSQP pass for tighter constraint handling
    if time.time() < deadline - 3:
        try:
            # Define constraints for SLSQP
            def _constraint_func(x, i, j):
                """Distance constraint between circles i and j."""
                pos_c = x[:2*N].reshape(N, 2)
                r_c = x[2*N:]
                dx = pos_c[i, 0] - pos_c[j, 0]
                dy = pos_c[i, 1] - pos_c[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - r_c[i] - r_c[j]
            
            def _boundary_constraint(x, i, axis, side):
                """Boundary constraint for circle i."""
                pos_c = x[:2*N].reshape(N, 2)
                r_c = x[2*N:]
                if axis == 0:
                    return (pos_c[i, 0] - r_c[i]) if side == 0 else (1.0 - pos_c[i, 0] - r_c[i])
                else:
                    return (pos_c[i, 1] - r_c[i]) if side == 0 else (1.0 - pos_c[i, 1] - r_c[i])
            
            # Build constraint list (limit to avoid too many constraints)
            constraints = []
            # Only add constraints for pairs that are close
            pos_c = x0[:2*N].reshape(N, 2)
            r_c = x0[2*N:]
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos_c[i, 0] - pos_c[j, 0]
                    dy = pos_c[i, 1] - pos_c[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r_c[i] + r_c[j] + 0.05:
                        constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: _constraint_func(x, i, j)})
            
            # Add boundary constraints
            for i in range(N):
                for axis in [0, 1]:
                    for side in [0, 1]:
                        constraints.append({'type': 'ineq', 'fun': lambda x, i=i, axis=axis, side=side: _boundary_constraint(x, i, axis, side)})
            
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': 2000, 'ftol': 1e-14})
            
            if result.success or (result.fun < _objective(x0, 5000.0) - 1e-15):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Final Powell pass for fine-tuning
    if time.time() < deadline - 2:
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='Powell', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-15})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration with time-aware allocation."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # 1. Grow and Push (Strong local search for packing)
    if time.time() < deadline - 25:
        gp_pos, gp_radii, gp_score = _grow_and_push(best_pos, best_radii, deadline, n_iter=15000)
        if gp_score > best_score + 1e-12:
            best_score = gp_score
            best_pos = gp_pos.copy()
            best_radii = gp_radii.copy()
    
    # 2. Gradient Polish (Fine-tune with SLSQP)
    if time.time() < deadline - 8:
        pol_pos, pol_radii, pol_score = _polish(best_pos, best_radii, deadline)
        if pol_score > best_score + 1e-12:
            best_score = pol_score
            best_pos = pol_pos.copy()
            best_radii = pol_radii.copy()
    
    # 3. Second Grow and Push pass (Exploit new space from polishing)
    if time.time() < deadline - 8:
        gp_pos, gp_radii, gp_score = _grow_and_push(best_pos, best_radii, deadline, n_iter=8000)
        if gp_score > best_score + 1e-12:
            best_score = gp_score
            best_pos = gp_pos.copy()
            best_radii = gp_radii.copy()
    
    # 4. Final Polish (high precision)
    if time.time() < deadline - 3:
        pol_pos, pol_radii, pol_score = _polish(best_pos, best_radii, deadline)
        if pol_score > best_score + 1e-12:
            best_score = pol_score
            best_pos = pol_pos.copy()
            best_radii = pol_radii.copy()
            
    return best_pos, best_radii, best_score


def _differential_evolution_run(pos, radii, deadline, n_iter=200):
    """
    Differential evolution global search to explore different topological configurations.
    """
    from scipy.optimize import differential_evolution
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.05, 0.95))
        bounds.append((0.05, 0.95))
    for i in range(N):
        bounds.append((0.02, 0.18))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    try:
        result = differential_evolution(
            lambda x: _objective(x, 2000.0),
            bounds=bounds,
            maxiter=n_iter,
            tol=1e-10,
            seed=42,
            polish=True,
            init='sobol',
            updating='deferred'
        )
        
        if result.fun < _objective(x0, 2000.0) - 1e-10:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_center_ring()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_corner_heavy()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed hexagonal
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed corner-heavy
    pos, radii = _initial_corner_heavy()
    pos_pert = pos + rng.normal(0, 0.015, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Run differential evolution first as a global search
    if time.time() < deadline - 45:
        pos, radii = init_configs[0]  # Use grid as starting point for DE
        de_pos, de_radii, de_score = _differential_evolution_run(pos, radii, deadline, n_iter=180)
        
        if de_score > best_score + 1e-12:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
    
    # Run optimization from each initial config with time-aware allocation
    # Prioritize hexagonal, grid, and corner-heavy as they are denser
    config_order = [1, 0, 3, 5, 6, 7, 2, 4, 5]
    
    for idx in config_order:
        if time.time() > deadline - 15:
            break
        if idx >= len(init_configs):
            continue
        
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Fallback if nothing valid found
    if best_pos is None or best_score < 0:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx2 = i * 5 + j
                if idx2 < N:
                    pos[idx2, 0] = (i + 0.5) / 5.0
                    pos[idx2, 1] = (j + 0.5) / 5.0
                    radii[idx2] = 0.09
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 3:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
    
    # Final post-processing to ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
