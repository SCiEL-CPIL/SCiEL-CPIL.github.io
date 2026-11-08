"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True


def _compute_loss_and_grads(pos, radii, lam_ov=100.0, lam_bd=100.0):
    """
    Compute differentiable loss and gradients for circle packing.
    
    Loss = -sum(radii) + lam_ov * sum(overlap_violations^2) + lam_bd * sum(boundary_violations^2)
    
    We minimize this loss, which effectively maximizes sum(radii) while penalizing violations.
    """
    n = len(pos)
    grad_pos = np.zeros_like(pos)
    grad_rad = np.zeros_like(radii)
    loss = -np.sum(radii)  # We want to maximize sum(radii), so negate it
    
    # Overlap penalties: for each pair, if dist < r_i + r_j, add (r_i + r_j - dist)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                violation = min_dist - dist
                loss += lam_ov * violation * violation
                
                # Gradient w.r.t. positions
                if dist > 1e-12:
                    nx = dx / dist
                    ny = dy / dist
                    # d(violation)/d(pos_i) = -nx, -ny (pushing i away from j)
                    # d(violation)/d(pos_j) = nx, ny (pushing j away from i)
                    # d(loss)/d(pos_i) = 2*lam_ov*violation * (-nx, -ny)
                    # d(loss)/d(pos_j) = 2*lam_ov*violation * (nx, ny)
                    grad_pos[i, 0] -= 2 * lam_ov * violation * nx
                    grad_pos[i, 1] -= 2 * lam_ov * violation * ny
                    grad_pos[j, 0] += 2 * lam_ov * violation * nx
                    grad_pos[j, 1] += 2 * lam_ov * violation * ny
                
                # Gradient w.r.t. radii
                # d(violation)/d(r_i) = 1, d(violation)/d(r_j) = 1
                # d(loss)/d(r_i) = 2*lam_ov*violation
                grad_rad[i] += 2 * lam_ov * violation
                grad_rad[j] += 2 * lam_ov * violation
    
    # Boundary penalties: for each circle, if pos < r or pos > 1-r, add violation^2
    for i in range(n):
        # x-boundary
        if pos[i, 0] < radii[i]:
            violation = radii[i] - pos[i, 0]
            loss += lam_bd * violation * violation
            # d(violation)/d(pos_x) = -1, d(violation)/d(r) = 1
            grad_pos[i, 0] -= 2 * lam_bd * violation
            grad_rad[i] += 2 * lam_bd * violation
        elif pos[i, 0] > 1 - radii[i]:
            violation = pos[i, 0] - (1 - radii[i])
            loss += lam_bd * violation * violation
            # d(violation)/d(pos_x) = 1, d(violation)/d(r) = -1
            grad_pos[i, 0] += 2 * lam_bd * violation
            grad_rad[i] -= 2 * lam_bd * violation
        
        # y-boundary
        if pos[i, 1] < radii[i]:
            violation = radii[i] - pos[i, 1]
            loss += lam_bd * violation * violation
            grad_pos[i, 1] -= 2 * lam_bd * violation
            grad_rad[i] += 2 * lam_bd * violation
        elif pos[i, 1] > 1 - radii[i]:
            violation = pos[i, 1] - (1 - radii[i])
            loss += lam_bd * violation * violation
            grad_pos[i, 1] += 2 * lam_bd * violation
            grad_rad[i] -= 2 * lam_bd * violation
    
    # Also add gradient for maximizing radii (since we negated sum(radii) in loss)
    grad_rad -= 1.0
    
    return loss, grad_pos, grad_rad


def _gradient_optimize(pos, radii, time_budget=30.0, start_time=None, 
                       lr_pos=0.005, lr_rad=0.002, lam_ov=100.0, lam_bd=100.0,
                       momentum=0.9, max_iters=5000):
    """
    Gradient descent optimization with adaptive learning rates and momentum.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    # Momentum buffers
    vel_pos = np.zeros_like(pos)
    vel_rad = np.zeros_like(radii)
    
    best_loss = np.inf
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Adaptive learning rate parameters
    lr_pos_curr = lr_pos
    lr_rad_curr = lr_rad
    patience = 0
    max_patience = 200
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        loss, grad_pos, grad_rad = _compute_loss_and_grads(pos, radii, lam_ov, lam_bd)
        
        # Track best
        if loss < best_loss:
            best_loss = loss
            best_pos = pos.copy()
            best_radii = radii.copy()
            best_score = np.sum(radii)
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            # Reduce learning rates and reset patience
            lr_pos_curr *= 0.5
            lr_rad_curr *= 0.5
            patience = 0
            if lr_pos_curr < 1e-7 or lr_rad_curr < 1e-7:
                break
        
        # Update with momentum
        vel_pos = momentum * vel_pos - lr_pos_curr * grad_pos
        vel_rad = momentum * vel_rad - lr_rad_curr * grad_rad
        
        pos = pos + vel_pos
        radii = radii + vel_rad
        
        # Clamp radii to reasonable range
        radii = np.clip(radii, 0.001, 0.49)
        
        # Clamp positions to [0, 1]
        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
        
        # Ensure positions are within [r, 1-r] where possible
        for i in range(n):
            r = radii[i]
            pos[i, 0] = np.clip(pos[i, 0], r, 1 - r) if r < 0.5 else pos[i, 0]
            pos[i, 1] = np.clip(pos[i, 1], r, 1 - r) if r < 0.5 else pos[i, 1]
    
    # Check if final state is valid
    if _check_validity(pos, radii):
        score = np.sum(radii)
        if score > best_score:
            best_pos = pos.copy()
            best_radii = radii.copy()
            best_score = score
    
    return best_pos, best_radii


def _build_hex(n):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = 6
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.17
            y = (row + 0.5) * 0.149
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    r = 0.08
    radii = np.full(n, r)
    return pos, radii


def _build_grid(n):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    r = 0.42 / side
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=0):
    """Build a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.07)
    return pos, radii


def _build_corner(n):
    """Place circles at corners and edges for a structured start."""
    pos_list = []
    r_corner = 0.12
    # Corners
    pos_list.append([r_corner, r_corner])
    pos_list.append([1 - r_corner, r_corner])
    pos_list.append([r_corner, 1 - r_corner])
    pos_list.append([1 - r_corner, 1 - r_corner])
    
    # Edge midpoints
    r_edge = 0.1
    pos_list.append([0.5, r_edge])
    pos_list.append([0.5, 1 - r_edge])
    pos_list.append([r_edge, 0.5])
    pos_list.append([1 - r_edge, 0.5])
    
    # Fill remaining in a grid pattern in the center
    remaining = n - len(pos_list)
    if remaining > 0:
        side = int(np.ceil(np.sqrt(remaining)))
        for j in range(side):
            for i in range(side):
                if len(pos_list) >= n:
                    break
                x = 0.25 + (i + 0.5) * 0.5 / side
                y = 0.25 + (j + 0.5) * 0.5 / side
                pos_list.append([x, y])
    
    pos = np.array(pos_list[:n])
    radii = np.full(n, 0.08)
    return pos, radii


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # Multiple initial configurations
    configs = []
    
    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy(), "hex"))
    
    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy(), "grid"))
    
    pos, radii = _build_corner(N)
    configs.append((pos.copy(), radii.copy(), "corner"))
    
    for seed in [42, 123, 456, 789]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))
    
    # Process each configuration with gradient optimization
    for ci, (pos, radii, name) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        remaining = deadline - elapsed - 2
        
        # Phase 1: Gradient optimization
        time_alloc = min(25.0, remaining * 0.7)
        if time_alloc > 2:
            pos, radii = _gradient_optimize(pos, radii, time_budget=time_alloc,
                                             start_time=start_time,
                                             lr_pos=0.005, lr_rad=0.002,
                                             lam_ov=100.0, lam_bd=100.0,
                                             momentum=0.9, max_iters=5000)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    # Final polish with gradient optimization if time remains
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 5:
            # Try different penalty weights for final polish
            for lam_ov, lam_bd in [(50.0, 50.0), (200.0, 200.0), (150.0, 100.0)]:
                elapsed = time.time() - start_time
                remaining = deadline - elapsed - 1
                if remaining < 3:
                    break
                
                pos, radii = _gradient_optimize(best_pos, best_radii, 
                                                 time_budget=min(15.0, remaining),
                                                 start_time=start_time,
                                                 lr_pos=0.003, lr_rad=0.001,
                                                 lam_ov=lam_ov, lam_bd=lam_bd,
                                                 momentum=0.85, max_iters=3000)
                
                if _check_validity(pos, radii):
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
    
    # Return best result
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
