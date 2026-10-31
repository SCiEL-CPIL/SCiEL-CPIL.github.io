"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def _is_valid(circles, eps=1e-9):
    """Check that circles are inside [0,1]^2 and disjoint."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    n = len(circles)
    # Boundary check
    if np.any(x < -eps) or np.any(y < -eps) or np.any(x + r > 1 + eps) or np.any(y + r > 1 + eps):
        return False
    if np.any(r < -eps):
        return False
    # Pairwise distance check
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - eps:
                return False
    return True


def _score(circles):
    return np.sum(circles[:, 2])


def _initial_grid(n):
    """Create a reasonable initial grid configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.48 / side])
    return np.array(cells)


def _relax(circles, max_iter=5000, lr=0.01, time_budget=100.0):
    """
    Gradient-based relaxation to maximize sum of radii.
    Uses a penalty method for constraints.
    """
    start_time = time.time()
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    # Precompute for speed
    best_score = _score(circles)
    best_pos = pos.copy()
    best_rad = rad.copy()
    
    lr_pos = 0.005
    lr_rad = 0.002
    penalty_weight = 100.0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute gradients
        dpos = np.zeros_like(pos)
        drad = np.zeros_like(rad)
        
        # Objective: maximize sum(r) -> gradient of -sum(r) for minimization
        # We're doing ascent on sum(r), so gradient direction is +1 for each r
        # But we need to handle constraints via penalty
        
        # Penalty for boundary violations
        for i in range(n):
            # Left boundary
            if pos[i, 0] - rad[i] < 0:
                viol = -(pos[i, 0] - rad[i])
                dpos[i, 0] += penalty_weight * viol
                drad[i] -= penalty_weight * viol
            # Right boundary
            if pos[i, 0] + rad[i] > 1:
                viol = pos[i, 0] + rad[i] - 1
                dpos[i, 0] -= penalty_weight * viol
                drad[i] -= penalty_weight * viol
            # Bottom boundary
            if pos[i, 1] - rad[i] < 0:
                viol = -(pos[i, 1] - rad[i])
                dpos[i, 1] += penalty_weight * viol
                drad[i] -= penalty_weight * viol
            # Top boundary
            if pos[i, 1] + rad[i] > 1:
                viol = pos[i, 1] + rad[i] - 1
                dpos[i, 1] -= penalty_weight * viol
                drad[i] -= penalty_weight * viol
        
        # Penalty for overlap violations
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    viol = min_dist - dist
                    if dist > 1e-12:
                        # Gradient with respect to positions
                        dpos[i, 0] += penalty_weight * viol * dx / dist
                        dpos[i, 1] += penalty_weight * viol * dy / dist
                        dpos[j, 0] -= penalty_weight * viol * dx / dist
                        dpos[j, 1] -= penalty_weight * viol * dy / dist
                    # Gradient with respect to radii
                    drad[i] -= penalty_weight * viol
                    drad[j] -= penalty_weight * viol
        
        # Gradient of objective (we want to maximize sum of radii)
        # So the ascent direction for radii is +1
        obj_grad = np.ones_like(rad)
        
        # Combined gradient for ascent
        total_dr = obj_grad - drad  # drad is negative penalty gradient
        total_dp = -dpos  # dpos is negative penalty gradient
        
        # Update
        pos += lr_pos * total_dp
        rad += lr_rad * total_dr
        
        # Clip radii to be non-negative
        rad = np.maximum(rad, 1e-6)
        
        # Check if valid and update best
        current_circles = np.column_stack([pos, rad])
        if _is_valid(current_circles, eps=1e-7):
            current_score = _score(current_circles)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_rad = rad.copy()
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr_pos *= 0.98
            lr_rad *= 0.98
            penalty_weight *= 1.05
    
    return np.column_stack([best_pos, best_rad])


def _polish(circles, max_iter=2000, time_budget=20.0):
    """High-precision polish using smaller steps."""
    start_time = time.time()
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    best_score = _score(circles)
    best_pos = pos.copy()
    best_rad = rad.copy()
    
    lr_pos = 0.001
    lr_rad = 0.0005
    penalty_weight = 500.0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        dpos = np.zeros_like(pos)
        drad = np.zeros_like(rad)
        
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                viol = -(pos[i, 0] - rad[i])
                dpos[i, 0] += penalty_weight * viol
                drad[i] -= penalty_weight * viol
            if pos[i, 0] + rad[i] > 1:
                viol = pos[i, 0] + rad[i] - 1
                dpos[i, 0] -= penalty_weight * viol
                drad[i] -= penalty_weight * viol
            if pos[i, 1] - rad[i] < 0:
                viol = -(pos[i, 1] - rad[i])
                dpos[i, 1] += penalty_weight * viol
                drad[i] -= penalty_weight * viol
            if pos[i, 1] + rad[i] > 1:
                viol = pos[i, 1] + rad[i] - 1
                dpos[i, 1] -= penalty_weight * viol
                drad[i] -= penalty_weight * viol
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    viol = min_dist - dist
                    if dist > 1e-12:
                        dpos[i, 0] += penalty_weight * viol * dx / dist
                        dpos[i, 1] += penalty_weight * viol * dy / dist
                        dpos[j, 0] -= penalty_weight * viol * dx / dist
                        dpos[j, 1] -= penalty_weight * viol * dy / dist
                    drad[i] -= penalty_weight * viol
                    drad[j] -= penalty_weight * viol
        
        obj_grad = np.ones_like(rad)
        total_dr = obj_grad - drad
        total_dp = -dpos
        
        pos += lr_pos * total_dp
        rad += lr_rad * total_dr
        rad = np.maximum(rad, 1e-6)
        
        current_circles = np.column_stack([pos, rad])
        if _is_valid(current_circles, eps=1e-8):
            current_score = _score(current_circles)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_rad = rad.copy()
        
        if it % 50 == 0 and it > 0:
            lr_pos *= 0.97
            lr_rad *= 0.97
            penalty_weight *= 1.03
    
    return np.column_stack([best_pos, best_rad])


def construct():
    """Construct 26 disjoint circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Start with grid initialization
    circles = _initial_grid(N)
    
    # Main relaxation phase
    remaining = deadline - (time.time() - start_time)
    if remaining > 30:
        circles = _relax(circles, max_iter=8000, lr=0.01, time_budget=min(remaining - 20, 100))
    
    # Polish phase
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        circles = _polish(circles, max_iter=3000, time_budget=min(remaining - 5, 20))
    
    # Final validation and correction
    if not _is_valid(circles, eps=1e-7):
        # Slightly shrink radii to ensure validity
        circles = circles.copy()
        circles[:, 2] *= 0.999
        if not _is_valid(circles, eps=1e-7):
            circles[:, 2] *= 0.999
            if not _is_valid(circles, eps=1e-7):
                # Fallback to grid
                circles = _initial_grid(N)
    
    return circles
# EVOLVE-BLOCK-END
