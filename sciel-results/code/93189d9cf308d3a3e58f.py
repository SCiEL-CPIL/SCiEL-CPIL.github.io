"""Circle packing in unit square - 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

def _validate(circles):
    """Check if circles are valid (inside square, disjoint)."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r < 0 or x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True

def _score(circles):
    """Return sum of radii if valid, else 0."""
    if _validate(circles):
        return float(np.sum(circles[:, 2]))
    return 0.0

def _initial_grid():
    """Create a reasonable initial grid configuration."""
    # Use a 5x5 grid, pick 26 best positions
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    
    # Sort by distance from center, prefer central positions
    cx, cy = 0.5, 0.5
    positions.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    
    # Take first N positions
    selected = positions[:N]
    
    # Initial radius: estimate based on local density
    # For a 5x5 grid, spacing is 0.2, so max radius ~0.1
    # But we'll optimize
    r0 = 0.09
    circles = np.array([[x, y, r0] for x, y in selected])
    return circles

def _penalty_and_grad(circles, beta=50.0):
    """
    Compute smooth penalty for overlaps and boundary violations.
    Returns (penalty_value, gradient).
    
    We want to maximize sum(r_i) - penalty
    penalty = sum over pairs of max(0, r_i + r_j - dist_ij)^2
            + sum over i of max(0, r_i - x_i)^2 + max(0, r_i - (1-x_i))^2 + ...
    """
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    penalty = 0.0
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # Pairwise overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-12
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                # Smooth approximation: use softplus-like
                # penalty contribution: 0.5 * overlap^2
                penalty += 0.5 * overlap * overlap
                # Gradient
                # d/dx_i of 0.5*(r_i+r_j-dist)^2 = (r_i+r_j-dist) * (-1/dist * dx)
                # Actually: d(dist)/dx_i = dx/dist, so d(-dist)/dx_i = -dx/dist
                # d(overlap)/dx_i = -dx/dist
                coeff = overlap
                gx[i] += coeff * (dx / dist)
                gx[j] += -coeff * (dx / dist)
                gy[i] += coeff * (dy / dist)
                gy[j] += -coeff * (dy / dist)
                gr[i] += coeff
                gr[j] += coeff
    
    # Boundary penalties
    for i in range(n):
        # Left: x_i - r_i >= 0
        viol = r[i] - x[i]
        if viol > 0:
            penalty += 0.5 * viol * viol
            gx[i] += viol
            gr[i] += -viol
        
        # Right: 1 - x_i - r_i >= 0
        viol = r[i] - (1.0 - x[i])
        if viol > 0:
            penalty += 0.5 * viol * viol
            gx[i] += -viol
            gr[i] += -viol
        
        # Bottom: y_i - r_i >= 0
        viol = r[i] - y[i]
        if viol > 0:
            penalty += 0.5 * viol * viol
            gy[i] += viol
            gr[i] += -viol
        
        # Top: 1 - y_i - r_i >= 0
        viol = r[i] - (1.0 - y[i])
        if viol > 0:
            penalty += 0.5 * viol * viol
            gy[i] += -viol
            gr[i] += -viol
    
    # Objective: maximize sum(r) - beta * penalty
    # Gradient of objective:
    # d/dr_i = 1 - beta * gr[i]
    # d/dx_i = -beta * gx[i]
    # d/dy_i = -beta * gy[i]
    
    grad_obj = np.zeros_like(circles)
    grad_obj[:, 0] = -beta * gx
    grad_obj[:, 1] = -beta * gy
    grad_obj[:, 2] = 1.0 - beta * gr
    
    return penalty, grad_obj

def _optimize(circles, max_iter=500, lr=0.01, deadline=None):
    """Gradient ascent with adaptive learning rate."""
    circles = circles.copy()
    best_circles = circles.copy()
    best_score = _score(circles)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        penalty, grad = _penalty_and_grad(circles, beta=100.0)
        
        # Adaptive learning rate: decrease if penalty is high
        if penalty > 0.01:
            step_lr = lr * 0.1
        elif penalty > 0.001:
            step_lr = lr * 0.5
        else:
            step_lr = lr
        
        # Gradient step
        circles = circles + step_lr * grad
        
        # Clip radii to be positive
        circles[:, 2] = np.maximum(circles[:, 2], 1e-4)
        
        # Project positions to [0,1]
        circles[:, 0] = np.clip(circles[:, 0], 0.0, 1.0)
        circles[:, 1] = np.clip(circles[:, 1], 0.0, 1.0)
        
        # Check score
        curr_score = _score(circles)
        if curr_score > best_score:
            best_score = curr_score
            best_circles = circles.copy()
        
        # Early stopping if no improvement
        if it > 50 and curr_score < best_score * 0.95:
            break
    
    return best_circles, best_score

def _polish(circles, deadline=None):
    """Final high-precision polish with smaller steps."""
    circles = circles.copy()
    best_circles = circles.copy()
    best_score = _score(circles)
    
    for it in range(200):
        if deadline is not None and time.time() > deadline:
            break
        
        penalty, grad = _penalty_and_grad(circles, beta=500.0)
        
        # Very small steps
        step_lr = 0.001
        
        circles = circles + step_lr * grad
        circles[:, 2] = np.maximum(circles[:, 2], 1e-4)
        circles[:, 0] = np.clip(circles[:, 0], 0.0, 1.0)
        circles[:, 1] = np.clip(circles[:, 1], 0.0, 1.0)
        
        curr_score = _score(circles)
        if curr_score > best_score:
            best_score = curr_score
            best_circles = circles.copy()
    
    return best_circles, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Leave safety margin before 150s kill
    
    best_overall = None
    best_overall_score = 0.0
    
    # Try multiple initializations
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initial configurations
        if restart == 0:
            circles = _initial_grid()
        elif restart == 1:
            # Slightly perturbed grid
            circles = _initial_grid()
            circles[:, 0] += np.random.normal(0, 0.01, N)
            circles[:, 1] += np.random.normal(0, 0.01, N)
            circles[:, 0] = np.clip(circles[:, 0], 0.01, 0.99)
            circles[:, 1] = np.clip(circles[:, 1], 0.01, 0.99)
        elif restart == 2:
            # Random positions
            circles = np.random.uniform(0.1, 0.9, (N, 2))
            circles = np.column_stack([circles, np.full(N, 0.08)])
        elif restart == 3:
            # Hexagonal-ish packing
            circles = np.zeros((N, 3))
            for i in range(N):
                row = i // 5
                col = i % 5
                x = 0.1 + col * 0.2 + (0.05 if row % 2 == 1 else 0)
                y = 0.1 + row * 0.17
                circles[i] = [x, y, 0.08]
        else:
            # Random with varying radii
            circles = np.random.uniform(0.05, 0.95, (N, 2))
            radii = np.random.uniform(0.06, 0.12, N)
            circles = np.column_stack([circles, radii])
        
        # Optimize
        optimized, score = _optimize(circles, max_iter=300, lr=0.02, deadline=deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall = optimized.copy()
        
        # Polish if we have time
        if time.time() < deadline - 10:
            polished, polish_score = _polish(optimized, deadline=deadline)
            if polish_score > best_overall_score:
                best_overall_score = polish_score
                best_overall = polished.copy()
    
    if best_overall is None:
        # Fallback to grid
        best_overall = _initial_grid()
    
    return best_overall
