"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 (2*(W+H)=4 => W+H=2).
    Maximize sum of radii.
    
    Strategy:
    1. Initialize with a grid layout (5x5 grid, take 21 best).
    2. Use gradient ascent on positions and radii to maximize sum(r) subject to:
       - Circles stay inside rectangle
       - Circles don't overlap
       - Rectangle perimeter <= 4
    
    We'll use a penalty method: minimize -sum(r) + penalty * violations.
    
    The rectangle is defined by W, H with W+H=2. We can parameterize W in [0.5, 1.5].
    Actually, W+H=2, so H=2-W. W must be in (0, 2).
    
    Let's use W as a variable, with H=2-W.
    
    For each circle i at (x_i, y_i) with radius r_i:
    - x_i - r_i >= 0
    - x_i + r_i <= W
    - y_i - r_i >= 0
    - y_i + r_i <= H
    - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    
    Objective: maximize sum(r_i)
    
    We'll use a simple gradient ascent with penalty terms.
    """
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    
    # Initial layout: 5x5 grid
    side = 5
    # Start with a square-ish rectangle
    W = 1.0
    H = 1.0
    
    # Grid positions
    xs = []
    ys = []
    rs = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) * W / side
            y = (j + 0.5) * H / side
            r = 0.5 * min(W, H) / side - 0.005
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    rs = np.array(rs, dtype=np.float64)
    
    # Variables: xs, ys, rs, W (H = 2 - W)
    # We'll optimize W in [0.3, 1.7]
    W_val = 1.0
    H_val = 1.0
    
    best_sum_r = np.sum(rs)
    best_xs, best_ys, best_rs, best_W = xs.copy(), ys.copy(), rs.copy(), W_val
    
    # Penalty method
    penalty = 100.0
    lr = 0.01
    
    # Gradient ascent with penalty
    n_iter = 0
    while time.time() < deadline - 5.0:
        H_val = 2.0 - W_val
        
        # Compute violations
        # Boundary violations
        viol_x_left = np.maximum(0, 0 - (xs - rs))  # xs - rs < 0
        viol_x_right = np.maximum(0, (xs + rs) - W_val)
        viol_y_bottom = np.maximum(0, 0 - (ys - rs))
        viol_y_top = np.maximum(0, (ys + rs) - H_val)
        
        # Overlap violations
        overlap_viol = 0.0
        overlap_grad_x = np.zeros_like(xs)
        overlap_grad_y = np.zeros_like(ys)
        overlap_grad_r = np.zeros_like(rs)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    v = min_dist - dist
                    overlap_viol += v * v
                    if dist > 1e-10:
                        fx = dx / dist
                        fy = dy / dist
                    else:
                        fx, fy = 1.0, 0.0
                    overlap_grad_x[i] += 2 * v * fx
                    overlap_grad_x[j] -= 2 * v * fx
                    overlap_grad_y[i] += 2 * v * fy
                    overlap_grad_y[j] -= 2 * v * fy
                    overlap_grad_r[i] += 2 * v
                    overlap_grad_r[j] += 2 * v
        
        # Total penalty
        penalty_val = penalty * (np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + 
                                 np.sum(viol_y_bottom**2) + np.sum(viol_y_top**2) + 
                                 overlap_viol)
        
        obj = np.sum(rs) - penalty_val
        
        # Check if valid
        if (np.all(xs - rs >= -1e-6) and np.all(xs + rs <= W_val + 1e-6) and
            np.all(ys - rs >= -1e-6) and np.all(ys + rs <= H_val + 1e-6)):
            # Check overlaps
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    if dx*dx + dy*dy < (rs[i] + rs[j])**2 * (1 - 1e-6):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                if np.sum(rs) > best_sum_r:
                    best_sum_r = np.sum(rs)
                    best_xs, best_ys, best_rs, best_W = xs.copy(), ys.copy(), rs.copy(), W_val
        
        # Compute gradients
        # d/dxs of sum(rs) = 0, d/drs = 1
        grad_x = -penalty * (2 * np.maximum(0, 0 - (xs - rs)) * (-1) + 
                             2 * np.maximum(0, (xs + rs) - W_val) * 1 + 
                             overlap_grad_x)
        # Wait, let me be more careful.
        # Penalty for x_left: max(0, -(xs - rs))^2 = max(0, rs - xs)^2
        # d/dxs = 2 * max(0, rs - xs) * (-1) = -2 * max(0, rs - xs)
        # Penalty for x_right: max(0, xs + rs - W)^2
        # d/dxs = 2 * max(0, xs + rs - W) * 1 = 2 * max(0, xs + rs - W)
        # Total penalty for x: penalty * [sum(max(0, rs-xs)^2) + sum(max(0, xs+rs-W)^2) + ...]
        # d/dxs of penalty = penalty * [-2*max(0, rs-xs) + 2*max(0, xs+rs-W) + overlap_grad_x]
        # We want to maximize sum(rs) - penalty, so gradient of objective w.r.t. xs:
        # = 0 - penalty * [-2*max(0, rs-xs) + 2*max(0, xs+rs-W) + overlap_grad_x]
        # = penalty * [2*max(0, rs-xs) - 2*max(0, xs+rs-W) - overlap_grad_x]
        
        grad_x = penalty * (2 * np.maximum(0, rs - xs) - 2 * np.maximum(0, xs + rs - W_val) - overlap_grad_x)
        
        # Similarly for ys
        grad_y = penalty * (2 * np.maximum(0, rs - ys) - 2 * np.maximum(0, ys + rs - H_val) - overlap_grad_y)
        
        # For rs: d/drs of sum(rs) = 1
        # d/drs of penalty: 
        # x_left: 2*max(0, rs-xs)*1
        # x_right: 2*max(0, xs+rs-W)*1
        # y_bottom: 2*max(0, rs-ys)*1
        # y_top: 2*max(0, ys+rs-H)*1
        # overlap: overlap_grad_r
        grad_r = 1 - penalty * (2 * np.maximum(0, rs - xs) + 2 * np.maximum(0, xs + rs - W_val) +
                                 2 * np.maximum(0, rs - ys) + 2 * np.maximum(0, ys + rs - H_val) +
                                 overlap_grad_r)
        
        # For W: H = 2 - W
        # x_right: max(0, xs+rs-W)^2, d/dW = 2*max(0, xs+rs-W)*(-1) = -2*max(0, xs+rs-W)
        # y_top: max(0, ys+rs-(2-W))^2 = max(0, ys+rs-2+W)^2, d/dW = 2*max(0, ys+rs-2+W)*1
        grad_W = -penalty * (-2 * np.sum(np.maximum(0, xs + rs - W_val)) + 
                              2 * np.sum(np.maximum(0, ys + rs - (2.0 - W_val))))
        
        # Update
        xs += lr * grad_x
        ys += lr * grad_y
        rs += lr * grad_r
        W_val += lr * grad_W
        
        # Clamp
        W_val = np.clip(W_val, 0.3, 1.7)
        rs = np.clip(rs, 0.01, 1.0)
        
        n_iter += 1
        
        # Adaptive learning rate
        if n_iter % 50 == 0:
            lr *= 0.95
            penalty *= 1.1
    
    # Final result
    W_final = best_W
    H_final = 2.0 - W_final
    
    # Scale to ensure perimeter exactly 4
    # The perimeter is 2*(W+H) = 2*2 = 4, so it's already 4.
    
    # Construct result
    result = np.column_stack([best_xs, best_ys, best_rs])
    
    return result
# EVOLVE-BLOCK-END
