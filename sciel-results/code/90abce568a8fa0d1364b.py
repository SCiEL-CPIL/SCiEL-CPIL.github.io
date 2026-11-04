"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import sys

N = 21


def _soft_max_penalty(dists, radii_sum, eps):
    """Smooth penalty for overlaps and boundary violations.
    dists: array of (r_i + r_j - d_ij) for all pairs, plus boundary violations
    radii_sum: sum of all radii
    """
    # For overlaps: we want dists[i] < 0 (i.e., r_i + r_j < d_ij)
    # Penalty: soft max of dists (positive values indicate violation)
    # log-sum-exp for soft max
    if len(dists) == 0:
        return 0.0
    m = np.max(dists)
    if m < -50.0:
        return 0.0
    logsum = m + np.log(np.sum(np.exp((dists - m) / eps))) * eps
    return max(0.0, logsum)


def _objective_penalty(x, W, H, eps, penalty_weight):
    """Penalty-based objective: minimize penalty (we want to maximize sum of radii).
    x layout: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_N-1, y_N-1, r_N-1]
    """
    centers = x[0:N*3:3]
    ys = x[1:N*3:3]
    radii = x[2:N*3:3]
    
    # Boundary violations: circle must be inside rectangle [0,W] x [0,H]
    # x_i >= r_i, x_i <= W - r_i, y_i >= r_i, y_i <= H - r_i
    boundary_violations = np.maximum(0, radii - centers) + \
                          np.maximum(0, radii - (W - centers)) + \
                          np.maximum(0, radii - ys) + \
                          np.maximum(0, radii - (H - ys))
    
    # Pairwise overlap violations
    overlap_violations = []
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i] - centers[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx*dx + dy*dy)
            overlap_violations.append(radii[i] + radii[j] - d)
    overlap_violations = np.array(overlap_violations)
    
    all_violations = np.concatenate([boundary_violations, overlap_violations])
    
    # We want to minimize total violation and maximize sum of radii
    # Objective: penalty_weight * soft_max_penalty - sum(radii)
    # But we can reformulate: minimize (penalty_weight * max_violation) subject to constraints
    # Better: use a Lagrangian-like approach
    
    # Actually, let's use a different formulation:
    # Minimize: -sum(radii) + penalty_weight * sum of squared violations
    # This is smoother
    
    penalty = np.sum(all_violations ** 2)
    return -np.sum(radii) + penalty_weight * penalty


def _objective_soft(x, W, H, eps):
    """Objective using soft-max penalty on violations.
    Minimize: -sum(radii) + penalty_weight * softmax(violations)
    """
    centers = x[0:N*3:3]
    ys = x[1:N*3:3]
    radii = x[2:N*3:3]
    
    # Boundary violations (positive = violation)
    bv1 = np.maximum(0.0, radii - centers)
    bv2 = np.maximum(0.0, radii - (W - centers))
    bv3 = np.maximum(0.0, radii - ys)
    bv4 = np.maximum(0.0, radii - (H - ys))
    
    # Pairwise overlap violations
    ov = np.array([radii[i] + radii[j] - np.sqrt((centers[i]-centers[j])**2 + (ys[i]-ys[j])**2) 
                   for i in range(N) for j in range(i+1, N)])
    
    all_v = np.concatenate([bv1, bv2, bv3, bv4, ov])
    
    # Soft max penalty
    m = np.max(all_v)
    if m < -100.0:
        soft_max_val = 0.0
    else:
        # Use log-sum-exp for numerical stability
        scaled = (all_v - m) / eps
        logsum = m + eps * np.log(np.sum(np.exp(scaled)))
        soft_max_val = max(0.0, logsum)
    
    penalty_weight = 100.0
    return -np.sum(radii) + penalty_weight * soft_max_val


def _grad_objective_soft(x, W, H, eps):
    """Numerical gradient for the soft objective."""
    centers = x[0:N*3:3]
    ys = x[1:N*3:3]
    radii = x[2:N*3:3]
    
    grad = np.zeros_like(x)
    
    # d/d(r_i) of -sum(radii) = -1
    for i in range(N):
        grad[2*N + i] = -1.0
    
    # Gradient of soft max penalty
    all_v = np.concatenate([
        np.maximum(0.0, radii - centers),
        np.maximum(0.0, radii - (W - centers)),
        np.maximum(0.0, radii - ys),
        np.maximum(0.0, radii - (H - ys)),
        np.array([radii[i] + radii[j] - np.sqrt((centers[i]-centers[j])**2 + (ys[i]-ys[j])**2) 
                  for i in range(N) for j in range(i+1, N)])
    ])
    
    m = np.max(all_v)
    if m < -100.0:
        return grad
    
    scaled = (all_v - m) / eps
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    
    # d(softmax)/d(v_k) = exp(v_k/m_scaled) / sum = (1/eps) * exp_scaled_k / sum_exp
    # Actually: softmax(v) = max(v) + eps * log(sum(exp((v_k - max)/eps)))
    # d/dv_k = (1/eps) * exp((v_k - max)/eps) / sum(exp((v_j - max)/eps))
    
    weights = (1.0 / eps) * exp_scaled / sum_exp
    penalty_weight = 100.0
    
    # Now propagate gradients through each violation
    idx = 0
    for i in range(N):
        # bv1[i] = max(0, r_i - x_i), but we used max so gradient is 0 if negative
        if all_v[idx] > 0:
            grad[2*N+i] += penalty_weight * weights[idx]  # d(r_i - x_i)/dr_i = 1
            grad[i] += -penalty_weight * weights[idx]     # d(r_i - x_i)/dx_i = -1
        idx += 1
    
    for i in range(N):
        if all_v[idx] > 0:
            grad[2*N+i] += penalty_weight * weights[idx]  # d(r_i - (W-x_i))/dr_i = 1
            grad[i] += penalty_weight * weights[idx]      # d(r_i - W + x_i)/dx_i = 1
        idx += 1
    
    for i in range(N):
        if all_v[idx] > 0:
            grad[2*N+i] += penalty_weight * weights[idx]  # d(r_i - y_i)/dr_i = 1
            grad[N+2*N+i] += -penalty_weight * weights[idx]  # d(r_i - y_i)/dy_i = -1
        idx += 1
    
    for i in range(N):
        if all_v[idx] > 0:
            grad[2*N+i] += penalty_weight * weights[idx]  # d(r_i - (H-y_i))/dr_i = 1
            grad[N+2*N+i] += penalty_weight * weights[idx]  # d(r_i - H + y_i)/dy_i = 1
        idx += 1
    
    # Pairwise overlaps
    for i in range(N):
        for j in range(i+1, N):
            if all_v[idx] > 0:
                dx = centers[i] - centers[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-10:
                    # v = r_i + r_j - d
                    # d/dr_i = 1, d/dr_j = 1
                    # d/dx_i = -dx/d, d/dx_j = dx/d
                    # d/dy_i = -dy/d, d/dy_j = dy/d
                    grad[2*N+i] += penalty_weight * weights[idx]
                    grad[2*N+j] += penalty_weight * weights[idx]
                    grad[i] += -penalty_weight * weights[idx] * dx / d
                    grad[N+2*N+i] += -penalty_weight * weights[idx] * dy / d
                    grad[j] += penalty_weight * weights[idx] * dx / d
                    grad[N+2*N+j] += penalty_weight * weights[idx] * dy / d
            idx += 1
    
    return grad


def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Rectangle with perimeter 4: 2*(W+H) = 4, so W+H = 2
    # Try different aspect ratios
    best_value = 0.0
    best_x = None
    best_W = 1.0
    best_H = 1.0
    
    # Initialize with a grid-like packing
    def init_packing(W, H):
        """Create initial circle positions."""
        # Use a hexagonal-like grid
        x = np.zeros(3*N)
        n_col = int(np.ceil(np.sqrt(N * W / H))) if H > 0 else N
        n_row = int(np.ceil(N / n_col)) if n_col > 0 else N
        n_col = max(1, n_col)
        n_row = max(1, n_row)
        
        for i in range(N):
            col = i % n_col
            row = i // n_col
            
            # Hexagonal offset for odd rows
            offset = 0.0
            if row % 2 == 1 and n_col > 1:
                offset = 0.5 / n_col * W
            
            x[3*i] = (col + 0.5) / n_col * W + offset
            x[3*i+1] = (row + 0.5) / n_row * H
            # Initial radius: estimate
            x[3*i+2] = min(W / n_col, H / n_row) / 2.5
        
        # Clamp to bounds
        for i in range(N):
            x[3*i] = np.clip(x[3*i], 0, W)
            x[3*i+1] = np.clip(x[3*i+1], 0, H)
            x[3*i+2] = np.clip(x[3*i+2], 1e-6, min(W, H) / 2)
        
        return x
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.667, 1.333, 0.5, 2.0]
    
    for ar in aspect_ratios:
        if time_left() < 5.0:
            break
        
        W = 2.0 * ar / (1 + ar)
        H = 2.0 / (1 + ar)
        
        x0 = init_packing(W, H)
        
        # Phase 1: Coarse optimization with L-BFGS-B
        bounds = []
        for i in range(N):
            bounds.append((1e-6, W - 1e-6))  # x
            bounds.append((1e-6, H - 1e-6))  # y
            bounds.append((1e-6, min(W, H) / 2))  # r
        
        try:
            res = minimize(
                _objective_soft,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x1 = res.x
            val1 = -np.sum(x1[2:N*3:3])  # sum of radii (negate objective's -sum)
            
            # Check validity
            valid = _is_valid(x1, W, H)
            if valid and val1 > best_value:
                best_value = val1
                best_x = x1.copy()
                best_W = W
                best_H = H
        
        except Exception:
            pass
        
        if time_left() < 3.0:
            break
        
        # Phase 2: Fine polish with smaller eps
        if best_x is not None and best_W == W:
            try:
                for eps in [0.01, 0.001, 0.0001, 1e-5]:
                    if time_left() < 2.0:
                        break
                    
                    # Redefine objective with this eps
                    def obj_fine(x, eps_val=eps):
                        return _objective_soft(x, W, H, eps_val)
                    
                    res2 = minimize(
                        obj_fine,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10}
                    )
                    x2 = res2.x
                    val2 = -np.sum(x2[2:N*3:3])
                    
                    valid2 = _is_valid(x2, W, H)
                    if valid2 and val2 > best_value:
                        best_value = val2
                        best_x = x2.copy()
                        best_W = W
                        best_H = H
            
            except Exception:
                pass
    
    if best_x is None:
        # Fallback: simple grid
        W = 1.0
        H = 1.0
        x = np.zeros(3*N)
        n_col = 5
        for i in range(N):
            col = i % n_col
            row = i // n_col
            x[3*i] = (col + 0.5) / n_col
            x[3*i+1] = (row + 0.5) / 5
            x[3*i+2] = 0.09
        return x
    
    return best_x


def _is_valid(x, W, H, tol=1e-9):
    """Check if the packing is valid (no overlaps, all within bounds)."""
    centers = x[0:N*3:3]
    ys = x[1:N*3:3]
    radii = x[2:N*3:3]
    
    # Check boundaries
    for i in range(N):
        if centers[i] < radii[i] - tol or centers[i] > W - radii[i] + tol:
            return False
        if ys[i] < radii[i] - tol or ys[i] > H - radii[i] + tol:
            return False
    
    # Check pairwise distances
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((centers[i]-centers[j])**2 + (ys[i]-ys[j])**2)
            if d < radii[i] + radii[j] - tol:
                return False
    
    return True
