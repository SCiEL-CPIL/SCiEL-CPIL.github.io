import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    best_sum = 0.0
    best_solution = None
    
    # Precompute pairwise indices for vectorized overlap computation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Gradient-based optimization with soft penalty
    # Objective: maximize sum(r) - lambda * penalty
    # Penalty: soft violations of overlap and boundary constraints
    
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    
    np.random.seed(42)
    
    for W in aspect_ratios:
        H = 2.0 - W
        
        # Skip if too degenerate
        if H < 0.2 or W < 0.2:
            continue
        
        # Multi-start with different initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Hexagonal grid initialization for better starting point
            if trial == 0:
                # Hexagonal packing
                cols = int(np.ceil(np.sqrt(N * W / (H * 0.866))))
                rows = int(np.ceil(N / cols))
                x = np.zeros(N)
                y = np.zeros(N)
                r = np.zeros(N)
                for k in range(N):
                    col = k % cols
                    row = k // cols
                    x[k] = (col + 0.5) * (W / cols)
                    y[k] = (row + 0.5) * (H / rows)
                    if row % 2 == 1:
                        x[k] += 0.5 * (W / cols)
                    x[k] = np.clip(x[k], 0.05, W - 0.05)
                    y[k] = np.clip(y[k], 0.05, H - 0.05)
                    r[k] = 0.04
            else:
                # Random initialization
                x = np.random.uniform(0.05, W - 0.05, N)
                y = np.random.uniform(0.05, H - 0.05, N)
                r = np.full(N, 0.04)
            
            # Gradient-based optimization
            lr = 0.01
            penalty_weight = 10.0
            softness = 0.01  # Softness parameter for smooth penalty
            
            best_local_sum = 0.0
            best_local_x = x.copy()
            best_local_y = y.copy()
            best_local_r = r.copy()
            
            max_iter = 3000
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute gradient of objective
                # Objective = sum(r) - penalty_weight * penalty
                # penalty = sum of soft violations
                
                # Boundary violations (soft)
                # x_i - r_i >= 0 -> violation = max(0, r_i - x_i)
                # x_i + r_i <= W -> violation = max(0, x_i + r_i - W)
                # y_i - r_i >= 0 -> violation = max(0, r_i - y_i)
                # y_i + r_i <= H -> violation = max(0, y_i + r_i - H)
                
                # Soft penalty using smooth approximation
                # Use smooth max: logsumexp approximation
                def smooth_max(a, b):
                    """Smooth maximum of a and b"""
                    m = np.maximum(a, b)
                    return m + softness * np.log(np.exp((a - m) / softness) + np.exp((b - m) / softness))
                
                # Compute soft boundary violations
                viol_x_left = smooth_max(r - x, np.zeros(N))
                viol_x_right = smooth_max(x + r - W, np.zeros(N))
                viol_y_bottom = smooth_max(r - y, np.zeros(N))
                viol_y_top = smooth_max(y + r - H, np.zeros(N))
                
                # Overlap violations (soft)
                dx = x[i_idx] - x[j_idx]
                dy = y[i_idx] - y[j_idx]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                req = r[i_idx] + r[j_idx]
                overlap = smooth_max(req - dist, np.zeros(len(i_idx)))
                
                # Total penalty
                penalty = np.sum(viol_x_left) + np.sum(viol_x_right) + \
                          np.sum(viol_y_bottom) + np.sum(viol_y_top) + \
                          penalty_weight * np.sum(overlap)
                
                # Objective
                obj = np.sum(r) - penalty
                
                # Track best
                if obj > best_local_sum:
                    best_local_sum = obj
                    best_local_x = x.copy()
                    best_local_y = y.copy()
                    best_local_r = r.copy()
                
                # Compute gradients
                # d(obj)/dx, d(obj)/dy, d(obj)/dr
                grad_x = np.zeros(N)
                grad_y = np.zeros(N)
                grad_r = np.zeros(N)
                
                # From boundary violations
                # d(viol_x_left)/dx = -1 if x < r else 0 (soft version)
                # Use derivative of smooth_max
                def d_smooth_max_da(a, b):
                    m = np.maximum(a, b)
                    ea = np.exp((a - m) / softness)
                    eb = np.exp((b - m) / softness)
                    return ea / (ea + eb)
                
                def d_smooth_max_db(a, b):
                    m = np.maximum(a, b)
                    ea = np.exp((a - m) / softness)
                    eb = np.exp((b - m) / softness)
                    return eb / (ea + eb)
                
                # Gradient contributions from boundary
                # viol_x_left = smooth_max(r - x, 0)
                # d/dx = -d_smooth_max_da(r-x, 0)
                # d/dr = d_smooth_max_da(r-x, 0)
                
                # For simplicity, use finite differences or approximate gradients
                # Let's use a simpler approach: compute gradients analytically for the main terms
                
                # Overlap gradient (dominant term)
                # overlap_ij = smooth_max(r_i + r_j - dist_ij, 0)
                # d(overlap)/dx_i = (x_i - x_j) / dist_ij * d_smooth_max_da
                # d(overlap)/dr_i = d_smooth_max_da
                
                overlap_grad_x = np.zeros(N)
                overlap_grad_y = np.zeros(N)
                overlap_grad_r = np.zeros(N)
                
                for idx in range(len(i_idx)):
                    i = i_idx[idx]
                    j = j_idx[idx]
                    d = dist[idx]
                    if d < 1e-9:
                        continue
                    
                    # Derivative of smooth_max(req - dist, 0) w.r.t. (req - dist)
                    a = req[idx] - d
                    b = 0.0
                    m = max(a, b)
                    ea = np.exp((a - m) / softness)
                    eb = np.exp((b - m) / softness)
                    da_darg = ea / (ea + eb)
                    
                    # d(dist)/dx_i = (x_i - x_j) / dist
                    ddist_dx_i = dx[idx] / d
                    ddist_dx_j = -dx[idx] / d
                    ddist_dy_i = dy[idx] / d
                    ddist_dy_j = -dy[idx] / d
                    
                    # d(overlap)/dx_i = -ddist_dx_i * da_darg * penalty_weight
                    # d(overlap)/dr_i = da_darg * penalty_weight
                    overlap_grad_x[i] += -ddist_dx_i * da_darg * penalty_weight
                    overlap_grad_x[j] += -ddist_dx_j * da_darg * penalty_weight
                    overlap_grad_y[i] += -ddist_dy_i * da_darg * penalty_weight
                    overlap_grad_y[j] += -ddist_dy_j * da_darg * penalty_weight
                    overlap_grad_r[i] += da_darg * penalty_weight
                    overlap_grad_r[j] += da_darg * penalty_weight
                
                # Add gradient from sum(r)
                grad_r += 1.0
                
                # Add gradient from overlap penalty
                grad_x += overlap_grad_x
                grad_y += overlap_grad_y
                grad_r -= overlap_grad_r  # Minimize penalty
                
                # Boundary gradients (approximate)
                # For x: if x < r, push right; if x > W - r, push left
                for i in range(N):
                    if x[i] < r[i]:
                        grad_x[i] += penalty_weight
                    elif x[i] > W - r[i]:
                        grad_x[i] -= penalty_weight
                    if y[i] < r[i]:
                        grad_y[i] += penalty_weight
                    elif y[i] > H - r[i]:
                        grad_y[i] -= penalty_weight
                    # Radius constraint from boundaries
                    max_r = min(x[i], W - x[i], y[i], H - y[i])
                    if r[i] > max_r:
                        grad_r[i] -= penalty_weight
                
                # Update with momentum-like step
                # Simple gradient ascent
                x += lr * grad_x
                y += lr * grad_y
                r += lr * grad_r
                
                # Clamp to valid ranges
                x = np.clip(x, 0.01, W - 0.01)
                y = np.clip(y, 0.01, H - 0.01)
                r = np.clip(r, 0.01, min(W, H) / 2)
                
                # Decay learning rate
                lr *= 0.999
                
                # If lr too small, break
                if lr < 1e-6:
                    break
            
            # After optimization, check validity
            if best_local_sum > best_sum:
                # Verify the best solution found
                if _check_validity(best_local_x, best_local_y, best_local_r, W, H, i_idx, j_idx):
                    best_sum = best_local_sum
                    best_solution = np.column_stack([best_local_x, best_local_y, best_local_r])
    
    # If no valid solution found, return a fallback
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x_c = (i + 0.5) * (1.0 / side)
                    y_c = (j + 0.5) * (1.0 / side)
                    r_c = 0.5 / side - 1e-4
                    cells.append([x_c, y_c, r_c])
        return np.array(cells)
    
    return best_solution

def _check_validity(x, y, r, W, H, i_idx, j_idx):
    """Check if circles are valid (no overlaps, within bounds)"""
    tol = 1e-6
    if np.any(x - r < -tol) or np.any(x + r > W + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > H + tol):
        return False
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx*dx + dy*dy)
    req = r[i_idx] + r[j_idx]
    if np.any(dist < req - tol):
        return False
    if np.any(r <= 0):
        return False
    return True
