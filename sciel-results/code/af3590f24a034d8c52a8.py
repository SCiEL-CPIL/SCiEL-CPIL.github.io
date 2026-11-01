import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Helper to check validity
    def is_valid(xy, r, tol=1e-7):
        if np.any(r <= 0) or np.any(r > 0.5):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    # Objective function for scipy.optimize.minimize (L-BFGS-B)
    # We minimize negative sum of radii + penalties
    def objective(params):
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        
        # Objective: -sum(r)
        obj = -np.sum(r)
        
        # Penalty for boundary violations
        # d_left = x - r, d_right = 1 - x - r, etc.
        # We want d >= 0. Penalty = sum(softplus(-d)^2) or similar.
        # Let's use a simple quadratic penalty for violations.
        d_left = xy[:, 0] - r
        d_right = 1.0 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1.0 - xy[:, 1] - r
        
        viol = np.concatenate([d_left, d_right, d_bottom, d_top])
        # Only penalize if < 0
        neg_viol = np.maximum(0, -viol)
        penalty_boundary = np.sum(neg_viol**2)
        
        # Penalty for overlaps
        # dist < r_i + r_j
        # overlap = r_i + r_j - dist
        # Use log-sum-exp for smoothness? Or just sum of squared overlaps.
        # Sum of squared overlaps is C1 continuous, good for L-BFGS-B.
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, None] + r[None, :]
        
        overlap = min_dist - dist
        # Mask diagonal
        iu = np.triu_indices(N, k=1)
        overlap_iu = overlap[iu]
        neg_overlap = np.maximum(0, overlap_iu)
        penalty_overlap = np.sum(neg_overlap**2)
        
        # Weights
        w_b = 100.0
        w_o = 100.0
        
        return obj + w_b * penalty_boundary + w_o * penalty_overlap

    def gradient(params):
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        
        grad = np.zeros_like(params)
        
        # Gradient of -sum(r)
        grad[2*N:] = -1.0
        
        # Gradient of boundary penalty
        d_left = xy[:, 0] - r
        d_right = 1.0 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1.0 - xy[:, 1] - r
        
        # d/dx_left = 1, d/dx_right = -1, etc.
        # Penalty term: w_b * sum(max(0, -d)^2)
        # If d < 0, d/dx of (-d)^2 = 2(-d)(-1) = 2d.
        # So grad_x += w_b * 2 * d * (indicator d<0)
        
        def grad_penalty(d, sign):
            # d is distance from boundary. sign is +1 for left/bottom (x-r), -1 for right/top (1-x-r)
            # Wait, d_left = x - r. If x < r, d_left < 0.
            # Penalty = w * max(0, -(x-r))^2 = w * (r-x)^2 if x<r.
            # d/dx = w * 2 * (r-x) * (-1) = -2w(r-x) = 2w(x-r) = 2w*d.
            # So if d < 0, grad += 2*w*d.
            # If d >= 0, grad += 0.
            mask = d < 0
            return 2.0 * 100.0 * d * mask # w_b = 100
        
        # x components
        gx_left = grad_penalty(d_left, 1)
        gx_right = grad_penalty(d_right, -1) # Note: d_right = 1-x-r. d/dx(1-x-r) = -1.
        # Let's re-derive carefully.
        # P = w * (max(0, r-x))^2. If x<r, P = w(r-x)^2. dP/dx = 2w(r-x)(-1) = -2w(r-x) = 2w(x-r).
        # d_left = x-r. So dP/dx = 2w * d_left.
        # P = w * (max(0, x-1+r))^2. If x>1-r, P = w(x-1+r)^2. dP/dx = 2w(x-1+r).
        # d_right = 1-x-r. So x-1+r = -d_right. dP/dx = 2w(-d_right) = -2w*d_right.
        
        grad[:N, 0] += 2.0 * 100.0 * (np.where(d_left < 0, d_left, 0) - np.where(d_right < 0, d_right, 0))
        grad[N:2*N, 0] += 0.0 # y doesn't affect x boundary directly? No, xy is (x,y).
        # Wait, params[:2N] is flattened xy.
        # params[0:N] are x, params[N:2N] are y.
        
        # Let's restart gradient calculation properly.
        grad = np.zeros_like(params)
        # Objective -sum(r)
        grad[2*N:] = -1.0
        
        # Boundary Penalties
        # x-coords are params[0:N]
        # y-coords are params[N:2N]
        
        # x-boundaries
        d_left = xy[:, 0] - r
        d_right = 1.0 - xy[:, 0] - r
        # grad_x += 2*w_b*(d_left if d_left<0 else 0) + 2*w_b*(-d_right if d_right<0 else 0)
        # Actually:
        # P_left = w * (max(0, r-x))^2. dP/dx = -2w(r-x) = 2w(x-r) = 2w*d_left.
        # P_right = w * (max(0, x-1+r))^2. dP/dx = 2w(x-1+r) = -2w(1-x-r) = -2w*d_right.
        grad[0:N, 0] += 2.0 * 100.0 * (np.where(d_left < 0, d_left, 0) - np.where(d_right < 0, d_right, 0))
        
        # y-boundaries
        d_bottom = xy[:, 1] - r
        d_top = 1.0 - xy[:, 1] - r
        grad[N:2*N, 0] += 2.0 * 100.0 * (np.where(d_bottom < 0, d_bottom, 0) - np.where(d_top < 0, d_top, 0))
        
        # r-boundaries
        # P_left = w * (max(0, r-x))^2. dP/dr = 2w(r-x) = 2w*d_left.
        # P_right = w * (max(0, x-1+r))^2. dP/dr = 2w(x-1+r) = -2w*d_right.
        # P_bottom = w * (max(0, r-y))^2. dP/dr = 2w*d_bottom.
        # P_top = w * (max(0, y-1+r))^2. dP/dr = -2w*d_top.
        grad[2*N:] += 2.0 * 100.0 * (
            np.where(d_left < 0, d_left, 0) - np.where(d_right < 0, d_right, 0) +
            np.where(d_bottom < 0, d_bottom, 0) - np.where(d_top < 0, d_top, 0)
        )
        
        # Overlap Penalties
        # P = w_o * sum_{i<j} (max(0, r_i+r_j-d_ij))^2
        # d/dx_i: 2*w_o * (r_i+r_j-d_ij) * d/dx_i(d_ij) * (-1)?
        # Let o_ij = r_i+r_j-d_ij. P = sum w o_ij^2.
        # dP/dx_i = sum_j 2 w o_ij * d(o_ij)/dx_i
        # d(o_ij)/dx_i = - d(d_ij)/dx_i = - (x_i - x_j)/d_ij
        # So dP/dx_i = -2w sum_j o_ij (x_i-x_j)/d_ij
        # Note: only if o_ij > 0.
        
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, None] + r[None, :]
        overlap = min_dist - dist
        
        # Mask diagonal and upper triangle for calculation, but force is symmetric?
        # No, force on i from j is distinct.
        # We sum over all j != i.
        # o_ij = overlap[i,j]
        # Term for i: -2 * w_o * o_ij * (x_i - x_j) / d_ij
        # We can use matrix multiplication.
        # coeff[i,j] = 2 * w_o * overlap[i,j] / dist[i,j] if overlap[i,j] > 0 else 0
        # Then force_x[i] = - sum_j coeff[i,j] * dx[i,j]
        # force_y[i] = - sum_j coeff[i,j] * dy[i,j]
        
        coeff = np.zeros((N, N))
        mask = overlap > 0
        # Avoid div by zero
        safe_dist = np.where(dist > 1e-9, dist, 1.0)
        coeff[mask] = 2.0 * 100.0 * overlap[mask] / safe_dist[mask]
        
        # Force on x
        # F_x[i] = - sum_j coeff[i,j] * dx[i,j]
        # dx[i,j] = x_i - x_j
        # This is equivalent to - (coeff @ dx).T? No.
        # Sum_j coeff[i,j] * dx[i,j] is row i of coeff @ dx.
        grad[0:N, 0] -= np.sum(coeff * dx, axis=1)
        grad[N:2*N, 0] -= np.sum(coeff * dy, axis=1)
        
        # Force on r
        # dP/dr_i = sum_j 2 w_o o_ij * d(o_ij)/dr_i = sum_j 2 w_o o_ij * 1 = 2 w_o sum_j o_ij
        # Only where o_ij > 0.
        grad[2*N:] += 2.0 * 100.0 * np.sum(np.where(overlap > 0, overlap, 0), axis=1)
        
        return grad

    # Initializations
    inits = []
    
    # 1. 6x6 Grid
    side = 6
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
        r[i] = 0.07
    inits.append((xy, r))
    
    # 2. 5x7 Asymmetric Grid
    rows, cols = 5, 7
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // cols
        col = i % cols
        xy[i] = [(col + 0.5)/cols, (row + 0.5)/rows]
        r[i] = 0.07
    inits.append((xy, r))
    
    # 3. 7x5 Asymmetric Grid
    rows, cols = 7, 5
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // cols
        col = i % cols
        xy[i] = [(col + 0.5)/cols, (row + 0.5)/rows]
        r[i] = 0.07
    inits.append((xy, r))
    
    # 4. Random
    rng = np.random.RandomState(42)
    xy = rng.uniform(0.1, 0.9, size=(N, 2))
    r = rng.uniform(0.05, 0.08, size=N)
    inits.append((xy, r))
    
    # 5. Hexagonal
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    dx = 0.18
    dy = dx * np.sqrt(3) / 2
    idx = 0
    for row in range(6):
        for col in range(6):
            if idx >= N: break
            x = 0.1 + col * dx + (0.09 if row % 2 == 1 else 0)
            y = 0.1 + row * dy
            if x < 0.95 and y < 0.95:
                xy[idx] = [x, y]
                r[idx] = 0.08
                idx += 1
    for k in range(idx, N):
        xy[k] = rng.uniform(0.1, 0.9, size=2)
        r[k] = 0.05
    inits.append((xy, r))
    
    best_result = None
    best_score = -np.inf
    
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    for _ in range(N):
        bounds.append((0.01, 0.5)) # r
        
    for i, (xy_init, r_init) in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        x0 = np.concatenate([xy_init.flatten(), r_init])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.success:
                xy_opt = res.x[:2*N].reshape(N, 2)
                r_opt = res.x[2*N:]
                
                # Check validity
                if is_valid(xy_opt, r_opt):
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([xy_opt, r_opt])
                        
                # Even if not strictly valid due to tolerance, check if it's close
                # and maybe fix it?
                # For now, just track valid ones.
                
        except Exception as e:
            pass

    # Final Polish
    if best_result is not None:
        xy_best = best_result[:, :2]
        r_best = best_result[:, 2]
        
        # Run a shorter, tighter optimization
        x0 = np.concatenate([xy_best.flatten(), r_best])
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
            if res.success:
                xy_opt = res.x[:2*N].reshape(N, 2)
                r_opt = res.x[2*N:]
                if is_valid(xy_opt, r_opt):
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([xy_opt, r_opt])
        except Exception:
            pass

    if best_result is None:
        # Fallback
        side = 6
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.07) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
