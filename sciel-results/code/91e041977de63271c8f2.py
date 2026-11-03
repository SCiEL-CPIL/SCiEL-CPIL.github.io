import numpy as np
import scipy.optimize as opt
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    best_value = -np.inf
    best_solution = None
    
    # Precompute pairs for efficiency
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    
    def objective_and_grads(params):
        """
        params: shape (3*N,) -> [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
        Returns: (value, gradient)
        Value is -sum(r) + penalty * violations
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:]
        
        # 1. Sum of radii (we minimize -sum)
        val = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2*N:] = -1.0
        
        # 2. Boundary Violations
        # x < r  -> x - r < 0. Violation is max(0, r - x)
        # x > 1-r -> x - (1-r) > 0. Violation is max(0, x - (1-r))
        # Similarly for y
        
        # Lower bounds: x >= r, y >= r
        viol_x_low = np.maximum(0, r - x)
        viol_y_low = np.maximum(0, r - y)
        # Upper bounds: x <= 1-r, y <= 1-r
        viol_x_high = np.maximum(0, x - (1 - r))
        viol_y_high = np.maximum(0, y - (1 - r))
        
        # Sum of squared violations
        b_viol_sq = np.sum(viol_x_low**2) + np.sum(viol_y_low**2) + np.sum(viol_x_high**2) + np.sum(viol_y_high**2)
        val += 1000.0 * b_viol_sq
        
        # Gradients for boundary violations
        # d/dx max(0, r-x)^2 = 2(r-x) * (-1) if r>x else 0
        # d/dx max(0, x-(1-r))^2 = 2(x-(1-r)) * 1 if x>1-r else 0
        
        # We need to be careful with gradients at kinks, but L-BFGS-B is robust to slight noise.
        # Let's compute them explicitly.
        
        # Lower x: max(0, r-x)^2
        mask_xl = (r > x)
        grad[:N][mask_xl] += 1000.0 * 2 * (r[mask_xl] - x[mask_xl]) * (-1)
        grad[2*N:][mask_xl] += 1000.0 * 2 * (r[mask_xl] - x[mask_xl]) * (1)
        
        # Upper x: max(0, x-(1-r))^2
        mask_xh = (x > 1 - r)
        grad[:N][mask_xh] += 1000.0 * 2 * (x[mask_xh] - (1 - r[mask_xh])) * (1)
        grad[2*N:][mask_xh] += 1000.0 * 2 * (x[mask_xh] - (1 - r[mask_xh])) * (-1)
        
        # Lower y
        mask_yl = (r > y)
        grad[N:2*N][mask_yl] += 1000.0 * 2 * (r[mask_yl] - y[mask_yl]) * (-1)
        grad[2*N:][mask_yl] += 1000.0 * 2 * (r[mask_yl] - y[mask_yl]) * (1)
        
        # Upper y
        mask_yh = (y > 1 - r)
        grad[N:2*N][mask_yh] += 1000.0 * 2 * (y[mask_yh] - (1 - r[mask_yh])) * (1)
        grad[2*N:][mask_yh] += 1000.0 * 2 * (y[mask_yh] - (1 - r[mask_yh])) * (-1)
        
        # 3. Overlap Violations
        # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
        # viol_ij = max(0, (ri+rj) - d_ij)^2
        
        # Vectorized calculation for all pairs
        # xi, xj
        xi = x[pairs[:, 0]]
        xj = x[pairs[:, 1]]
        yi = y[pairs[:, 0]]
        yj = y[pairs[:, 1]]
        ri = r[pairs[:, 0]]
        rj = r[pairs[:, 1]]
        
        dx = xi - xj
        dy = yi - yj
        d = np.sqrt(dx**2 + dy**2 + 1e-12) # Avoid div by zero
        
        sum_r = ri + rj
        overlap = np.maximum(0, sum_r - d)
        
        o_viol_sq = np.sum(overlap**2)
        val += 1000.0 * o_viol_sq
        
        # Gradients for overlaps
        # d/dx_i (max(0, sum_r - d)^2)
        # = 2 * (sum_r - d) * (-1) * d/dx_i (d)  if overlap > 0
        # d/dx_i (d) = (xi - xj) / d = dx / d
        
        mask_o = (overlap > 0)
        if np.any(mask_o):
            # Only compute for active constraints to save time? 
            # The vector is already computed.
            coeff = 2 * overlap[mask_o]
            dx_active = dx[mask_o]
            dy_active = dy[mask_o]
            d_active = d[mask_o]
            
            # Gradient w.r.t x_i
            # term = coeff * (-1) * (dx / d)
            # We need to scatter this back to grad
            # This is tricky with vectorized pairs. 
            # Let's use np.add.at for accumulation
            
            g_xi = coeff * (-1) * (dx_active / d_active)
            g_xj = coeff * (-1) * (-dx_active / d_active) # d/dxj d = (xj-xi)/d = -dx/d
            g_yi = coeff * (-1) * (dy_active / d_active)
            g_yj = coeff * (-1) * (-dy_active / d_active)
            g_ri = coeff * (-1) * (1) # d/dri (sum_r - d) = 1
            g_rj = coeff * (-1) * (1)
            
            idx_i = pairs[mask_o, 0]
            idx_j = pairs[mask_o, 1]
            
            np.add.at(grad[:N], idx_i, 1000.0 * g_xi)
            np.add.at(grad[:N], idx_j, 1000.0 * g_xj)
            np.add.at(grad[N:2*N], idx_i, 1000.0 * g_yi)
            np.add.at(grad[N:2*N], idx_j, 1000.0 * g_yj)
            np.add.at(grad[2*N:], idx_i, 1000.0 * g_ri)
            np.add.at(grad[2*N:], idx_j, 1000.0 * g_rj)
            
        return val, grad

    def is_valid(sol, tol=1e-5):
        x = sol[:N]
        y = sol[N:2*N]
        r = sol[2*N:]
        # Check bounds
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        # Check overlaps
        for i, j in pairs:
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
        return True

    def get_value(sol):
        return np.sum(sol[2*N:])

    # Initializations
    # 1. Jittered Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                # Add small jitter
                cx += np.random.uniform(-0.01, 0.01)
                cy += np.random.uniform(-0.01, 0.01)
                # Clip to valid range for r=0.05
                cx = np.clip(cx, 0.05, 0.95)
                cy = np.clip(cy, 0.05, 0.95)
                cells.append((cx, cy, 0.05))
    
    # 2. Random Starts
    n_restarts = 10
    starts = [np.array(cells, dtype=float)]
    for _ in range(n_restarts):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        starts.append(np.concatenate([x, y, r]))

    for start in starts:
        if time.time() - start_time > deadline:
            break
        
        # Run optimization
        try:
            # Bounds: x, y in [0,1], r in [0, 0.5]
            bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
            
            res = opt.minimize(
                objective_and_grads,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if res.success or res.status in [0, 1, 2, 3, 4, 5]: # Common success statuses
                sol = res.x
                # Refine: If valid, we have a candidate.
                # Sometimes L-BFGS-B converges to a point with tiny overlaps due to tolerance.
                # We can try to "push" it back or just check with tight tolerance.
                if is_valid(sol, tol=1e-4):
                    val = get_value(sol)
                    if val > best_value:
                        best_value = val
                        best_solution = sol.copy()
                        
        except Exception as e:
            pass

    # If no valid solution found, return the trivial grid (fallback)
    if best_solution is None:
        # Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
