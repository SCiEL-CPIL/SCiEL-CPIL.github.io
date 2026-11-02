import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def _is_valid(cx, cy, r, W, H, tol=1e-9):
        if W <= 0.01 or H <= 0.01:
            return False
        if np.any(cx - r < -tol) or np.any(cy - r < -tol):
            return False
        if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
            return False
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d2 = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < r_sum[iu]**2 - tol):
            return False
        return True

    def _force_relax(x, y, r, W, H, max_iter=3000, dt_init=0.01):
        """
        Vectorized force-directed relaxation to resolve overlaps and boundary violations.
        """
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        
        dt = dt_init
        min_dt = 1e-7
        
        for it in range(max_iter):
            if time.time() > deadline - 8:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Boundary forces (strong push)
            viol_left = np.maximum(r - x, 0)
            fx += viol_left * 500
            viol_right = np.maximum(x + r - W, 0)
            fx -= viol_right * 500
            viol_bottom = np.maximum(r - y, 0)
            fy += viol_bottom * 500
            viol_top = np.maximum(y + r - H, 0)
            fy -= viol_top * 500
            
            # Pairwise forces
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d2 = dx*dx + dy*dy
            d = np.sqrt(np.maximum(d2, 1e-12))
            r_sum = r[:, None] + r[None, :]
            
            # Overlap amount
            overlap = np.maximum(r_sum - d, 0)
            
            # Check convergence
            max_overlap = np.max(overlap)
            max_viol = max(np.max(viol_left), np.max(viol_right), np.max(viol_bottom), np.max(viol_top))
            if max_overlap < 1e-10 and max_viol < 1e-10:
                break
            
            # Forces from overlap
            # Avoid division by zero
            ux = np.where(d > 1e-12, dx / d, 0)
            uy = np.where(d > 1e-12, dy / d, 0)
            
            # Force magnitude proportional to overlap
            f_mag = overlap * 10.0
            
            # Sum forces for each i
            fx += np.sum(f_mag * ux, axis=1)
            fy += np.sum(f_mag * uy, axis=1)
            
            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Clip to box to prevent escaping (hard constraint)
            # This is crucial for stability
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            # Decay dt
            dt *= 0.995
            
        return x, y, r

    def _expand_and_relax(x, y, r, W, H, max_steps=500):
        """
        Iteratively increase radii and relax positions.
        Uses binary search on scale factor to find max valid expansion.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
                
            # Try to expand
            # We want to find the maximum factor s such that r*s is packable.
            # Let's try a range of factors.
            # Start with a large factor, if it fails, binary search down.
            # Or just try a fixed small increment?
            # Fixed increment is safer but slower.
            # Let's try: increase by 1% first. If valid, continue. If not, try 0.5%, etc.
            
            success = False
            for factor in [1.02, 1.01, 1.005, 1.001]:
                r_new = r * factor
                x_new, y_new, _ = _force_relax(x, y, r_new, W, H, max_iter=2000, dt_init=0.005)
                if _is_valid(x_new, y_new, r_new, W, H):
                    x = x_new
                    y = y_new
                    r = r_new
                    success = True
                    break
            
            if not success:
                break
                
        return x, y, r

    def _slsqp_polish(x, y, r, W, H):
        """
        Final polish using SLSQP to squeeze out last bit of sum(r).
        """
        if time.time() > deadline - 15:
            return x, y, r

        # Variables: x, y, r
        # Objective: maximize sum(r) -> minimize -sum(r)
        # Constraints:
        # 1. x_i >= r_i
        # 2. x_i <= W - r_i
        # 3. y_i >= r_i
        # 4. y_i <= H - r_i
        # 5. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        
        z0 = np.concatenate([x, y, r])
        
        def objective(z):
            return -np.sum(z[2*N:3*N])
        
        def objective_grad(z):
            grad = np.zeros(3*N)
            grad[2*N:3*N] = -1.0
            return grad
            
        constraints = []
        
        # Boundary constraints
        for i in range(N):
            # x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[i] - z[2*N+i],
                'jac': lambda z, i=i: np.concatenate([np.eye(N)[i], -np.eye(N)[i], -np.eye(N)[i]])
            })
            # W - x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i: W - z[i] - z[2*N+i],
                'jac': lambda z, i=i: np.concatenate([-np.eye(N)[i], np.zeros(N), -np.eye(N)[i]])
            })
            # y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[N+i] - z[2*N+i],
                'jac': lambda z, i=i: np.concatenate([np.zeros(N), np.eye(N)[i], -np.eye(N)[i]])
            })
            # H - y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i: H - z[N+i] - z[2*N+i],
                'jac': lambda z, i=i: np.concatenate([np.zeros(N), -np.eye(N)[i], -np.eye(N)[i]])
            })
        
        # Overlap constraints
        # This is expensive to define all 210 constraints with lambdas.
        # SLSQP might struggle with 294 constraints.
        # Let's try a reduced set or just skip SLSQP if it's too slow.
        # Given the time limit, let's only add a few critical overlap constraints
        # or use a penalty method instead.
        
        # Actually, let's just do a few steps of gradient ascent on r with projection?
        # No, SLSQP is better if it converges.
        # Let's try adding overlap constraints.
        
        # To save time, we only add constraints for pairs that are close?
        # Or just all of them and hope SLSQP is fast enough.
        # 210 constraints * 63 vars. Might be slow.
        
        # Alternative: Use scipy.optimize.minimize with method='SLSQP' and maxiter=100.
        
        # Let's define the overlap constraints.
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        # We can vectorize the fun, but jac is hard to write compactly for SLSQP's lambda interface easily.
        # Let's stick to force relax + expand. It's been working well.
        # SLSQP is risky for time.
        
        return x, y, r

    # Main Loop: Try different aspect ratios and initial guesses
    
    # Aspect ratios W (H = 2-W)
    # Perimeter 4 => W + H = 2.
    # Square is W=1, H=1.
    # Let's try a range.
    W_values = np.linspace(0.8, 1.2, 9)
    
    def _get_guesses(W, H, seed_offset=0):
        guesses = []
        # 1. Grid
        side = int(np.ceil(np.sqrt(N)))
        z_grid = np.zeros((N, 3))
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * W
            cy = (row + 0.5) / side * H
            r0 = min(W / side, H / side) * 0.45
            z_grid[i] = [cx, cy, r0]
        guesses.append(z_grid)
        
        # 2. Random
        rng = np.random.default_rng(42 + seed_offset)
        for _ in range(5):
            z_rand = np.zeros((N, 3))
            for i in range(N):
                r0 = rng.uniform(0.05, 0.15)
                cx = rng.uniform(r0, W - r0)
                cy = rng.uniform(r0, H - r0)
                z_rand[i] = [cx, cy, r0]
            guesses.append(z_rand)
            
        return guesses

    for idx, W in enumerate(W_values):
        H = 2.0 - W
        if H < 0.5 or W < 0.5:
            continue
            
        guesses = _get_guesses(W, H, seed_offset=idx*10)
        
        for z0 in guesses:
            if time.time() > deadline - 15:
                break
                
            x0 = z0[:, 0]
            y0 = z0[:, 1]
            r0 = z0[:, 2]
            
            # Phase 1: Force Relax to get a valid packing
            x, y, r = _force_relax(x0, y0, r0, W, H, max_iter=2000)
            
            # Check validity
            if not _is_valid(x, y, r, W, H):
                # If still invalid, shrink radii until valid
                scale = 1.0
                for _ in range(20):
                    scale *= 0.9
                    r_trial = r * scale
                    x_trial, y_trial, _ = _force_relax(x, y, r_trial, W, H, max_iter=200)
                    if _is_valid(x_trial, y_trial, r_trial, W, H):
                        x = x_trial
                        y = y_trial
                        r = r_trial
                        break
                else:
                    continue # Skip this guess if we can't make it valid
            
            # Phase 2: Expand and Relax
            x, y, r = _expand_and_relax(x, y, r, W, H, max_steps=300)
            
            # Final check
            if _is_valid(x, y, r, W, H):
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_result = np.column_stack((x, y, r))
                    
    if best_result is None:
        # Fallback: simple grid
        W = 1.0
        H = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N: break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([cx, cy, r])
        return np.array(cells[:N])
        
    return best_result
