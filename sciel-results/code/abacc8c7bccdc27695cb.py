import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Precompute pair indices for vectorized overlap checks
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def is_valid(x, y, r, tol=1e-9):
        if np.any(x < r - tol) or np.any(1 - x < r - tol) or np.any(y < r - tol) or np.any(1 - y < r - tol):
            return False
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(r[i_idx] + r[j_idx] > dist + tol):
            return False
        return True
    
    def project_valid(x, y, r, tol=1e-9):
        """Project to validity by shrinking radii."""
        r = np.minimum(r, np.minimum(x, 1 - x))
        r = np.minimum(r, np.minimum(y, 1 - y))
        for _ in range(100):
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            violations = r[i_idx] + r[j_idx] - dist
            if np.max(violations) < tol:
                break
            r *= 0.98
            r = np.minimum(r, np.minimum(x, 1 - x))
            r = np.minimum(r, np.minimum(y, 1 - y))
        return r
    
    def soft_penalty(x, y, r, beta=50.0, weight=1000.0):
        """Smooth penalty using soft-max for overlap terms."""
        # Boundary violations (smooth)
        bx_pen = np.logaddexp(0, -beta * (x - r)) / beta
        bx_pen += np.logaddexp(0, -beta * (1 - x - r)) / beta
        by_pen = np.logaddexp(0, -beta * (y - r)) / beta
        by_pen += np.logaddexp(0, -beta * (1 - y - r)) / beta
        boundary_pen = np.sum(bx_pen) + np.sum(by_pen)
        
        # Pairwise overlap (smooth soft-max)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        overlap = r[i_idx] + r[j_idx] - dist
        # Soft-max of positive overlaps
        max_ov = np.max(overlap)
        if max_ov > 1e-12:
            pair_pen = np.sum(np.exp(beta * overlap) / beta) - max_ov / beta
        else:
            pair_pen = 0.0
        
        return weight * (boundary_pen + pair_pen)
    
    def objective(vars, weight=1000.0, beta=50.0):
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[2*N:3*N]
        return -np.sum(r) + soft_penalty(x, y, r, beta, weight)
    
    def objective_grad(vars, weight=1000.0, beta=50.0):
        """Objective with gradient for better convergence."""
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[2*N:3*N]
        
        val = -np.sum(r) + soft_penalty(x, y, r, beta, weight)
        
        # Numerical gradient (finite differences) for reliability
        eps = 1e-7
        grad = np.zeros_like(vars)
        for i in range(len(vars)):
            vp = vars.copy()
            vm = vars.copy()
            vp[i] += eps
            vm[i] -= eps
            grad[i] = (objective(vp, weight, beta) - objective(vm, weight, beta)) / (2 * eps)
        
        return val, grad
    
    def initial_greedy_best_fit():
        """Initialize by placing circles greedily with best-fit positions."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Start with a hexagonal-like pattern for first few, then greedy
        # Place first circle at center
        x[0] = 0.5
        y[0] = 0.5
        r[0] = 0.15
        
        for k in range(1, N):
            # Try to find best position for circle k
            best_pos = None
            best_min_dist = 0
            best_radius = 0
            
            # Candidate positions: grid + around existing circles
            candidates = []
            
            # Grid candidates
            for gx in np.linspace(0.05, 0.95, 15):
                for gy in np.linspace(0.05, 0.95, 15):
                    candidates.append((gx, gy))
            
            # Around existing circles
            for i in range(k):
                for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                    for dist_factor in [1.0, 1.5, 2.0, 2.5]:
                        cx = x[i] + dist_factor * r[i] * np.cos(angle)
                        cy = y[i] + dist_factor * r[i] * np.sin(angle)
                        if 0.02 <= cx <= 0.98 and 0.02 <= cy <= 0.98:
                            candidates.append((cx, cy))
            
            # Evaluate each candidate
            for cx, cy in candidates:
                # Max radius that fits at this position
                max_r = min(cx, 1 - cx, cy, 1 - cy)
                for i in range(k):
                    d = np.sqrt((cx - x[i])**2 + (cy - y[i])**2)
                    max_r = min(max_r, max(0, d - r[i]))
                
                if max_r > best_min_dist:
                    best_min_dist = max_r
                    best_pos = (cx, cy)
                    best_radius = max_r
            
            if best_pos is not None and best_radius > 0.005:
                x[k] = best_pos[0]
                y[k] = best_pos[1]
                r[k] = best_radius * 0.98  # Small margin for safety
            else:
                # Fallback: small circle in a random spot
                x[k] = np.random.uniform(0.1, 0.9)
                y[k] = np.random.uniform(0.1, 0.9)
                r[k] = 0.02
        
        # Project to validity
        r = project_valid(x, y, r)
        return x, y, r
    
    # Use greedy best-fit initialization
    x, y, r = initial_greedy_best_fit()
    
    best_score = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())
    
    # Main optimization loop with multiple restarts
    max_restarts = 12
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # For each restart, do a gradient-based optimization
        current_x, current_y, current_r = best_state
        
        # Initial variables
        vars0 = np.concatenate([current_x, current_y, current_r])
        
        # Bounds
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        # Stage 1: L-BFGS-B with smooth penalty
        try:
            result = minimize(
                objective,
                vars0,
                args=(1000.0, 50.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
            )
        except Exception:
            continue
        
        if result.success or result.fun < objective(vars0, 1000.0, 50.0):
            new_vars = result.x
            new_x = new_vars[:N]
            new_y = new_vars[N:2*N]
            new_r = new_vars[2*N:3*N]
            
            # Project to validity
            new_r = project_valid(new_x, new_y, new_r)
            
            score = np.sum(new_r)
            if score > best_score:
                best_score = score
                best_state = (new_x.copy(), new_y.copy(), new_r.copy())
        
        # Stage 2: SLSQP polish with tighter bounds and higher precision
        if time.time() > deadline - 8.0:
            break
        
        polish_x, polish_y, polish_r = best_state
        vars0 = np.concatenate([polish_x, polish_y, polish_r])
        
        # Tighter bounds based on current positions
        tight_bounds = []
        for i in range(N):
            x_lo = max(0, polish_x[i] - 0.03)
            x_hi = min(1, polish_x[i] + 0.03)
            tight_bounds.append((x_lo, x_hi))
        for i in range(N):
            y_lo = max(0, polish_y[i] - 0.03)
            y_hi = min(1, polish_y[i] + 0.03)
            tight_bounds.append((y_lo, y_hi))
        for i in range(N):
            r_lo = max(0, polish_r[i] * 0.9)
            r_hi = min(0.5, polish_r[i] * 1.1)
            tight_bounds.append((r_lo, r_hi))
        
        try:
            result2 = minimize(
                objective,
                vars0,
                args=(5000.0, 100.0),
                method='SLSQP',
                bounds=tight_bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'eps': 1e-10}
            )
        except Exception:
            continue
        
        if result2.success or result2.fun < objective(vars0, 5000.0, 100.0):
            new_vars = result2.x
            new_x = new_vars[:N]
            new_y = new_vars[N:2*N]
            new_r = new_vars[2*N:3*N]
            
            # Project to validity
            new_r = project_valid(new_x, new_y, new_r)
            
            score = np.sum(new_r)
            if score > best_score:
                best_score = score
                best_state = (new_x.copy(), new_y.copy(), new_r.copy())
        
        # Stage 3: High-precision final polish with very tight bounds
        if time.time() > deadline - 3.0:
            break
        
        hp_x, hp_y, hp_r = best_state
        vars0 = np.concatenate([hp_x, hp_y, hp_r])
        
        # Very tight bounds
        hp_bounds = []
        for i in range(N):
            hp_bounds.append((max(0, hp_x[i] - 0.01), min(1, hp_x[i] + 0.01)))
        for i in range(N):
            hp_bounds.append((max(0, hp_y[i] - 0.01), min(1, hp_y[i] + 0.01)))
        for i in range(N):
            hp_bounds.append((max(0, hp_r[i] * 0.98), min(0.5, hp_r[i] * 1.02)))
        
        try:
            result3 = minimize(
                objective,
                vars0,
                args=(10000.0, 200.0),
                method='SLSQP',
                bounds=hp_bounds,
                options={'maxiter': 300, 'ftol': 1e-18, 'eps': 1e-12}
            )
        except Exception:
            continue
        
        if result3.success or result3.fun < objective(vars0, 10000.0, 200.0):
            new_vars = result3.x
            new_x = new_vars[:N]
            new_y = new_vars[N:2*N]
            new_r = new_vars[2*N:3*N]
            
            new_r = project_valid(new_x, new_y, new_r)
            
            score = np.sum(new_r)
            if score > best_score:
                best_score = score
                best_state = (new_x.copy(), new_y.copy(), new_r.copy())
        
        # Perturb for next restart to explore different local minima
        if restart < max_restarts - 1:
            perturb_x = best_state[0] + np.random.uniform(-0.015, 0.015, N)
            perturb_y = best_state[1] + np.random.uniform(-0.015, 0.015, N)
            perturb_r = best_state[2] + np.random.uniform(-0.005, 0.005, N)
            
            perturb_x = np.clip(perturb_x, 0, 1)
            perturb_y = np.clip(perturb_y, 0, 1)
            perturb_r = np.clip(perturb_r, 0, 0.5)
            
            perturb_r = project_valid(perturb_x, perturb_y, perturb_r)
            
            # Use perturbed state as next starting point
            best_state = (perturb_x.copy(), perturb_y.copy(), perturb_r.copy())
            # But remember the best score separately
            # Actually, keep best_state as the best found, and use a separate var for exploration
            # Let's revert: use best_state for next restart, add perturbation to create new start
            best_state = (best_state[0].copy(), best_state[1].copy(), best_state[2].copy())
    
    # Final validity check and projection
    bx, by, br = best_state
    br = project_valid(bx, by, br)
    
    final_state = np.column_stack([bx, by, br])
    return final_state
