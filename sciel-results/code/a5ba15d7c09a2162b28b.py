import numpy as np
import time
from scipy.optimize import minimize

N = 26
EPS = 1e-9

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
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(r[i_idx] + r[j_idx] > dist + tol):
            return False
        return True
    
    def project_valid(x, y, r, tol=1e-9):
        """Project a state to be valid by shrinking radii."""
        r = np.minimum(r, np.minimum(x, 1 - x))
        r = np.minimum(r, np.minimum(y, 1 - y))
        r = np.maximum(r, 0)
        for _ in range(200):
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx * dx + dy * dy)
            violations = r[i_idx] + r[j_idx] - dist
            if np.max(violations) < tol:
                break
            r *= 0.99
            r = np.minimum(r, np.minimum(x, 1 - x))
            r = np.minimum(r, np.minimum(y, 1 - y))
            r = np.maximum(r, 0)
        return r
    
    def force_anneal(x, y, r, time_budget=5.0):
        """Force-directed annealing to improve packing."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        
        # Initialize temperature
        T = 0.005
        T_min = 1e-7
        cooling = 0.9985
        
        elapsed = 0.0
        step = 0.002
        best_sum = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        no_improve_count = 0
        max_no_improve = 200
        
        while elapsed < time_budget and T > T_min and no_improve_count < max_no_improve:
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Boundary forces
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 100
                if 1 - x[i] < r[i]:
                    fx[i] -= (r[i] - (1 - x[i])) * 100
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 100
                if 1 - y[i] < r[i]:
                    fy[i] -= (r[i] - (1 - y[i])) * 100
            
            # Pairwise repulsion
            for i in range(len(i_idx)):
                a, b = i_idx[i], j_idx[i]
                dx = x[a] - x[b]
                dy = y[a] - y[b]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    dx, dy = 0.001, 0.001
                    dist = np.sqrt(dx * dx + dy * dy)
                
                overlap = r[a] + r[b] - dist
                if overlap > 0:
                    force = overlap * 80
                    fx[a] += force * dx / dist
                    fy[a] += force * dy / dist
                    fx[b] -= force * dx / dist
                    fy[b] -= force * dy / dist
            
            # Move with noise
            noise = T * np.random.randn(N)
            x += step * fx + noise * 0.005
            y += step * fy + noise * 0.005
            
            # Clamp to unit square
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            # Project radii to be valid
            r = project_valid(x, y, r)
            
            current_sum = np.sum(r)
            if current_sum > best_sum:
                best_sum = current_sum
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Cool down
            T *= cooling
            step *= 0.9998
            
            # Update elapsed time
            elapsed = time.time() - start_time
        
        return best_x, best_y, best_r, best_sum
    
    def gradient_optimize(x, y, r, time_budget=3.0):
        """Gradient-based optimization using L-BFGS-B with smooth penalty."""
        def objective(vars):
            vx = vars[:N]
            vy = vars[N:2*N]
            vr = vars[2*N:3*N]
            
            # Smooth penalty using soft-min approach
            # Boundary violations
            pen = 0.0
            # Use smooth max(0, x) approximation: 0.5*(x + sqrt(x^2 + eps^2))
            eps_pen = 1e-4
            def smooth_max(v):
                return 0.5 * (v + np.sqrt(v**2 + eps_pen**2))
            
            pen += 500.0 * np.sum(smooth_max(vr - vx)**2)
            pen += 500.0 * np.sum(smooth_max(vr - (1 - vx))**2)
            pen += 500.0 * np.sum(smooth_max(vr - vy)**2)
            pen += 500.0 * np.sum(smooth_max(vr - (1 - vy))**2)
            
            dx = vx[i_idx] - vx[j_idx]
            dy = vy[i_idx] - vy[j_idx]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = vr[i_idx] + vr[j_idx] - dist
            pen += 500.0 * np.sum(smooth_max(overlap)**2)
            
            return -np.sum(vr) + pen
        
        vars0 = np.concatenate([x, y, r])
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_score = np.sum(r)
        
        try:
            result = minimize(
                objective,
                vars0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_vars = result.x
            new_x = new_vars[:N]
            new_y = new_vars[N:2*N]
            new_r = new_vars[2*N:3*N]
            
            new_r = project_valid(new_x, new_y, new_r)
            score = np.sum(new_r)
            
            if score > best_score:
                best_x, best_y, best_r = new_x, new_y, new_r
                best_score = score
                
        except Exception:
            pass
        
        # Try SLSQP for polish if time allows
        if time.time() < deadline - 5:
            def slsqp_objective(vars):
                vx = vars[:N]
                vy = vars[N:2*N]
                vr = vars[2*N:3*N]
                pen = 0.0
                pen += 1000.0 * np.sum(np.maximum(0, vr - vx)**2)
                pen += 1000.0 * np.sum(np.maximum(0, vr - (1 - vx))**2)
                pen += 1000.0 * np.sum(np.maximum(0, vr - vy)**2)
                pen += 1000.0 * np.sum(np.maximum(0, vr - (1 - vy))**2)
                dx = vx[i_idx] - vx[j_idx]
                dy = vy[i_idx] - vy[j_idx]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = np.maximum(0, vr[i_idx] + vr[j_idx] - dist)
                pen += 1000.0 * np.sum(overlap**2)
                return -np.sum(vr) + pen
            
            def slsqp_grad(vars):
                vx = vars[:N]
                vy = vars[N:2*N]
                vr = vars[2*N:3*N]
                grad = np.zeros(3 * N)
                
                # Objective gradient
                grad[2*N:3*N] -= 1.0
                
                # Boundary penalties
                mask_xl = vr > vx
                mask_xr = vr > (1 - vx)
                mask_yl = vr > vy
                mask_yr = vr > (1 - vy)
                
                if np.any(mask_xl):
                    grad[:N][mask_xl] += 2000.0 * (vr[mask_xl] - vx[mask_xl])
                    grad[2*N:3*N][mask_xl] += 2000.0 * (vr[mask_xl] - vx[mask_xl])
                
                if np.any(mask_xr):
                    grad[:N][mask_xr] -= 2000.0 * (vr[mask_xr] - (1 - vx[mask_xr]))
                    grad[2*N:3*N][mask_xr] += 2000.0 * (vr[mask_xr] - (1 - vx[mask_xr]))
                
                if np.any(mask_yl):
                    grad[N:2*N][mask_yl] += 2000.0 * (vr[mask_yl] - vy[mask_yl])
                    grad[2*N:3*N][mask_yl] += 2000.0 * (vr[mask_yl] - vy[mask_yl])
                
                if np.any(mask_yr):
                    grad[N:2*N][mask_yr] -= 2000.0 * (vr[mask_yr] - (1 - vy[mask_yr]))
                    grad[2*N:3*N][mask_yr] += 2000.0 * (vr[mask_yr] - (1 - vy[mask_yr]))
                
                # Overlap penalties
                dx = vx[i_idx] - vx[j_idx]
                dy = vy[i_idx] - vy[j_idx]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = vr[i_idx] + vr[j_idx] - dist
                mask_ov = overlap > 0
                
                if np.any(mask_ov):
                    for k in range(len(mask_ov)):
                        if mask_ov[k]:
                            a = i_idx[k]
                            b = j_idx[k]
                            d = dist[k]
                            if d < 1e-10:
                                continue
                            ox = dx[k] / d
                            oy = dy[k] / d
                            grad[a] += 2000.0 * overlap[k] * ox
                            grad[b] -= 2000.0 * overlap[k] * ox
                            grad[N + a] += 2000.0 * overlap[k] * oy
                            grad[N + b] -= 2000.0 * overlap[k] * oy
                            grad[2*N + a] += 2000.0 * overlap[k]
                            grad[2*N + b] += 2000.0 * overlap[k]
                
                return grad
            
            try:
                result2 = minimize(
                    slsqp_objective,
                    np.concatenate([best_x, best_y, best_r]),
                    method='SLSQP',
                    jac=slsqp_grad,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                
                new_vars = result2.x
                new_x = new_vars[:N]
                new_y = new_vars[N:2*N]
                new_r = new_vars[2*N:3*N]
                
                new_r = project_valid(new_x, new_y, new_r)
                score = np.sum(new_r)
                
                if score > best_score:
                    best_x, best_y, best_r = new_x, new_y, new_r
                    best_score = score
                    
            except Exception:
                pass
        
        return best_x, best_y, best_r, best_score
    
    # Generate multiple initial configurations
    def initial_lattice():
        """5x5 grid with one removed."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx < N:
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    r[idx] = 0.5 / side * 0.98
                    idx += 1
        return x, y, r
    
    def initial_hexagonal():
        """Hexagonal packing approximation."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Hexagonal arrangement
        r0 = 1 / (2 * np.sqrt(3) * np.sqrt(26 / 1.5))
        r0 = min(r0, 0.2)
        
        dx = 2 * r0
        dy = dx * np.sqrt(3) / 2
        
        idx = 0
        for row in range(6):
            for col in range(5):
                if idx >= N:
                    break
                x[idx] = col * dx + (row % 2) * dx / 2 + r0
                y[idx] = row * dy + r0
                r[idx] = r0
                idx += 1
            if idx >= N:
                break
        
        # Center the configuration
        cx = np.mean(x[:N])
        cy = np.mean(y[:N])
        x[:N] -= cx - 0.5
        y[:N] -= cy - 0.5
        
        # Project to valid
        r[:N] = project_valid(x, y, r)
        return x, y, r
    
    def initial_random():
        """Random initialization."""
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.15, N)
        r = project_valid(x, y, r)
        return x, y, r
    
    # Try multiple initializations
    best_score = 0
    best_state = None
    
    initializations = [
        ("lattice", initial_lattice),
        ("hex", initial_hexagonal),
    ]
    
    # Add random initializations
    for i in range(5):
        initializations.append((f"rand{i}", initial_random))
    
    for name, init_func in initializations:
        if time.time() > deadline - 20:
            break
        
        x, y, r = init_func()
        score = np.sum(r)
        
        if score > best_score:
            best_score = score
            best_state = (x.copy(), y.copy(), r.copy())
        
        # Force anneal
        if time.time() > deadline - 10:
            break
        
        x, y, r, score = force_anneal(x, y, r, time_budget=min(8.0, deadline - time.time() - 5))
        
        if score > best_score:
            best_score = score
            best_state = (x.copy(), y.copy(), r.copy())
        
        # Gradient optimization
        if time.time() > deadline - 8:
            break
        
        x, y, r, score = gradient_optimize(x, y, r, time_budget=min(4.0, deadline - time.time() - 3))
        
        if score > best_score:
            best_score = score
            best_state = (x.copy(), y.copy(), r.copy())
    
    # Final polish with more iterations on best state
    if best_state is not None and time.time() < deadline - 10:
        bx, by, br = best_state
        bx, by, br, score = force_anneal(bx, by, br, time_budget=min(8.0, deadline - time.time() - 8))
        if score > best_score:
            best_score = score
            best_state = (bx.copy(), by.copy(), br.copy())
    
    if best_state is not None and time.time() < deadline - 5:
        bx, by, br = best_state
        bx, by, br, score = gradient_optimize(bx, by, br, time_budget=min(6.0, deadline - time.time() - 3))
        if score > best_score:
            best_score = score
            best_state = (bx.copy(), by.copy(), br.copy())
    
    # High-precision final polish
    if best_state is not None and time.time() < deadline - 2:
        bx, by, br = best_state
        # Do a final tight optimization
        def final_objective(vars):
            vx = vars[:N]
            vy = vars[N:2*N]
            vr = vars[2*N:3*N]
            pen = 0.0
            pen += 10000.0 * np.sum(np.maximum(0, vr - vx)**2)
            pen += 10000.0 * np.sum(np.maximum(0, vr - (1 - vx))**2)
            pen += 10000.0 * np.sum(np.maximum(0, vr - vy)**2)
            pen += 10000.0 * np.sum(np.maximum(0, vr - (1 - vy))**2)
            dx = vx[i_idx] - vx[j_idx]
            dy = vy[i_idx] - vy[j_idx]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = np.maximum(0, vr[i_idx] + vr[j_idx] - dist)
            pen += 10000.0 * np.sum(overlap**2)
            return -np.sum(vr) + pen
        
        bounds_final = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        try:
            result_final = minimize(
                final_objective,
                np.concatenate([bx, by, br]),
                method='L-BFGS-B',
                bounds=bounds_final,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_vars = result_final.x
            new_x = new_vars[:N]
            new_y = new_vars[N:2*N]
            new_r = new_vars[2*N:3*N]
            new_r = project_valid(new_x, new_y, new_r)
            score = np.sum(new_r)
            if score > best_score:
                best_score = score
                best_state = (new_x.copy(), new_y.copy(), new_r.copy())
        except Exception:
            pass
    
    # Final projection
    if best_state is not None:
        bx, by, br = best_state
        br = project_valid(bx, by, br)
        final_state = np.column_stack([bx, by, br])
    else:
        # Fallback: simple grid
        x, y, r = initial_lattice()
        r = project_valid(x, y, r)
        final_state = np.column_stack([x, y, r])
    
    return final_state
