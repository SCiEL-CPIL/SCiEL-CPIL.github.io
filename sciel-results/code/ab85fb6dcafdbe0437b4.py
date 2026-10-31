import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # Safety margin before 150s kill

def _evaluate(x, W, H):
    """
    Evaluate the sum of radii for a packing.
    x: (N, 3) array [cx, cy, r]
    Returns sum of radii if valid, else None.
    """
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    
    # Check bounds and positivity
    if np.any(r < 1e-6):
        return None
    if np.any(cx < r - 1e-7) or np.any(cx > W - r + 1e-7):
        return None
    if np.any(cy < r - 1e-7) or np.any(cy > H - r + 1e-7):
        return None
        
    # Check overlaps
    # Vectorized distance calculation
    cx_diff = cx[:, None] - cx[None, :]
    cy_diff = cy[:, None] - cy[None, :]
    dist = np.sqrt(cx_diff**2 + cy_diff**2)
    r_sum = r[:, None] + r[None, :]
    
    # Upper triangle check
    iu = np.triu_indices(n, 1)
    if np.any(dist[iu] < r_sum[iu] - 1e-7):
        return None
        
    return float(np.sum(r))

def _force_relax(x, W, H, max_iter=100, dt=0.01, grow_factor=0.0):
    """
    Force-directed relaxation to resolve overlaps.
    Modifies x in place or returns new x.
    If grow_factor > 0, radii are increased slightly each step if there is slack.
    """
    n = len(x)
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall forces (strong repulsion)
        for i in range(n):
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 100
            if cx[i] > W - r[i]:
                fx[i] -= (cx[i] - (W - r[i])) * 100
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 100
            if cy[i] > H - r[i]:
                fy[i] -= (cy[i] - (H - r[i])) * 100
        
        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-8:
                    overlap = min_d - d
                    # Force proportional to overlap
                    fx[i] += (dx / d) * overlap * 50
                    fy[i] += (dy / d) * overlap * 50
                    fx[j] -= (dx / d) * overlap * 50
                    fy[j] -= (dy / d) * overlap * 50
        
        cx += dt * fx
        cy += dt * fy
        
        # Clamp positions to valid range given current radii
        for i in range(n):
            cx[i] = np.clip(cx[i], r[i], W - r[i])
            cy[i] = np.clip(cy[i], r[i], H - r[i])
            
        # Optional: Grow radii if we are growing
        if grow_factor > 0:
             for i in range(n):
                 # Calculate local slack
                 slack = min(cx[i] - r[i], W - cx[i] - r[i], cy[i] - r[i], H - cy[i] - r[i])
                 for j in range(n):
                     if i == j: continue
                     dx = cx[i] - cx[j]
                     dy = cy[i] - cy[j]
                     d = np.sqrt(dx*dx + dy*dy)
                     slack = min(slack, d - r[i] - r[j])
                 
                 if slack > 1e-5:
                     r[i] += grow_factor * slack
                     # Re-clamp position if radius grew
                     cx[i] = np.clip(cx[i], r[i], W - r[i])
                     cy[i] = np.clip(cy[i], r[i], H - r[i])
            
    x[:, 0] = cx
    x[:, 1] = cy
    x[:, 2] = r
    return x

def _polish(x, W, H, time_budget=5.0):
    """
    Iteratively grow radii and relax positions.
    """
    n = len(x)
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    
    start = time.time()
    best_val = np.sum(r)
    best_state = (cx.copy(), cy.copy(), r.copy())
    
    # Multiple passes of growth
    for _ in range(100):
        if time.time() - start > time_budget:
            break
            
        improved = False
        # Try to grow each radius slightly
        for i in range(n):
            # Calculate slack
            slack = min(cx[i] - r[i], W - cx[i] - r[i], cy[i] - r[i], H - cy[i] - r[i])
            
            for j in range(n):
                if i == j: continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                slack = min(slack, d - r[i] - r[j])
                
            if slack > 1e-6:
                # Grow by a small amount
                growth = min(0.01, slack * 0.5)
                r[i] += growth
                improved = True
        
        if not improved:
            break
            
        # Relax to resolve overlaps
        x_temp = np.column_stack([cx, cy, r])
        x_temp = _force_relax(x_temp, W, H, max_iter=20, dt=0.005)
        cx = x_temp[:, 0]
        cy = x_temp[:, 1]
        r = x_temp[:, 2]
        
        current_val = np.sum(r)
        if current_val > best_val:
            best_val = current_val
            best_state = (cx.copy(), cy.copy(), r.copy())
            
    cx, cy, r = best_state
    return np.column_stack([cx, cy, r])

def _run_slsqp(W, H, initial_x):
    """
    Run SLSQP for a fixed W, H.
    initial_x: (N, 3) array
    """
    n = N
    # Flatten x: [x1, y1, r1, ..., xN, yN, rN]
    x0 = initial_x.flatten()
    
    def objective(z):
        r = z[2::3][:n]
        return -np.sum(r)
        
    def gradient(z):
        # Gradient of -sum(r) w.r.t z
        grad = np.zeros_like(z)
        grad[2::3][:n] = -1.0
        return grad
        
    # Constraints: d_ij - r_i - r_j >= 0
    # Wall constraints: x_i - r_i >= 0, etc.
    
    def make_constraints():
        cons = []
        # Walls
        for i in range(n):
            # Left
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[3*i] - z[3*i+2],
                'jac': lambda z, i=i: np.array([1.0 if k==3*i else -1.0 if k==3*i+2 else 0.0 for k in range(3*n)])
            })
            # Right
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: W - z[3*i] - z[3*i+2],
                'jac': lambda z, i=i: np.array([-1.0 if k==3*i else -1.0 if k==3*i+2 else 0.0 for k in range(3*n)])
            })
            # Bottom
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[3*i+1] - z[3*i+2],
                'jac': lambda z, i=i: np.array([1.0 if k==3*i+1 else -1.0 if k==3*i+2 else 0.0 for k in range(3*n)])
            })
            # Top
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: H - z[3*i+1] - z[3*i+2],
                'jac': lambda z, i=i: np.array([-1.0 if k==3*i+1 else -1.0 if k==3*i+2 else 0.0 for k in range(3*n)])
            })
            
        # Overlaps
        for i in range(n):
            for j in range(i+1, n):
                def fun(z, i=i, j=j):
                    xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
                    xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy + 1e-12)
                    return d - ri - rj
                    
                def jac(z, i=i, j=j):
                    xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
                    xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy + 1e-12)
                    j_vec = np.zeros(3*n)
                    j_vec[3*i] = dx/d
                    j_vec[3*i+1] = dy/d
                    j_vec[3*i+2] = -1.0
                    j_vec[3*j] = -dx/d
                    j_vec[3*j+1] = -dy/d
                    j_vec[3*j+2] = -1.0
                    return j_vec
                    
                cons.append({'type': 'ineq', 'fun': fun, 'jac': jac})
                
        return cons
        
    try:
        constraints = make_constraints()
        bounds = [(0, W), (0, H), (1e-6, min(W, H)/2)] * n
        
        res = minimize(
            objective, 
            x0, 
            jac=gradient,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-8, 'maxls': 20}
        )
        
        if res.success:
            x_res = res.x.reshape(n, 3)
            # Validate
            val = _evaluate(x_res, W, H)
            if val is not None:
                return x_res, val
    except Exception:
        pass
        
    return None, 0.0

def _run_slsqp_unified(initial_z):
    """
    Run SLSQP with unified parameters: [x1, y1, r1, ..., xN, yN, rN, W]
    H = 2 - W
    """
    n = N
    z0 = initial_z.copy()
    
    def objective(z):
        r = z[2::3][:n]
        return -np.sum(r)
        
    def gradient(z):
        grad = np.zeros_like(z)
        grad[2::3][:n] = -1.0
        return grad
        
    def constraints():
        cons = []
        # Walls
        for i in range(n):
            # Left: x_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[3*i] - z[3*i+2],
                'jac': lambda z, i=i: np.array([1.0 if k==3*i else -1.0 if k==3*i+2 else 0.0 for k in range(3*n+1)])
            })
            # Right: W - x_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[-1] - z[3*i] - z[3*i+2],
                'jac': lambda z, i=i: np.array([-1.0 if k==3*i else -1.0 if k==3*i+2 else 1.0 if k==3*n else 0.0 for k in range(3*n+1)])
            })
            # Bottom: y_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[3*i+1] - z[3*i+2],
                'jac': lambda z, i=i: np.array([1.0 if k==3*i+1 else -1.0 if k==3*i+2 else 0.0 for k in range(3*n+1)])
            })
            # Top: H - y_i - r_i >= 0  => (2 - W) - y_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: 2.0 - z[-1] - z[3*i+1] - z[3*i+2],
                'jac': lambda z, i=i: np.array([-1.0 if k==3*i+1 else -1.0 if k==3*i+2 else -1.0 if k==3*n else 0.0 for k in range(3*n+1)])
            })
            
        # Overlaps
        for i in range(n):
            for j in range(i+1, n):
                def fun(z, i=i, j=j):
                    xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
                    xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy + 1e-12)
                    return d - ri - rj
                    
                def jac(z, i=i, j=j):
                    xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
                    xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy + 1e-12)
                    j_vec = np.zeros(3*n+1)
                    j_vec[3*i] = dx/d
                    j_vec[3*i+1] = dy/d
                    j_vec[3*i+2] = -1.0
                    j_vec[3*j] = -dx/d
                    j_vec[3*j+1] = -dy/d
                    j_vec[3*j+2] = -1.0
                    return j_vec
                    
                cons.append({'type': 'ineq', 'fun': fun, 'jac': jac})
                
        # Aspect ratio bounds: 0.5 <= W <= 1.5
        cons.append({'type': 'ineq', 'fun': lambda z: z[-1] - 0.5, 'jac': lambda z: np.array([0.0]*(3*n) + [1.0])})
        cons.append({'type': 'ineq', 'fun': lambda z: 1.5 - z[-1], 'jac': lambda z: np.array([0.0]*(3*n) + [-1.0])})
        
        return cons

    try:
        cons = constraints()
        bounds = [(0, 1.5), (0, 1.5), (1e-6, 0.75)] * n
        bounds.append((0.5, 1.5))
        
        res = minimize(
            objective, 
            z0, 
            jac=gradient,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-8, 'maxls': 20}
        )
        
        if res.success:
            z_res = res.x
            W = z_res[-1]
            H = 2.0 - W
            x_res = z_res[:3*n].reshape(n, 3)
            
            # Validate
            val = _evaluate(x_res, W, H)
            if val is not None:
                return x_res, W, H, val
    except Exception:
        pass
        
    return None, 1.0, 1.0, 0.0

def _generate_initial_guesses():
    """
    Generate diverse initial guesses in unified space.
    """
    guesses = []
    n = N
    
    # Grid-based starts with varying aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8]
    side = int(np.ceil(np.sqrt(N)))
    
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        z = []
        # Arrange in a grid
        for i in range(N):
            row, col = divmod(i, side)
            # Add some jitter to avoid symmetry issues
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            # Estimate radius
            r0 = min(w0 / side, h0 / side) * 0.45
            z.extend([cx, cy, r0])
        z.append(w0)
        
        # Relax slightly to ensure feasibility
        z_arr = np.array(z)
        W = z_arr[-1]
        H = 2.0 - W
        x_part = z_arr[:3*n].reshape(n, 3)
        x_relaxed = _force_relax(x_part, W, H, max_iter=50, dt=0.01)
        
        z_final = np.concatenate([x_relaxed.flatten(), [W]])
        guesses.append(z_final)
        
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        w0 = rng.uniform(0.7, 1.3)
        h0 = 2.0 - w0
        z = []
        for i in range(N):
            cx = rng.uniform(0.1*w0, 0.9*w0)
            cy = rng.uniform(0.1*h0, 0.9*h0)
            r0 = rng.uniform(0.05, 0.15)
            z.extend([cx, cy, r0])
        z.append(w0)
        
        z_arr = np.array(z)
        W = z_arr[-1]
        H = 2.0 - W
        x_part = z_arr[:3*n].reshape(n, 3)
        x_relaxed = _force_relax(x_part, W, H, max_iter=50, dt=0.01)
        
        z_final = np.concatenate([x_relaxed.flatten(), [W]])
        guesses.append(z_final)
        
    return guesses

def construct():
    best_val = 0.0
    best_x = None
    best_W = 1.0
    best_H = 1.0
    
    guesses = _generate_initial_guesses()
    
    for z_guess in guesses:
        if time.time() > DEADLINE - 10:
            break
            
        # Run SLSQP with unified parameters
        x_opt, W_opt, H_opt, opt_val = _run_slsqp_unified(z_guess)
        
        current_best_val = opt_val
        current_best_x = x_opt
        current_W = W_opt
        current_H = H_opt
        
        # If SLSQP failed, try to use the relaxed guess directly
        if x_opt is None:
            W_guess = z_guess[-1]
            H_guess = 2.0 - W_guess
            x_guess = z_guess[:3*N].reshape(N, 3)
            
            # Relax more
            x_relaxed = _force_relax(x_guess, W_guess, H_guess, max_iter=100, dt=0.01)
            val = _evaluate(x_relaxed, W_guess, H_guess)
            if val is not None and val > current_best_val:
                current_best_val = val
                current_best_x = x_relaxed
                current_W = W_guess
                current_H = H_guess
        
        # Polish
        if current_best_x is not None and time.time() < DEADLINE - 5:
            x_polished = _polish(current_best_x, current_W, current_H, time_budget=5.0)
            pol_val = _evaluate(x_polished, current_W, current_H)
            if pol_val is not None and pol_val > current_best_val:
                current_best_val = pol_val
                current_best_x = x_polished
                
        if current_best_val > best_val:
            best_val = current_best_val
            best_x = current_best_x
            best_W = current_W
            best_H = current_H
            
    # Final check
    if best_x is None:
        # Fallback: simple grid
        W, H = 1.0, 1.0
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Ensure final validity
    final_val = _evaluate(best_x, best_W, best_H)
    if final_val is None:
        # Fallback
        W, H = 1.0, 1.0
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_x
