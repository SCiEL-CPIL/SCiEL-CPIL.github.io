import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 140.0  # Safety margin

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
    
    # Check bounds
    for i in range(n):
        if r[i] < 1e-6:
            return None
        if cx[i] < r[i] - 1e-7 or cx[i] > W - r[i] + 1e-7:
            return None
        if cy[i] < r[i] - 1e-7 or cy[i] > H - r[i] + 1e-7:
            return None
            
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - 1e-7:
                return None
                
    return float(np.sum(r))

def _force_relax(x, W, H, max_iter=100, dt=0.005):
    """
    Force-directed relaxation to resolve overlaps.
    Modifies x in place or returns new x.
    """
    n = len(x)
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall forces
        for i in range(n):
            # Left
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 10
            # Right
            if cx[i] > W - r[i]:
                fx[i] -= (cx[i] - (W - r[i])) * 10
            # Bottom
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 10
            # Top
            if cy[i] > H - r[i]:
                fy[i] -= (cy[i] - (H - r[i])) * 10
        
        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-8:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 5
                    fy[i] += (dy / d) * overlap * 5
                    fx[j] -= (dx / d) * overlap * 5
                    fy[j] -= (dy / d) * overlap * 5
        
        cx += dt * fx
        cy += dt * fy
        
        # Clamp
        for i in range(n):
            cx[i] = np.clip(cx[i], 0, W)
            cy[i] = np.clip(cy[i], 0, H)
            # Ensure radius doesn't exceed wall distance
            max_r = min(cx[i], cy[i], W - cx[i], H - cy[i])
            if r[i] > max_r:
                r[i] = max_r
            
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
    
    # Multiple passes of growth
    for _ in range(50):
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
                growth = min(0.005, slack * 0.5)
                r[i] += growth
                improved = True
        
        if not improved:
            break
            
        # Relax to resolve overlaps
        x_temp = np.column_stack([cx, cy, r])
        x_temp = _force_relax(x_temp, W, H, max_iter=20, dt=0.002)
        cx = x_temp[:, 0]
        cy = x_temp[:, 1]
        r = x_temp[:, 2]
        
        current_val = np.sum(r)
        if current_val > best_val:
            best_val = current_val
        else:
            # If it got worse, revert? Or just break? 
            # Usually, if we can't grow, we stop.
            pass
            
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

def _generate_initial_guess(W, H):
    """
    Generate a good initial guess for SLSQP.
    """
    n = N
    # Try a few grid configurations
    best_x = None
    best_val = 0
    
    # Determine grid size
    # If W > H, more cols, else more rows
    aspect = W / H
    if aspect > 1.5:
        kx, ky = 7, 3
    elif aspect > 1.0:
        kx, ky = 5, 4
    elif aspect > 0.6:
        kx, ky = 4, 5
    else:
        kx, ky = 3, 7
        
    # Ensure kx*ky >= N
    while kx * ky < N:
        if W >= H:
            kx += 1
        else:
            ky += 1
            
    # Grid layout
    xs = np.linspace(W/(2*kx), W - W/(2*kx), kx)
    ys = np.linspace(H/(2*ky), H - H/(2*ky), ky)
    
    pts = []
    for y in ys:
        for x in xs:
            r = min(x, y, W-x, H-y) * 0.9
            pts.append([x, y, r])
            
    # Pad or truncate
    if len(pts) < N:
        while len(pts) < N:
            pts.append([W/2, H/2, 0.1])
    pts = pts[:N]
    x_grid = np.array(pts)
    
    # Relax it to fix overlaps
    x_relaxed = _force_relax(x_grid.copy(), W, H, max_iter=100, dt=0.005)
    
    val = _evaluate(x_relaxed, W, H)
    if val is not None and val > best_val:
        best_val = val
        best_x = x_relaxed
        
    # Also try a random start
    rng = np.random.default_rng(42)
    x_rand = np.column_stack([
        rng.uniform(0, W, N),
        rng.uniform(0, H, N),
        rng.uniform(0.05, 0.2, N)
    ])
    x_rand_relaxed = _force_relax(x_rand, W, H, max_iter=100, dt=0.005)
    val_rand = _evaluate(x_rand_relaxed, W, H)
    if val_rand is not None and val_rand > best_val:
        best_val = val_rand
        best_x = x_rand_relaxed
        
    return best_x, best_val

def construct():
    best_val = 0.0
    best_x = None
    best_W = 1.0
    best_H = 1.0
    
    # Time budget per aspect ratio
    # Total time ~ 140s.
    # Let's try 10 aspect ratios. 14s each.
    
    # Aspect ratios: W from 0.5 to 1.5
    # Since perimeter is 4, W+H=2.
    # Symmetric, so W in [0.5, 1.5] covers all.
    
    Ws = np.linspace(0.5, 1.5, 11)
    
    for W in Ws:
        if time.time() > DEADLINE - 10:
            break
            
        H = 2.0 - W
        if H < 0.1 or W < 0.1:
            continue
            
        # Generate initial guess
        x_init, init_val = _generate_initial_guess(W, H)
        if x_init is None:
            continue
            
        # Run SLSQP
        x_opt, opt_val = _run_slsqp(W, H, x_init)
        
        current_best_val = init_val
        current_best_x = x_init
        
        if x_opt is not None and opt_val > current_best_val:
            current_best_val = opt_val
            current_best_x = x_opt
            
        # Polish
        if time.time() < DEADLINE - 5:
            x_polished = _polish(current_best_x, W, H, time_budget=3.0)
            pol_val = _evaluate(x_polished, W, H)
            if pol_val is not None and pol_val > current_best_val:
                current_best_val = pol_val
                current_best_x = x_polished
                
        if current_best_val > best_val:
            best_val = current_best_val
            best_x = current_best_x
            best_W = W
            best_H = H
            
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
