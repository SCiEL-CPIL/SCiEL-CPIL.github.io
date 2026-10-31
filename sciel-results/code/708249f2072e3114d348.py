import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _objective(x):
    """
    Objective: minimize -sum(radii).
    x = [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    """
    radii = x[2::3]
    return -np.sum(radii)

def _make_constraints():
    """
    Generate constraint functions for SLSQP.
    Returns a list of dictionaries for scipy.optimize.minimize.
    """
    constraints = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(N):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i] - x[3*i+2]
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i+2]
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2]
        })
        
    # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
            })
            
    return constraints

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Try to fit a hexagonal pattern
    r = 0.1
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    # Fill remaining with random small circles
    rng = np.random.RandomState(42)
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        min_dist = min(x, 1 - x, y, 1 - y)
        for j in range(k):
            d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
            min_dist = min(min_dist, d - radii[j])
        if min_dist > 0.02:
            pos[k] = [x, y]
            radii[k] = min_dist * 0.95
            k += 1
        else:
            # If no space, place tiny circle
            pos[k] = [rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_corner_biased():
    """Corner-biased initialization."""
    rng = np.random.RandomState(7)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.18
    
    for i in range(4, N):
        # Place along edges or in center
        if rng.random() < 0.5:
            edge = rng.randint(4)
            t = rng.uniform(0.2, 0.8)
            if edge == 0:
                x, y = t, 0.1
            elif edge == 1:
                x, y = t, 0.9
            elif edge == 2:
                x, y = 0.1, t
            else:
                x, y = 0.9, t
        else:
            x, y = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
            
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
            
        if max_r < 0.01:
            max_r = 0.01
            
        pos[i] = [x, y]
        radii[i] = max_r * 0.95
        
    return pos, radii

def _initial_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.1, N)
    return pos, radii

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(100):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max_r * 0.9999
                valid = False
                
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale down both to fit
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] *= k
                        radii[j] *= k
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < (min_dist - tol)**2:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Pre-compute constraints for reuse if possible, but they depend on x indices which are fixed
    constraints = _make_constraints()
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    best_score = -np.inf
    best_x = None
    
    # Initializations
    inits = []
    pos_h, rad_h = _initial_hexagonal()
    inits.append(np.concatenate([pos_h.flatten(), rad_h]))
    
    pos_c, rad_c = _initial_corner_biased()
    inits.append(np.concatenate([pos_c.flatten(), rad_c]))
    
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append(np.concatenate([pos_r.flatten(), rad_r]))
        
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Run SLSQP
        try:
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': 300, 'ftol': 1e-10})
            
            if result.success or result.fun < best_score:
                x_opt = result.x
                pos_opt = x_opt[0::3].reshape(N, 2)
                rad_opt = x_opt[2::3]
                
                # Post-process to ensure validity
                pos_pp, rad_pp = _post_process(pos_opt, rad_opt)
                
                if _is_valid(pos_pp, rad_pp):
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
                        
        except Exception as e:
            pass
            
    # If we have a good solution, try to polish it with more iterations
    if best_x is not None and time.time() < deadline - 10:
        try:
            result = minimize(_objective, best_x, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': 500, 'ftol': 1e-12})
            if result.fun < _objective(best_x):
                x_opt = result.x
                pos_opt = x_opt[0::3].reshape(N, 2)
                rad_opt = x_opt[2::3]
                pos_pp, rad_pp = _post_process(pos_opt, rad_opt)
                if _is_valid(pos_pp, rad_pp):
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass
            
    if best_x is None:
        # Fallback
        pos, rad = _initial_hexagonal()
        pos, rad = _post_process(pos, rad)
        best_x = np.concatenate([pos.flatten(), rad])
        
    pos_final = best_x[0::3].reshape(N, 2)
    rad_final = best_x[2::3]
    
    return np.column_stack([pos_final, rad_final])
