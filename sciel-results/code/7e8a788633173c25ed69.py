"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
TIME_LIMIT = 115  # safety margin below 120s

def construct():
    t0 = time.time()
    
    # Generate initial positions using a grid-like pattern with some perturbation
    # Use a 5x6 grid arrangement for 26 circles
    cols, rows = 6, 5
    positions = []
    for i in range(cols):
        for j in range(rows):
            x = (i + 0.5) / cols
            y = (j + 0.5) / rows
            positions.append([x, y])
    positions = np.array(positions[:N])
    
    # Initial radii: small uniform
    r0 = 0.08
    radii0 = np.full(N, r0)
    
    def make_vars(pos, rad):
        return np.concatenate([pos.ravel(), rad])
    
    def unpack_vars(v):
        pos = v[:2*N].reshape(N, 2)
        rad = v[2*N:]
        return pos, rad
    
    def objective(v):
        pos, rad = unpack_vars(v)
        # Maximize sum of radii -> minimize negative
        return -np.sum(rad)
    
    def constraints(v):
        pos, rad = unpack_vars(v)
        cons = []
        # Boundary constraints: circles must be inside [0,1]^2
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i, p=pos: p[i,0] - rad[i]})  # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i, p=pos: 1 - p[i,0] - rad[i]})  # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i, p=pos: p[i,1] - rad[i]})  # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i, p=pos: 1 - p[i,1] - rad[i]})  # 1 - y - r >= 0
        # Non-overlap: dist(i,j) >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 
                             'fun': lambda v, i=i, j=j, p=pos, r=rad: 
                             np.sum((p[i]-p[j])**2) - (r[i]+r[j])**2})
        return cons
    
    def constraints_fast(v):
        """Vectorized constraints for SLSQP."""
        pos, rad = unpack_vars(v)
        cons = []
        # Boundary
        cons.append(pos[:,0] - rad)  # x - r >= 0
        cons.append(1 - pos[:,0] - rad)  # 1 - x - r >= 0
        cons.append(pos[:,1] - rad)  # y - r >= 0
        cons.append(1 - pos[:,1] - rad)  # 1 - y - r >= 0
        # Non-overlap: for each pair (i,j), dist^2 - (ri+rj)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                d2 = np.sum((pos[i]-pos[j])**2)
                cons.append(np.array([d2 - (rad[i]+rad[j])**2]))
        return cons
    
    # Use a smoother approach: minimize with penalty for violations
    def penalized_objective(v, mu=1.0):
        pos, rad = unpack_vars(v)
        obj = -np.sum(rad)
        penalty = 0.0
        # Boundary penalties
        penalty += mu * np.sum(np.maximum(0, rad - pos[:,0])**2)
        penalty += mu * np.sum(np.maximum(0, rad - (1-pos[:,0]))**2)
        penalty += mu * np.sum(np.maximum(0, rad - pos[:,1])**2)
        penalty += mu * np.sum(np.maximum(0, rad - (1-pos[:,1]))**2)
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                d2 = np.sum((pos[i]-pos[j])**2)
                overlap = rad[i] + rad[j] - np.sqrt(max(d2, 1e-12))
                if overlap > 0:
                    penalty += mu * overlap**2
        return obj + penalty
    
    # Initial variable vector
    v0 = make_vars(positions, radii0)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 1.0))  # x
        bounds.append((0.01, 1.0))  # y
        bounds.append((0.01, 0.5))  # radius
    
    best_v = v0.copy()
    best_obj = penalized_objective(v0, mu=1.0)
    
    # Phase 1: SLSQP with increasing penalty
    mu_values = [1.0, 5.0, 20.0, 100.0, 500.0]
    
    for mu in mu_values:
        if time.time() - t0 > TIME_LIMIT - 10:
            break
        try:
            res = minimize(penalized_objective, best_v, args=(mu,), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 50, 'ftol': 1e-12})
            if res.success or res.fun < best_obj:
                best_v = res.x
                best_obj = res.fun
        except:
            pass
    
    # Phase 2: SLSQP with actual constraints for polishing
    if time.time() - t0 < TIME_LIMIT - 15:
        try:
            res = minimize(objective, best_v, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-14})
            if res.success:
                pos, rad = unpack_vars(res.x)
                # Verify validity
                valid = True
                for i in range(N):
                    if pos[i,0] < rad[i] or pos[i,0] > 1-rad[i]:
                        valid = False
                    if pos[i,1] < rad[i] or pos[i,1] > 1-rad[i]:
                        valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt(np.sum((pos[i]-pos[j])**2))
                        if d < rad[i] + rad[j] - 1e-9:
                            valid = False
                if valid and -res.fun > -best_obj:
                    best_v = res.x
                    best_obj = res.fun
        except:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() - t0 < TIME_LIMIT - 5:
        try:
            res = minimize(objective, best_v, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 100, 'ftol': 1e-16})
            if res.success:
                pos, rad = unpack_vars(res.x)
                valid = True
                for i in range(N):
                    if pos[i,0] < rad[i] or pos[i,0] > 1-rad[i]:
                        valid = False
                    if pos[i,1] < rad[i] or pos[i,1] > 1-rad[i]:
                        valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt(np.sum((pos[i]-pos[j])**2))
                        if d < rad[i] + rad[j] - 1e-9:
                            valid = False
                if valid:
                    best_v = res.x
        except:
            pass
    
    pos, rad = unpack_vars(best_v)
    
    # Final validation and clamping
    for i in range(N):
        # Clamp positions to be valid
        pos[i,0] = np.clip(pos[i,0], rad[i], 1-rad[i])
        pos[i,1] = np.clip(pos[i,1], rad[i], 1-rad[i])
    
    # Check for overlaps and fix
    for iteration in range(10):
        fixed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt(np.sum((pos[i]-pos[j])**2))
                min_d = rad[i] + rad[j]
                if d < min_d - 1e-10 and d > 1e-12:
                    # Push apart
                    overlap = min_d - d
                    direction = (pos[i] - pos[j]) / d
                    pos[i] += direction * overlap / 2
                    pos[j] -= direction * overlap / 2
                    fixed = True
        if not fixed:
            break
    
    # Re-clamp after fixes
    for i in range(N):
        pos[i,0] = np.clip(pos[i,0], rad[i], 1-rad[i])
        pos[i,1] = np.clip(pos[i,1], rad[i], 1-rad[i])
    
    result = np.column_stack([pos, rad])
    return result
