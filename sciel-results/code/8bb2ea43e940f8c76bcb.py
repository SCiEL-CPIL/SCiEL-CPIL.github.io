import numpy as np
from scipy.optimize import minimize
import random

N = 21

def construct():
    # Strategy:
    # 1. Sequential Autoregressive Construction to get a good initial packing.
    # 2. Global Refinement using SLSQP to maximize sum of radii with perimeter constraint.
    
    # --- Phase 1: Sequential Construction ---
    # We will try a few different random orders/starts and pick the best one.
    best_sum_r = 0
    best_solution = None
    
    # Fixed W, H for construction phase to simplify constraints
    # Optimal aspect ratio for packing is often close to 1, but let's start with 1x1
    # Perimeter 4 => W+H=2 => 1x1 is valid.
    W_const = 1.0
    H_const = 1.0
    
    for attempt in range(10):
        # Randomize initial centers slightly to break symmetry
        random.seed(attempt)
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Order of insertion: Random permutation? Or fixed?
        # Fixed order with random starting positions is usually robust.
        # Let's use a random order of indices to assign to positions 0..N-1
        # Actually, the "autoregressive" nature implies we place circle 0, then 1, etc.
        # The "content" of circle i doesn't matter, just the set.
        
        for i in range(N):
            # Optimize position (x_i, y_i) and radius r_i
            # Objective: Maximize r_i
            # Constraints:
            # 1. x_i - r_i >= 0
            # 2. x_i + r_i <= W_const
            # 3. y_i - r_i >= 0
            # 4. y_i + r_i <= H_const
            # 5. dist(i, j) >= r_i + r_j for all j < i
            
            # Initial guess for circle i
            # Place it somewhere random inside
            x0 = random.uniform(0.1, W_const - 0.1)
            y0 = random.uniform(0.1, H_const - 0.1)
            r0 = 0.05 # Small initial radius
            
            v0 = np.array([x0, y0, r0])
            
            def obj_sequential(v, i_idx=i):
                x_i, y_i, r_i = v
                # Penalize invalid states
                penalty = 0
                if r_i < 0:
                    penalty += r_i**2
                # Boundary
                if x_i - r_i < 0: penalty += (x_i - r_i)**2
                if x_i + r_i > W_const: penalty += (x_i + r_i - W_const)**2
                if y_i - r_i < 0: penalty += (y_i - r_i)**2
                if y_i + r_i > H_const: penalty += (y_i + r_i - H_const)**2
                
                # Overlap with previous circles
                for j in range(i_idx):
                    dx = x_i - x[j]
                    dy = y_i - y[j]
                    dist_sq = dx*dx + dy*dy
                    rad_sum = r_i + r[j]
                    if dist_sq < rad_sum**2:
                        penalty += (rad_sum**2 - dist_sq)
                        
                return -r_i + 100 * penalty
            
            # Constraints for SLSQP
            cons = []
            # Boundary
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - v[2]}) # x >= r
            cons.append({'type': 'ineq', 'fun': lambda v: W_const - v[0] - v[2]}) # x+r <= W
            cons.append({'type': 'ineq', 'fun': lambda v: v[1] - v[2]}) # y >= r
            cons.append({'type': 'ineq', 'fun': lambda v: H_const - v[1] - v[2]}) # y+r <= H
            
            # Overlap
            for j in range(i):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, j=j: (v[0]-x[j])**2 + (v[1]-y[j])**2 - (v[2]+r[j])**2
                })
                
            # r > 0
            cons.append({'type': 'ineq', 'fun': lambda v: v[2]})
            
            try:
                res = minimize(obj_sequential, v0, method='SLSQP', constraints=cons, 
                               options={'maxiter': 50, 'ftol': 1e-6})
                if res.success:
                    x[i] = res.x[0]
                    y[i] = res.x[1]
                    r[i] = res.x[2]
                else:
                    # Fallback: just place it small
                    x[i] = x0
                    y[i] = y0
                    r[i] = 0.01
            except Exception:
                x[i] = x0
                y[i] = y0
                r[i] = 0.01

        total_r = np.sum(r)
        if total_r > best_sum_r:
            best_sum_r = total_r
            best_solution = (x.copy(), y.copy(), r.copy())
            
    if best_solution is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
        y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
        r = np.full(N, 0.5 / side - 1e-4)
        W = 1.0
        H = 1.0
    else:
        x, y, r = best_solution
        W = 1.0
        H = 1.0

    # --- Phase 2: Global Refinement ---
    # Optimize x, y, r, W, H to maximize sum(r) subject to 2(W+H)=4
    
    x0 = np.concatenate([x, y, r, [W, H]])
    
    def obj_global(v):
        x_g = v[:N]
        y_g = v[N:2*N]
        r_g = v[2*N:3*N]
        W_g = v[3*N]
        H_g = v[3*N+1]
        
        penalty = 0
        # Perimeter
        perim = 2 * (W_g + H_g)
        penalty += (perim - 4)**2
        
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x_g[i]-x_g[j])**2 + (y_g[i]-y_g[j])**2
                rad_sum = r_g[i] + r_g[j]
                if dist_sq < rad_sum**2:
                    penalty += (rad_sum**2 - dist_sq)
                    
        # Bounds
        for i in range(N):
            if x_g[i] - r_g[i] < 0: penalty += (x_g[i] - r_g[i])**2
            if x_g[i] + r_g[i] > W_g: penalty += (x_g[i] + r_g[i] - W_g)**2
            if y_g[i] - r_g[i] < 0: penalty += (y_g[i] - r_g[i])**2
            if y_g[i] + r_g[i] > H_g: penalty += (y_g[i] + r_g[i] - H_g)**2
            
        # Negative radii
        for i in range(N):
            if r_g[i] < 0: penalty += r_g[i]**2
            
        return -np.sum(r_g) + 100 * penalty
        
    def jac_global(v):
        g = np.zeros_like(v)
        g[2*N:3*N] = -1.0
        return g
        
    constraints = []
    # Perimeter equality
    constraints.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
    
    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
            })
            
    # Bounds
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
        
    # Radii > 0
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
        
    # W, H > 0
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N]})
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N+1]})

    res = minimize(obj_global, x0, jac=jac_global, method='SLSQP', constraints=constraints, 
                   options={'maxiter': 200, 'ftol': 1e-8})
    
    if res.success:
        x_final = res.x[:N]
        y_final = res.x[N:2*N]
        r_final = res.x[2*N:3*N]
        W_final = res.x[3*N]
        H_final = res.x[3*N+1]
    else:
        x_final = x
        y_final = y
        r_final = r
        W_final = W
        H_final = H
        
    # Final validity check and scaling
    perim = 2 * (W_final + H_final)
    if perim > 4:
        scale = 4.0 / perim
        x_final *= scale
        y_final *= scale
        r_final *= scale
        W_final *= scale
        H_final *= scale
        
    # Check for overlaps and fix if necessary (shrink radii slightly)
    # This is a safety net
    for i in range(N):
        if r_final[i] <= 0:
            r_final[i] = 1e-6
    for i in range(N):
        for j in range(i+1, N):
            dist_sq = (x_final[i]-x_final[j])**2 + (y_final[i]-y_final[j])**2
            rad_sum = r_final[i] + r_final[j]
            if dist_sq < rad_sum**2:
                # Shrink both slightly
                excess = rad_sum - np.sqrt(dist_sq)
                shrink = excess / 2 + 1e-6
                r_final[i] -= shrink
                r_final[j] -= shrink
                if r_final[i] < 1e-6: r_final[i] = 1e-6
                if r_final[j] < 1e-6: r_final[j] = 1e-6

    result = np.zeros((N, 3))
    result[:, 0] = x_final
    result[:, 1] = y_final
    result[:, 2] = r_final
    
    return result
