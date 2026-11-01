"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _feasibility_penalty(x, n):
    """
    x: (2n, 2n) array of [x1,y1,r1, x2,y2,r2, ...]
    Returns (violation_sum, perimeter)
    """
    n = n
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Perimeter constraint: 2*(W+H) <= 4 => W+H <= 2
    # We'll fix W and H externally; here we just check bounds
    # Actually, let's make W and H variables too? No, let's fix them for simplicity in this version.
    # Wait, the rectangle dimensions are part of the optimization.
    # Let's use a different representation: x = [x1..xn, y1..yn, r1..rn, W, H]
    # Total length: 3n + 2
    
    pass # This helper is replaced by the main optimization logic below


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Variables: x_coords (n), y_coords (n), radii (n), W, H
    # Total 3n + 2 variables
    # Constraints:
    # 1. 0 <= xi <= W
    # 2. 0 <= yi <= H
    # 3. ri >= 0
    # 4. 2*(W+H) <= 4  => W+H <= 2
    # 5. Distance between centers >= ri + rj
    
    n = N
    dim = 3 * n + 2
    
    def unpack(v):
        xs = v[:n]
        ys = v[n:2*n]
        rs = v[2*n:3*n]
        W = v[3*n]
        H = v[3*n+1]
        return xs, ys, rs, W, H

    def objective(v):
        xs, ys, rs, W, H = unpack(v)
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(rs)

    def constraints(v):
        xs, ys, rs, W, H = unpack(v)
        cons = []
        
        # Perimeter constraint: W + H <= 2
        cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - (v[3*n] + v[3*n+1])})
        
        # Bounds are handled by bounds parameter, but let's add explicit inequalities for clarity if needed
        # Actually, SLSQP handles bounds.
        
        # Circle non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        # Rearranged for ineq: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dr = rs[i] + rs[j]
                # This is non-convex.
                # We can add these as penalty or constraints.
                # Using constraints directly might be slow for 21 circles (210 pairs).
                # Let's use a penalty approach in the objective function instead of hard constraints for non-overlap
                # to allow SLSQP to navigate, or use a smooth approximation.
                
                # Better: Use a penalty in the objective function for violations.
                # But SLSQP works best with explicit constraints.
                # Let's try explicit constraints first. If too slow, switch to penalty.
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: (v[i]-v[i+n])**2 + (v[i+n]-v[j+n])**2 - (v[i+2*n]+v[j+2*n])**2})
        
        return cons

    # Initial guess: Grid layout
    # Let's start with a 5x5 grid in a 2x1 rectangle (perimeter 6? No, 2*(2+1)=6 > 4).
    # Perimeter 4 means if square, side=1. Area=1.
    # If rectangle 2x0? No.
    # Let's start with W=1.2, H=0.8 (Sum=2.0, Perim=4.0)
    W0 = 1.2
    H0 = 0.8
    
    # Place circles in a grid
    rows = 5
    cols = 5
    x0 = np.zeros(n)
    y0 = np.zeros(n)
    r0 = np.zeros(n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x0[i] = (col + 0.5) * W0 / cols
        y0[i] = (row + 0.5) * H0 / rows
        r0[i] = min(W0/cols, H0/rows) * 0.4  # Small initial radius
    
    x0_full = np.concatenate([x0, y0, r0, [W0, H0]])
    
    bounds = []
    for i in range(n):
        bounds.append((0, 2.0)) # x
    for i in range(n):
        bounds.append((0, 2.0)) # y
    for i in range(n):
        bounds.append((0, 1.0)) # r
    bounds.append((0.5, 2.0)) # W
    bounds.append((0.5, 2.0)) # H
    
    # SLSQP can be slow with many constraints.
    # Let's try a simpler approach: Fixed W, H and optimize positions and radii.
    # Then optimize W, H separately?
    
    # Alternative: Use a penalty method with fewer constraints.
    # Objective: -sum(r) + lambda * sum(violations)
    
    def penalty_objective(v, lam):
        xs, ys, rs, W, H = unpack(v)
        
        # Perimeter violation
        perim_viol = max(0, (W + H) - 2.0)
        
        # Boundary violations
        bound_viol = 0.0
        for i in range(n):
            bound_viol += max(0, -xs[i]) + max(0, xs[i] - W)
            bound_viol += max(0, -ys[i]) + max(0, ys[i] - H)
            bound_viol += max(0, -rs[i])
        
        # Overlap violations
        overlap_viol = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                min_dist = rs[i] + rs[j]
                if min_dist > 0:
                    dist = np.sqrt(dist_sq)
                    if dist < min_dist:
                        overlap_viol += (min_dist - dist)**2
        
        return -np.sum(rs) + lam * (perim_viol**2 + bound_viol**2 + overlap_viol**2)

    # Start with small lambda, increase it
    lam = 1.0
    x_best = x0_full.copy()
    val_best = -np.sum(x0_full[2*n:3*n])
    
    # Multistart with different initial W, H ratios
    best_val = -np.inf
    best_x = None
    
    ratios = [(1.0, 1.0), (1.2, 0.8), (0.8, 1.2), (1.5, 0.5), (0.5, 1.5), (2.0, 0.01)] # W, H such that W+H<=2
    
    for W_init, H_init in ratios:
        if W_init + H_init > 2.0:
            continue
            
        # Initialize positions
        x0_t = np.zeros(n)
        y0_t = np.zeros(n)
        r0_t = np.zeros(n)
        
        # Simple grid
        for i in range(n):
            row = i // 5
            col = i % 5
            x0_t[i] = (col + 0.5) * W_init / 5.0
            y0_t[i] = (row + 0.5) * H_init / 5.0
            r0_t[i] = 0.05
        
        x0_t_full = np.concatenate([x0_t, y0_t, r0_t, [W_init, H_init]])
        
        # Optimization
        for _ in range(5): # 5 iterations of increasing penalty
            res = minimize(
                penalty_objective, 
                x0_t_full, 
                args=(lam,), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            x0_t_full = res.x
            lam *= 10.0
            
            if time.time() > deadline:
                break
                
        # Evaluate
        xs, ys, rs, W, H = unpack(x0_t_full)
        # Check validity
        valid = True
        if W + H > 2.0 + 1e-6: valid = False
        for i in range(n):
            if xs[i] < -1e-6 or xs[i] > W + 1e-6 or ys[i] < -1e-6 or ys[i] > H + 1e-6 or rs[i] < 0:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if dist < rs[i] + rs[j] - 1e-6:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            val = np.sum(rs)
            if val > best_val:
                best_val = val
                best_x = x0_t_full.copy()
        
        if time.time() > deadline:
            break
    
    if best_x is None:
        # Fallback to a simple valid solution
        W, H = 1.0, 1.0
        xs = np.array([(i % 5 + 0.5) * 0.2 for i in range(n)])
        ys = np.array([(i // 5 + 0.5) * 0.2 for i in range(n)])
        rs = np.full(n, 0.09)
        best_x = np.concatenate([xs, ys, rs, [W, H]])
        best_val = np.sum(rs)
    
    xs, ys, rs, W, H = unpack(best_x)
    
    # Final greedy expansion
    # Try to increase each radius slightly
    for _ in range(10):
        for i in range(n):
            # Binary search for max r_i
            lo = rs[i]
            hi = rs[i] * 1.5 + 0.1
            for _ in range(20):
                mid = (lo + hi) / 2
                rs_new = rs.copy()
                rs_new[i] = mid
                # Check overlap
                ok = True
                for j in range(n):
                    if i == j: continue
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if dist < mid + rs[j] - 1e-7:
                        ok = False
                        break
                if ok:
                    lo = mid
                else:
                    hi = mid
            rs[i] = lo
    
    # Construct output array
    result = np.zeros((n, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = rs
    
    return result

# EVOLVE-BLOCK-END
