# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
import time

def squares_in_square10() -> tuple:
    """
    Solve the "10 squares in a unit square" packing problem.
    Primary: MILP via scipy.optimize.milp.
    Fallback/Refinement: Local search (Nelder-Mead) on a penalty function.
    """
    n = 10
    
    # --- Strategy 1: MILP ---
    try:
        # Variable ordering: x_0..x_9, y_0..y_9, s_0..s_9, then binaries
        npairs = n * (n - 1) // 2
        nbin = 4 * npairs
        nvars = 3 * n + nbin
        
        c = np.zeros(nvars)
        for i in range(n):
            c[2 * n + i] = -1.0
        
        lb = np.zeros(nvars)
        ub = np.ones(nvars)
        lb[2*n:3*n] = 1e-9
        ub[2*n:3*n] = 1.0
        lb[3*n:] = 0.0
        ub[3*n:] = 1.0
        
        integrality = np.zeros(nvars, dtype=int)
        integrality[3*n:] = 1
        
        A_ub_list = []
        b_ub_list = []
        
        M = 2.0
        
        def add_ineq(A_row, b_val):
            A_ub_list.append(A_row)
            b_ub_list.append(b_val)
            
        for i in range(n):
            row = np.zeros(nvars)
            row[i] = -1.0
            row[2*n + i] = 0.5
            add_ineq(row, 0.0)
            
            row = np.zeros(nvars)
            row[i] = 1.0
            row[2*n + i] = -0.5
            add_ineq(row, 1.0)
            
            row = np.zeros(nvars)
            row[n + i] = -1.0
            row[2*n + i] = 0.5
            add_ineq(row, 0.0)
            
            row = np.zeros(nvars)
            row[n + i] = 1.0
            row[2*n + i] = -0.5
            add_ineq(row, 1.0)
            
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                a_idx = 3 * n + pair_idx
                b_idx = 3 * n + pair_idx + 1
                c_idx = 3 * n + pair_idx + 2
                d_idx = 3 * n + pair_idx + 3
                
                row = np.zeros(nvars)
                row[a_idx] = 1.0
                row[b_idx] = 1.0
                row[c_idx] = 1.0
                row[d_idx] = 1.0
                add_ineq(-row, -1.0)
                
                row = np.zeros(nvars)
                row[a_idx] = 1.0
                row[b_idx] = 1.0
                add_ineq(row, 1.0)
                
                row = np.zeros(nvars)
                row[c_idx] = 1.0
                row[d_idx] = 1.0
                add_ineq(row, 1.0)
                
                # Separation
                row = np.zeros(nvars)
                row[i] = 1.0
                row[j] = -1.0
                row[2*n + i] = 0.5
                row[2*n + j] = 0.5
                row[a_idx] = -M
                add_ineq(row, 0.0)
                
                row = np.zeros(nvars)
                row[j] = 1.0
                row[i] = -1.0
                row[2*n + j] = 0.5
                row[2*n + i] = 0.5
                row[b_idx] = -M
                add_ineq(row, 0.0)
                
                row = np.zeros(nvars)
                row[n + i] = 1.0
                row[n + j] = -1.0
                row[2*n + i] = 0.5
                row[2*n + j] = 0.5
                row[c_idx] = -M
                add_ineq(row, 0.0)
                
                row = np.zeros(nvars)
                row[n + j] = 1.0
                row[n + i] = -1.0
                row[2*n + j] = 0.5
                row[2*n + i] = 0.5
                row[d_idx] = -M
                add_ineq(row, 0.0)
                
                pair_idx += 1
        
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
        bounds = Bounds(lb, ub)
        
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            x = result.x
            centers = np.zeros((n, 2), dtype=np.float64)
            sides = np.zeros(n, dtype=np.float64)
            for i in range(n):
                centers[i, 0] = x[i]
                centers[i, 1] = x[n + i]
                sides[i] = x[2*n + i]
            
            # Verify and clamp
            sides = np.maximum(sides, 1e-9)
            
            # Check validity
            valid = True
            tol = 1e-7
            for i in range(n):
                if centers[i,0] - sides[i]/2 < -tol or centers[i,0] + sides[i]/2 > 1+tol:
                    valid = False
                    break
                if centers[i,1] - sides[i]/2 < -tol or centers[i,1] + sides[i]/2 > 1+tol:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = abs(centers[i,0] - centers[j,0])
                        dy = abs(centers[i,1] - centers[j,1])
                        t = (sides[i] + sides[j])/2 - tol
                        if dx < t and dy < t:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                return centers, sides
            
    except Exception:
        pass
    
    # --- Strategy 2: Local Search Fallback ---
    # Start with a grid
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    
    # Define penalty function
    def penalty(params):
        # params: [cx_0, cy_0, s_0, ..., cx_9, cy_9, s_9]
        pen = 0.0
        c = params[:2*n].reshape(n, 2)
        s = params[2*n:]
        
        # Bounds penalty
        for i in range(n):
            if c[i,0] - s[i]/2 < 0: pen += (0 - (c[i,0] - s[i]/2))**2
            if c[i,0] + s[i]/2 > 1: pen += ((c[i,0] + s[i]/2) - 1)**2
            if c[i,1] - s[i]/2 < 0: pen += (0 - (c[i,1] - s[i]/2))**2
            if c[i,1] + s[i]/2 > 1: pen += ((c[i,1] + s[i]/2) - 1)**2
            if s[i] < 0: pen += s[i]**2
            
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(c[i,0] - c[j,0])
                dy = abs(c[i,1] - c[j,1])
                t = (s[i] + s[j])/2
                if dx < t and dy < t:
                    # Overlap amount
                    ox = t - dx
                    oy = t - dy
                    pen += ox**2 + oy**2
        
        # Objective: Maximize sum(s) -> Minimize -sum(s) + penalty
        return -np.sum(s) + 1000 * pen

    # Initial params
    init_params = np.concatenate([centers.flatten(), sides])
    
    # Optimize
    res = minimize(penalty, init_params, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if res.success:
        best_params = res.x
        centers = best_params[:2*n].reshape(n, 2)
        sides = best_params[2*n:]
        
        # Ensure sides > 0
        sides = np.maximum(sides, 1e-9)
        
        # Final validity check
        tol = 1e-7
        valid = True
        for i in range(n):
            if centers[i,0] - sides[i]/2 < -tol or centers[i,0] + sides[i]/2 > 1+tol:
                valid = False
            if centers[i,1] - sides[i]/2 < -tol or centers[i,1] + sides[i]/2 > 1+tol:
                valid = False
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    t = (sides[i] + sides[j])/2 - tol
                    if dx < t and dy < t:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            return centers, sides

    # Return original grid if all else fails
    return centers, sides

# EVOLVE-BLOCK-END
