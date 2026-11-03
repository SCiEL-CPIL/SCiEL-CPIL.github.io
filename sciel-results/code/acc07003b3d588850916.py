import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # We will use a penalty method with L-BFGS-B
    # Variables: x (N), y (N), r (N), W, H
    # Total 3N + 2 = 65 variables
    
    def penalized_objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        # Ensure positive radii and dimensions for calculation stability
        r_pos = np.maximum(r, 1e-6)
        W_pos = max(W, 1e-6)
        H_pos = max(H, 1e-6)
        
        # Objective: Maximize sum of radii -> Minimize -sum(r)
        obj = -np.sum(r_pos)
        
        penalty = 0.0
        lam = 100.0 # Penalty weight
        
        # Overlap penalty
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r_pos[i] + r_pos[j]
                if dist_sq < rad_sum**2:
                    # Quadratic penalty for overlap
                    penalty += (rad_sum**2 - dist_sq)
        
        # Boundary penalty
        for i in range(N):
            # Left/Right
            if x[i] < r_pos[i]:
                penalty += (r_pos[i] - x[i])**2
            if x[i] > W_pos - r_pos[i]:
                penalty += (x[i] - (W_pos - r_pos[i]))**2
            # Top/Bottom
            if y[i] < r_pos[i]:
                penalty += (r_pos[i] - y[i])**2
            if y[i] > H_pos - r_pos[i]:
                penalty += (y[i] - (H_pos - r_pos[i]))**2
        
        # Perimeter penalty
        perim = 2 * (W_pos + H_pos)
        if perim > 4.0:
             penalty += (perim - 4.0)**2
        # We also want to encourage W+H=2, but allow flexibility if penalty is high?
        # Actually, strict perimeter is required.
        
        return obj + lam * penalty

    def is_valid(x, y, r, W, H):
        if W <= 0 or H <= 0:
            return False
        if abs(2*(W+H) - 4) > 1e-5:
            return False
        for i in range(N):
            if r[i] <= 0:
                return False
            if x[i] - r[i] < -1e-5 or x[i] + r[i] > W + 1e-5:
                return False
            if y[i] - r[i] < -1e-5 or y[i] + r[i] > H + 1e-5:
                return False
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-5:
                    return False
        return True

    best_score = -np.inf
    best_v = None
    
    # Multi-start optimization
    # Try different aspect ratios and initial configurations
    num_restarts = 5
    
    for k in range(num_restarts):
        # Initial aspect ratio
        # Optimal aspect ratio for packing squares is 1, for circles might be different.
        # Let's try 1.0, 1.2, 0.8, 1.5, 0.7
        aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7]
        ratio = aspect_ratios[k % len(aspect_ratios)]
        
        # W + H = 2 => H = 2 - W
        # W / H = ratio => W = ratio * (2 - W) => W (1 + ratio) = 2 * ratio => W = 2 * ratio / (1 + ratio)
        W_init = 2.0 * ratio / (1.0 + ratio)
        H_init = 2.0 - W_init
        
        # Initialize circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        # Distribute N circles in side x side grid
        # We need to map N items to grid positions
        positions = []
        for i in range(N):
            gx = i % side
            gy = i // side
            positions.append((gx, gy))
        
        # Calculate initial radii and centers
        x_init = np.zeros(N)
        y_init = np.zeros(N)
        r_init = np.zeros(N)
        
        # Cell size
        cell_w = W_init / side
        cell_h = H_init / side
        
        for i in range(N):
            gx, gy = positions[i]
            x_init[i] = (gx + 0.5) * cell_w
            y_init[i] = (gy + 0.5) * cell_h
            # Initial radius is half the minimum cell dimension, minus a small buffer
            r_init[i] = 0.5 * min(cell_w, cell_h) - 1e-4
        
        v_init = np.concatenate([x_init, y_init, r_init, [W_init, H_init]])
        
        # Optimize
        res = minimize(penalized_objective, v_init, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-10})
        
        if res.success:
            v_opt = res.x
            x_opt = v_opt[:N]
            y_opt = v_opt[N:2*N]
            r_opt = v_opt[2*N:3*N]
            W_opt = v_opt[3*N]
            H_opt = v_opt[3*N+1]
            
            # Check validity
            if is_valid(x_opt, y_opt, r_opt, W_opt, H_opt):
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_v = v_opt.copy()
            else:
                # If invalid, try to repair or discard
                # We can try to scale down radii slightly to fix overlaps/bounds
                # But for now, we just discard if not strictly valid
                pass

    # If no valid solution found, fallback to a simple grid
    if best_v is None:
        # Simple 1x1 square grid
        W = 1.0
        H = 1.0
        side = 5 # 5x5 grid, 25 spots, use 21
        x = np.array([(i % side + 0.5) / side for i in range(N)])
        y = np.array([(i // side + 0.5) / side for i in range(N)])
        r = np.full(N, 0.5 / side - 0.01)
        best_v = np.concatenate([x, y, r, [W, H]])
        best_score = np.sum(r)

    # Final refinement with SLSQP on the best found solution to ensure strict constraints
    v_start = best_v.copy()
    
    def obj_slsqp(v):
        return -np.sum(v[2*N:3*N])
        
    def jac_slsqp(v):
        g = np.zeros_like(v)
        g[2*N:3*N] = -1.0
        return g
        
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
    
    # Only add constraints for the best solution to save time?
    # SLSQP is slow with many constraints.
    # Let's just return the L-BFGS-B result if it's valid.
    # The L-BFGS-B result might have tiny violations.
    # Let's do a quick check and scale if necessary.
    
    x_final = best_v[:N]
    y_final = best_v[N:2*N]
    r_final = best_v[2*N:3*N]
    W_final = best_v[3*N]
    H_final = best_v[3*N+1]
    
    # Ensure perimeter is exactly 4
    perim = 2 * (W_final + H_final)
    if abs(perim - 4.0) > 1e-8:
        scale = 4.0 / perim
        x_final *= scale
        y_final *= scale
        r_final *= scale
        W_final *= scale
        H_final *= scale
        
    # Check validity one last time
    if not is_valid(x_final, y_final, r_final, W_final, H_final):
        # Fallback to seed-like solution
        W_final = 1.0
        H_final = 1.0
        side = 5
        x_final = np.array([(i % side + 0.5) / side for i in range(N)])
        y_final = np.array([(i // side + 0.5) / side for i in range(N)])
        r_final = np.full(N, 0.5 / side - 0.01)

    result = np.zeros((N, 3))
    result[:, 0] = x_final
    result[:, 1] = y_final
    result[:, 2] = r_final
    
    return result
