"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as so

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_sum_r = -np.inf
    best_config = None
    
    # Helper to create constraints for SLSQP
    def make_constraints(W, H):
        constraints = []
        
        # Wall constraints: x_i >= r_i, x_i <= W - r_i, y_i >= r_i, y_i <= H - r_i
        # We can combine these or list them individually. 
        # SLSQP handles inequalities g(x) >= 0.
        # x_i - r_i >= 0
        # W - x_i - r_i >= 0
        # y_i - r_i >= 0
        # H - y_i - r_i >= 0
        
        def wall_x_min(v):
            x = v[:N]
            r = v[2*N:3*N]
            return x - r
        
        def wall_x_max(v):
            x = v[:N]
            r = v[2*N:3*N]
            return W - x - r
            
        def wall_y_min(v):
            y = v[N:2*N]
            r = v[2*N:3*N]
            return y - r
            
        def wall_y_max(v):
            y = v[N:2*N]
            r = v[2*N:3*N]
            return H - y - r
        
        constraints.append({'type': 'ineq', 'fun': wall_x_min})
        constraints.append({'type': 'ineq', 'fun': wall_x_max})
        constraints.append({'type': 'ineq', 'fun': wall_y_min})
        constraints.append({'type': 'ineq', 'fun': wall_y_max})
        
        # Circle-circle constraints: dist(i,j) - r_i - r_j >= 0
        # dist = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # To avoid sqrt in gradient, we can use squared distance? No, SLSQP needs differentiable.
        # sqrt is differentiable for d>0. We assume d>0.
        
        for i in range(N):
            for j in range(i+1, N):
                def circle_con(v, i=i, j=j):
                    xi = v[i]
                    xj = v[j]
                    yi = v[N+i]
                    yj = v[N+j]
                    ri = v[2*N+i]
                    rj = v[2*N+j]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-10:
                        return -100.0 # Large penalty if overlapping centers
                    return d - ri - rj
                constraints.append({'type': 'ineq', 'fun': circle_con})
                
        return constraints

    def objective(v):
        """
        Minimize -sum(r_i)
        """
        r = v[2*N:3*N]
        return -np.sum(r)

    def bounds():
        # x in [0, W], y in [0, H], r in [0, 1]
        # Actually r is bounded by W/2 and H/2, but [0,1] is safe since W,H <= 2
        b = []
        for i in range(N):
            b.append((0.0, 1.0)) # x
        for i in range(N):
            b.append((0.0, 1.0)) # y
        for i in range(N):
            b.append((0.0, 1.0)) # r
        return b

    def generate_grid_init(W, H, rows, cols):
        """
        Generate a grid initialization for N circles in W x H.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Distribute N circles into rows x cols grid
        # If rows*cols < N, some cells will have multiple? No, we just place N circles.
        # Let's just place them in a grid pattern, filling row by row.
        # If N > rows*cols, we might need to adjust. But for now, let's assume we pick rows/cols such that rows*cols >= N.
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * W / cols
                y[idx] = (i + 0.5) * H / rows
                # Initial radius estimate
                # Min distance to wall or neighbor
                r[idx] = 0.05
                idx += 1
        
        # If we didn't fill all N (should not happen if rows*cols >= N), fill remaining randomly
        if idx < N:
             for k in range(idx, N):
                 x[k] = np.random.uniform(0.1, W-0.1)
                 y[k] = np.random.uniform(0.1, H-0.1)
                 r[k] = 0.05
                 
        v = np.concatenate([x, y, r])
        return v

    # Main loop: Try different aspect ratios and topologies
    # Perimeter 4 => 2(W+H) = 4 => W+H = 2.
    # W in [0.5, 1.5] (symmetric, so just check W in [0.5, 1.0] and mirror if needed, or just optimize W)
    
    # Let's try a few aspect ratios
    aspect_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
    # Corresponding W values:
    # W/H = ar => W = ar*H => W + W/ar = 2 => W(1 + 1/ar) = 2 => W = 2 * ar / (ar + 1)
    
    # Also try different grid topologies (rows, cols)
    # N=21. 
    # 3x7, 4x6 (24 slots), 5x5 (25 slots), 7x3, 6x4, 5x5
    topologies = [
        (3, 7), (4, 6), (5, 5), (7, 3), (6, 4)
    ]
    
    np.random.seed(42)
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 - W
        
        # Check if W, H are reasonable
        if W <= 0.1 or H <= 0.1:
            continue
            
        # Generate initial configurations for each topology
        for rows, cols in topologies:
            if time.time() > deadline - 10:
                break
                
            # Only use topology if it fits
            if rows * cols < N:
                continue
                
            v0 = generate_grid_init(W, H, rows, cols)
            
            # Add small noise to break symmetry
            v0 += np.random.normal(0, 0.01, size=N*3)
            
            # Clip to bounds to ensure valid start
            v0[:N] = np.clip(v0[:N], 0.1, W-0.1)
            v0[N:2*N] = np.clip(v0[N:2*N], 0.1, H-0.1)
            v0[2*N:3*N] = np.clip(v0[2*N:3*N], 0.01, 1.0)
            
            constraints = make_constraints(W, H)
            bounds = bounds()
            
            # Run SLSQP
            # maxiter should be enough to converge but not take forever
            # 200 iterations * ~20ms per eval? 4s per run.
            # We have ~150s total. 5 ar * 5 topologies = 25 runs. 25 * 4s = 100s. OK.
            try:
                res = so.minimize(
                    objective, 
                    v0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-6}
                )
                
                if res.success:
                    v_opt = res.x
                    sum_r = -res.fun
                    
                    # Verify validity
                    x = v_opt[:N]
                    y = v_opt[N:2*N]
                    r = v_opt[2*N:3*N]
                    
                    valid = True
                    # Check walls
                    if np.any(x < r - 1e-5) or np.any(W - x < r - 1e-5):
                        valid = False
                    if np.any(y < r - 1e-5) or np.any(H - y < r - 1e-5):
                        valid = False
                        
                    # Check overlaps
                    for i in range(N):
                        if not valid: break
                        for j in range(i+1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            d = np.sqrt(dx*dx + dy*dy)
                            if d < r[i] + r[j] - 1e-5:
                                valid = False
                                break
                                
                    if valid and sum_r > best_sum_r:
                        best_sum_r = sum_r
                        config = np.zeros((N, 3))
                        config[:, 0] = x
                        config[:, 1] = y
                        config[:, 2] = r
                        best_config = config
                        
            except Exception as e:
                pass

    # Fallback
    if best_config is None:
        # Simple grid fallback
        W = 1.0
        H = 1.0
        x = np.linspace(0.2, 0.8, 5)
        y = np.linspace(0.2, 0.8, 5)
        xx, yy = np.meshgrid(x, y)
        xx = xx.flatten()
        yy = yy.flatten()
        
        config = np.zeros((N, 3))
        config[:N, 0] = xx[:N]
        config[:N, 1] = yy[:N]
        config[:N, 2] = 0.1
        best_config = config

    return best_config

# EVOLVE-BLOCK-END
