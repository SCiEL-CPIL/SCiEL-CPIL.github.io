"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize as opt
import time

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_val = 0
    best_result = None
    
    # Try a few different aspect ratios (widths)
    # Perimeter 4 -> w + h = 2. h = 2 - w.
    # Square is w=1.0.
    # Let's try w in [0.8, 0.9, 1.0, 1.1, 1.2]
    widths = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initialize circles in a grid
        # 21 circles. 5x4 grid is 20. 6x4 is 24. 
        # Let's use a 5x5 grid and drop 4, or 4x5.
        # Better: use a hexagonal-like packing initialization if possible, or just a grid.
        # Grid initialization:
        rows = 4
        cols = 5
        # 20 circles in 4x5. Add 1 more? Or use 5x5 and remove 4 random?
        # Let's use a 5x5 grid, remove 4 corners or edges to make it 21?
        # Or just place 21 in a grid of 5x5, with some empty spots.
        # Simplest: 5 rows, 5 cols. 
        # Positions:
        xs = []
        ys = []
        rs = []
        
        # We want to maximize sum of radii. 
        # Initial radii: small, e.g., 0.05
        r_init = 0.05
        
        # Create a 5x5 grid
        for i in range(5):
            for j in range(5):
                x = (j + 0.5) * (w / 5.0)
                y = (i + 0.5) * (h / 5.0)
                xs.append(x)
                ys.append(y)
                rs.append(r_init)
        
        # We have 25 circles. We need 21.
        # Remove 4 circles. Let's remove the corners to allow larger radii in the middle?
        # Or just take the first 21?
        # Taking first 21 from row-major order:
        # Row 0: 5
        # Row 1: 5
        # Row 2: 5
        # Row 3: 5
        # Row 4: 1
        # This leaves the bottom right empty.
        # Let's try to be smarter: remove 4 circles that are most "constrained" or just random?
        # For now, let's just use the first 21.
        xs = xs[:21]
        ys = ys[:21]
        rs = rs[:21]
        
        x0 = np.array(xs + ys + rs + [w])
        
        # Define bounds
        # x_i in [r_i, w-r_i] -> but r_i is variable.
        # We can use loose bounds for x,y and enforce constraints via penalty or bounds on r.
        # Better: 
        # x_i in [0, w], y_i in [0, h], r_i in [0.01, min(w,h)/2]
        # But we must ensure circle is inside.
        # Let's use penalty for boundary violations too.
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, w)) # x
            bounds.append((0.0, h)) # y
            bounds.append((0.01, min(w, h)/2.0)) # r
        
        # w is fixed in this loop, so no bound needed for w if we don't optimize it.
        # Wait, the knowledge says "Joint aspect-ratio optimization".
        # If I fix w, I am not doing joint optimization.
        # Let's optimize w as well.
        # But w is coupled with h.
        # If I optimize w, I must update h in the objective.
        # This makes the objective non-smooth if I use h = 2-w directly in bounds? No, h is just a value.
        # But the bounds for x,y,r depend on w.
        # L-BFGS-B requires static bounds.
        # So I must either:
        # 1. Fix w and optimize x,y,r. (Simple, robust)
        # 2. Use a transformation for x,y,r that makes bounds independent of w?
        #    e.g., x = w * u, y = h * v, r = min(w,h) * z.
        #    This is a good idea.
        
        # Let's restart with the transformed variables approach.
        # Variables: u_i, v_i, z_i, w
        # x_i = w * u_i
        # y_i = (2-w) * v_i
        # r_i = min(w, 2-w) * z_i  -- This min() is non-smooth.
        # Instead, let's assume w is close to 1.0.
        # Or, just fix w to a few values. Given the time limit, fixing w to 5 values and optimizing x,y,r for each is very safe and fast.
        # The "joint optimization" improvement was small (1.136 -> 1.137).
        # Let's try fixing w first to get a strong baseline, then maybe add w optimization if time permits.
        
        # Let's stick to fixed w for now to ensure robustness.
        
        def objective(params):
            # params: x1, y1, r1, ..., x21, y21, r21
            xs = params[0:N]
            ys = params[N:2*N]
            rs = params[2*N:3*N]
            
            # Penalty for overlaps
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                    d = np.sqrt(d2)
                    min_d = rs[i] + rs[j]
                    if d < min_d:
                        penalty += (min_d - d)**2
            
            # Penalty for being outside
            # x < r, x > w-r, y < r, y > h-r
            out_penalty = 0.0
            for i in range(N):
                if xs[i] < rs[i]:
                    out_penalty += (rs[i] - xs[i])**2
                if xs[i] > w - rs[i]:
                    out_penalty += (xs[i] - (w - rs[i]))**2
                if ys[i] < rs[i]:
                    out_penalty += (rs[i] - ys[i])**2
                if ys[i] > h - rs[i]:
                    out_penalty += (ys[i] - (h - rs[i]))**2
            
            # Maximize sum of radii -> Minimize -sum(r) + lambda * (overlap_penalty + out_penalty)
            val = -np.sum(rs) + 100.0 * (penalty + out_penalty)
            return val
        
        # Initial guess
        x0 = np.array(xs + ys + rs)
        
        # Bounds
        b = []
        for i in range(N):
            b.append((0.0, w))
            b.append((0.0, h))
            b.append((0.01, min(w, h)/2.0))
            
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=b, options={'maxiter': 2000, 'ftol': 1e-12})
            
            if res.success:
                xs_opt = res.x[0:N]
                ys_opt = res.x[N:2*N]
                rs_opt = res.x[2*N:3*N]
                val = np.sum(rs_opt)
                
                # Check validity
                valid = True
                for i in range(N):
                    if xs_opt[i] < rs_opt[i] - 1e-6 or xs_opt[i] > w - rs_opt[i] + 1e-6:
                        valid = False
                    if ys_opt[i] < rs_opt[i] - 1e-6 or ys_opt[i] > h - rs_opt[i] + 1e-6:
                        valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((xs_opt[i]-xs_opt[j])**2 + (ys_opt[i]-ys_opt[j])**2)
                        if d < rs_opt[i] + rs_opt[j] - 1e-6:
                            valid = False
                
                if valid and val > best_val:
                    best_val = val
                    best_result = np.column_stack((xs_opt, ys_opt, rs_opt))
                    
        except Exception as e:
            pass
            
    # If no valid result found, fallback to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        
    return best_result

# EVOLVE-BLOCK-END
