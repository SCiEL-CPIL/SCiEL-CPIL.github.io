import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios for the rectangle
    # w + h = 2
    # We will sample w in a range, e.g., 0.5 to 1.5 (symmetric around 1.0)
    # But since the problem is symmetric w.r.t w and h (swapping x,y and w,h), 
    # we can just search w in [0.5, 1.5] and assume h = 2-w.
    # Actually, if w > 1, h < 1. The packing might be different.
    # Let's just try a grid of w values.
    
    aspect_ratios = np.linspace(0.6, 1.4, 9) # w values
    
    for w_init in aspect_ratios:
        # Ensure w is within (0, 2)
        if w_init <= 0 or w_init >= 2:
            continue
        h_init = 2.0 - w_init
        
        # Run multiple restarts
        num_restarts = 5
        
        for restart in range(num_restarts):
            # Generate initial layout
            positions = []
            radii = []
            
            if restart == 0:
                # Grid initialization
                aspect = w_init / h_init
                c = int(np.sqrt(N * aspect))
                if c < 1: c = 1
                r = int(np.ceil(N / c))
                
                cw = w_init / c
                ch = h_init / r
                
                for i in range(c):
                    for j in range(r):
                        if len(positions) >= N:
                            break
                        x = (i + 0.5) * cw
                        y = (j + 0.5) * ch
                        positions.append((x, y))
                        r_val = 0.49 * min(cw, ch)
                        radii.append(r_val)
            else:
                # Random initialization
                # Place circles randomly, ensure they are inside
                # Start with small radii to avoid initial overlaps
                base_r = 0.1
                for _ in range(N):
                    x = np.random.uniform(base_r, w_init - base_r)
                    y = np.random.uniform(base_r, h_init - base_r)
                    positions.append((x, y))
                    radii.append(base_r)
            
            # If we didn't get enough positions (shouldn't happen), fill
            while len(positions) < N:
                positions.append((w_init/2, h_init/2))
                radii.append(0.1)
            
            # Variables: [w, x1, y1, r1, x2, y2, r2, ...]
            x0 = [w_init]
            for i in range(N):
                x0.append(positions[i][0])
                x0.append(positions[i][1])
                x0.append(radii[i])
                
            x0 = np.array(x0)
            
            # Objective: minimize -sum(radii)
            def objective(x):
                w = x[0]
                h = 2.0 - w
                if w <= 0 or h <= 0:
                    return 1e6
                radii_sum = 0
                for i in range(N):
                    r = x[3 + 3*i]
                    if r < 0:
                        return 1e6
                    radii_sum += r
                return -radii_sum
                
            # Constraints: g(x) >= 0
            def constraints(x):
                w = x[0]
                h = 2.0 - w
                if w <= 0 or h <= 0:
                    n_pairs = N*(N-1)//2
                    return np.full(4*N + n_pairs, -1e6)
                
                cons = []
                # Wall constraints
                for i in range(N):
                    xi = x[1 + 3*i]
                    yi = x[2 + 3*i]
                    ri = x[3 + 3*i]
                    if ri < 0:
                        cons.extend([-1e6, -1e6, -1e6, -1e6])
                        continue
                    cons.append(xi - ri)       # left
                    cons.append(w - xi - ri)   # right
                    cons.append(yi - ri)       # bottom
                    cons.append(h - yi - ri)   # top
                    
                # Pair constraints
                for i in range(N):
                    xi = x[1 + 3*i]
                    yi = x[2 + 3*i]
                    ri = x[3 + 3*i]
                    for j in range(i+1, N):
                        xj = x[1 + 3*j]
                        yj = x[2 + 3*j]
                        rj = x[3 + 3*j]
                        dist_sq = (xi - xj)**2 + (yi - yj)**2
                        sum_r_sq = (ri + rj)**2
                        cons.append(dist_sq - sum_r_sq)
                        
                return np.array(cons)
                
            # Bounds
            bounds = [(1e-6, 2-1e-6)]
            for i in range(N):
                bounds.append((0, 2)) # x
                bounds.append((0, 2)) # y
                bounds.append((1e-6, 1.0)) # r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300, 'ftol': 1e-8})
                if res.success:
                    val = -res.fun
                    if val > best_value:
                        best_value = val
                        best_result = res.x
            except Exception:
                pass

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Extract result
    w = best_result[0]
    h = 2.0 - w
    circles = []
    for i in range(N):
        x = best_result[1 + 3*i]
        y = best_result[2 + 3*i]
        r = best_result[3 + 3*i]
        circles.append([x, y, r])
        
    return np.array(circles)
