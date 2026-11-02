import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit, use 120s as deadline
    
    def pack_hexagonal(rows):
        """
        Generate a hexagonal packing configuration.
        Returns: w, h, r, xs, ys
        """
        pts = []
        # Assume unit radius for generation, then scale
        # Distance between centers in hex grid is 2r.
        # If we set r=1, dist=2.
        # Row height is sqrt(3) * r = sqrt(3).
        # Row width depends on number of circles and offset.
        
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            # Offset alternate rows by 1 (half of 2r)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        
        pts = np.array(pts)
        
        # Center the points
        center = pts.mean(axis=0)
        pts -= center
        
        # Compute bounding box of centers
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        # Add diameter (2r) to get bounding box of circles
        # Here r=1, so add 2
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        # We need to fit this into a rectangle of perimeter 4 (w+h=2)
        # Scale factor s such that s*w_total + s*h_total = 2? No.
        # The rectangle dimensions w, h must satisfy w+h=2.
        # The circles must fit inside.
        # If we scale the entire configuration by s, the bounding box becomes s*w_total x s*h_total.
        # The radius becomes s*1.
        # We need s*w_total <= w and s*h_total <= h.
        # To maximize r (=s), we want to minimize the "cost" of the bounding box.
        # Given w+h=2, the best aspect ratio for a given bounding box W x H is when w/W = h/H = s.
        # Then w = sW, h = sH.
        # sW + sH = 2 => s(W+H) = 2 => s = 2/(W+H).
        # So r = s = 2 / (w_total + h_total).
        
        scale = 2.0 / (w_total + h_total)
        r = scale * 1.0
        w = scale * w_total
        h = scale * h_total
        
        # Transform points to final coordinates
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def pack_grid(rows, cols):
        """
        Generate a grid packing configuration.
        """
        pts = []
        # Square grid, spacing 2 (r=1)
        for i in range(cols):
            for j in range(rows):
                if i * rows + j >= N: break
                pts.append((i * 2.0, j * 2.0))
        
        pts = np.array(pts[:N])
        
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale * 1.0
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def refine(w, h, r, xs, ys):
        """
        Refine the packing using SLSQP.
        Variables: w, r, xs, ys. h is determined by 2-w.
        """
        # Check time
        if time.time() > deadline - 2.0:
            return w, h, r, xs, ys

        # Variable vector: [w, r, x0...x20, y0...y20]
        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0 or h_val <= 0 or r_val <= 0:
                return 1e6
            return -r_val

        def constraints(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs = v[2:2+N]
            ys = v[2+N:2+2*N]
            
            cons = []
            
            # Aspect ratio sanity
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
            
            # Boundary
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+N+i] - v[1]})
                
            # Overlap
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda v, i=i, j=j: np.sqrt((v[2+i]-v[2+j])**2 + (v[2+N+i]-v[2+N+j])**2) - 2*v[1]
                    })
            return cons

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 100, 'ftol': 1e-8, 'disp': False})
            
            if res.success:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2+N]
                ys_res = res.x[2+N:2+2*N]
                
                # Validate
                valid = True
                if w_res <= 0 or h_res <= 0 or r_res <= 0:
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res - 1e-6 or xs_res[i] > w_res - r_res + 1e-6:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-6 or ys_res[i] > h_res - r_res + 1e-6:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2)
                                if dist < 2*r_res - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
            
        return w, h, r, xs, ys

    best_r = 0
    best_config = None

    # List of initial configurations to try
    configs = []
    
    # 1. Hexagonal 7-7-7
    configs.append(pack_hexagonal([7, 7, 7]))
    
    # 2. Hexagonal 4-5-4-4-4 (21 circles)
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    
    # 3. Hexagonal 3-4-4-4-4-2? No, sum 21.
    # 3. 4-4-4-4-4-1?
    configs.append(pack_hexagonal([4, 4, 4, 4, 4, 1]))
    
    # 4. Grid 5x5 (using 21)
    configs.append(pack_grid(5, 5))
    
    # 5. Grid 7x3 (using 21)
    configs.append(pack_grid(3, 7))
    
    # 6. Grid 4x6 (using 24, take 21)
    configs.append(pack_grid(4, 6))
    
    # 7. Random restarts
    # Generate a few random configurations and refine
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
        # Random positions in a 1x1 square
        w = 1.0
        h = 1.0
        r = 0.05
        xs = np.random.uniform(r, w-r, N)
        ys = np.random.uniform(r, h-r, N)
        configs.append((w, h, r, xs, ys))

    for cfg in configs:
        if time.time() > deadline - 2.0:
            break
            
        w, h, r, xs, ys = cfg
        
        # Refine
        w_new, h_new, r_new, xs_new, ys_new = refine(w, h, r, xs, ys)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    if best_config is None:
        # Fallback
        w, h, r, xs, ys = configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r, xs, ys = best_config
    
    # Final safety adjustment
    r_final = r * (1 - 1e-5)
    
    return np.column_stack([xs, ys, np.full(N, r_final)])
