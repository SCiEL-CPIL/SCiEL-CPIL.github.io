import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s margin
    
    # Variables: w, h, x1...xN, y1...yN, r1...rN
    # Total size: 2 + 3*N
    n_vars = 2 + 3 * N
    
    def unpack(x):
        w = x[0]
        h = x[1]
        # Ensure w, h positive
        w = max(w, 1e-6)
        h = max(h, 1e-6)
        
        x_coords = x[2 : 2+N]
        y_coords = x[2+N : 2+2*N]
        r_coords = x[2+2*N : 2+3*N]
        r_coords = np.maximum(r_coords, 1e-6)
        
        return w, h, x_coords, y_coords, r_coords

    def objective(x):
        w, h, x_coords, y_coords, r_coords = unpack(x)
        # Maximize sum of radii
        return -np.sum(r_coords)

    def constraints(x):
        w, h, x_coords, y_coords, r_coords = unpack(x)
        
        cons = []
        
        # Perimeter constraint: 2*(w+h) <= 4 => w+h <= 2
        cons.append({'type': 'ineq', 'fun': lambda x, w=w, h=h: 2 - (w + h)})
        
        # Circle bounds: x >= r, x <= w-r
        # x_i - r_i >= 0
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, w=w, h=h, x_coords=x_coords, y_coords=y_coords, r_coords=r_coords: x_coords[i] - r_coords[i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, w=w, h=h, x_coords=x_coords, y_coords=y_coords, r_coords=r_coords: (w - r_coords[i]) - x_coords[i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, w=w, h=h, x_coords=x_coords, y_coords=y_coords, r_coords=r_coords: y_coords[i] - r_coords[i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, w=w, h=h, x_coords=x_coords, y_coords=y_coords, r_coords=r_coords: (h - r_coords[i]) - y_coords[i]})
            
        # Non-overlap: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                def non_overlap(x, i=i, j=j, x_coords=x_coords, y_coords=y_coords, r_coords=r_coords):
                    dx = x_coords[i] - x_coords[j]
                    dy = y_coords[i] - y_coords[j]
                    dr = r_coords[i] + r_coords[j]
                    return dx*dx + dy*dy - dr*dr
                cons.append({'type': 'ineq', 'fun': non_overlap})
                
        return cons

    def make_initial_guess(w, h):
        # Grid initialization
        # 5x5 grid
        side = 5
        cells = []
        # We need 21 circles. 5x5=25.
        # Let's just take the first 21 in row-major order?
        # Or center them?
        # Let's try to distribute them evenly.
        # 4 rows x 5 cols = 20. Plus 1?
        # Let's use a 5x5 grid and pick 21.
        indices = []
        for i in range(side):
            for j in range(side):
                indices.append((i, j))
        
        # Shuffle to pick random 21? No, deterministic is better for reproducibility, 
        # but random restarts help. 
        # Let's stick to a structured grid for the first guess.
        # Pick 21 cells.
        # To be symmetric, maybe remove corners?
        # Let's just take the first 21.
        selected = indices[:N]
        
        x0 = []
        x0.append(w)
        x0.append(h)
        
        # Initialize coords
        x_coords = []
        y_coords = []
        r_coords = []
        
        # Calculate max possible radius in grid
        # Grid spacing
        dx = w / side
        dy = h / side
        r_max = min(dx, dy) / 2 - 1e-4
        
        for i, j in selected:
            x_coords.append((i + 0.5) * dx)
            y_coords.append((j + 0.5) * dy)
            r_coords.append(r_max)
            
        x0.extend(x_coords)
        x0.extend(y_coords)
        x0.extend(r_coords)
        
        return np.array(x0)

    best_val = -np.inf
    best_x = None
    
    # Try a few different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        # w * ar = h ? No, w/h = ar => h = w/ar.
        # w + w/ar = 2 => w (1 + 1/ar) = 2 => w = 2 / (1 + 1/ar)
        w_init = 2 / (1 + 1/ar)
        h_init = w_init * ar
        
        # Ensure w, h are reasonable
        if w_init < 0.1 or h_init < 0.1:
            continue
            
        x0 = make_initial_guess(w_init, h_init)
        
        # Bounds
        # w, h: [0.1, 1.9]
        # x, y: [0, 2]
        # r: [1e-6, 1.0]
        bounds = [(0.1, 1.9), (0.1, 1.9)]
        bounds += [(0.0, 2.0)] * N
        bounds += [(0.0, 2.0)] * N
        bounds += [(1e-6, 1.0)] * N
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x
            else:
                # Even if not success, check if it's better and valid
                # SLSQP often returns "success" False but good result
                val = -res.fun
                # Check validity roughly
                w, h, x_coords, y_coords, r_coords = unpack(res.x)
                if w + h <= 2.0001:
                    # Check overlaps
                    valid = True
                    for i in range(N):
                        if x_coords[i] < r_coords[i] - 1e-5 or x_coords[i] > w - r_coords[i] + 1e-5:
                            valid = False
                            break
                        if y_coords[i] < r_coords[i] - 1e-5 or y_coords[i] > h - r_coords[i] + 1e-5:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = x_coords[i] - x_coords[j]
                                dy = y_coords[i] - y_coords[j]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < r_coords[i] + r_coords[j] - 1e-5:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid and val > best_val:
                        best_val = val
                        best_x = res.x
                        
        except Exception as e:
            pass

    if best_x is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
        return np.array(cells)
        
    # Reconstruct result
    w, h, x_coords, y_coords, r_coords = unpack(best_x)
    
    # Final check and slight adjustment for numerical errors
    # Shrink radii slightly to ensure disjoint interiors
    scale = 0.999
    r_final = r_coords * scale
    
    # Check if any circle is out of bounds after scaling? No, scaling r makes it smaller, so safer.
    # But we need to ensure x, y are still valid.
    # If r shrinks, x, y don't need to change, but the circle is smaller.
    # The constraints were x >= r. If r becomes smaller, x >= r_new is still true if x >= r_old.
    
    result = np.column_stack((x_coords, y_coords, r_final))
    
    return result
