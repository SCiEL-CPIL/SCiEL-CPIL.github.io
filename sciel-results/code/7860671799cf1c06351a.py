"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try multiple aspect ratios
    # Perimeter = 2*(w+h) = 4, so w+h = 2
    # Let's try aspect ratios w/h from 0.5 to 2.0
    aspect_ratios = np.linspace(0.5, 2.0, 15)
    
    # Also try some specific promising ratios
    aspect_ratios = np.unique(np.concatenate([aspect_ratios, [1.0, 1.5, 2.0/3, 0.75, 1.25]]))
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        # w + h = 2, w/h = ar => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initial configuration: grid-based
        # For 21 circles, try 5x4 grid (20) + 1, or 3x7, etc.
        # Let's use a 5x4 grid with one extra
        cols, rows = 5, 4
        # Actually, let's be smarter: try different grid layouts
        
        # Layout 1: 5 cols x 4 rows = 20, plus 1 in a gap
        # Layout 2: 3 cols x 7 rows = 21
        # Layout 3: 7 cols x 3 rows = 21
        # Layout 4: 2 cols x 11 rows = 22 (use 21)
        # Layout 5: 11 cols x 2 rows = 22 (use 21)
        
        layouts = [
            (5, 4, 1),   # 5x4 grid + 1 extra
            (3, 7, 0),   # 3x7 grid
            (7, 3, 0),   # 7x3 grid
            (2, 11, 0),  # 2x11 grid (use 21)
            (11, 2, 0),  # 11x2 grid (use 21)
            (4, 5, 1),   # 4x5 grid + 1 extra
        ]
        
        for (nc, nr, extra) in layouts:
            if time.time() > deadline - 3.0:
                break
            
            # Create initial positions
            positions = []
            radii = []
            
            # Grid positions
            for i in range(nc):
                for j in range(nr):
                    if len(positions) >= N:
                        break
                    x = (i + 0.5) * w / nc
                    y = (j + 0.5) * h / nr
                    positions.append([x, y])
            
            # Add extra circles if needed
            while len(positions) < N:
                # Place extra circles in gaps
                idx = len(positions)
                # Simple heuristic: place near center or in gaps
                x = w / 2 + (idx - N/2) * 0.1
                y = h / 2 + (idx - N/2) * 0.1
                positions.append([x, y])
            
            positions = np.array(positions[:N])
            
            # Initial radii: small
            r0 = min(w, h) / max(nc, nr) * 0.4
            radii = np.full(N, r0)
            
            # Optimize using Nelder-Mead
            # Variables: positions (2N) + radii (N) = 3N = 63
            def objective(params):
                if time.time() > deadline - 2.0:
                    return 1e10
                
                pos = params[:2*N].reshape(N, 2)
                rad = params[2*N:]
                
                # Penalty for out of bounds
                penalty = 0.0
                for i in range(N):
                    if pos[i][0] < rad[i] or pos[i][0] > w - rad[i]:
                        penalty += (max(0, rad[i] - pos[i][0]) + max(0, pos[i][0] - (w - rad[i]))) * 100
                    if pos[i][1] < rad[i] or pos[i][1] > h - rad[i]:
                        penalty += (max(0, rad[i] - pos[i][1]) + max(0, pos[i][1] - (h - rad[i]))) * 100
                
                # Penalty for overlap
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = rad[i] + rad[j]
                        if dist < min_dist:
                            penalty += (min_dist - dist) * 100
                
                # Maximize sum of radii => minimize negative
                return -np.sum(rad) + penalty
            
            x0 = np.concatenate([positions.flatten(), radii])
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                
                if result.fun < 1e9:  # valid result
                    pos = result.x[:2*N].reshape(N, 2)
                    rad = result.x[2*N:]
                    
                    # Verify validity
                    valid = True
                    for i in range(N):
                        if pos[i][0] < rad[i] - 1e-6 or pos[i][0] > w - rad[i] + 1e-6:
                            valid = False
                            break
                        if pos[i][1] < rad[i] - 1e-6 or pos[i][1] > h - rad[i] + 1e-6:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = pos[i][0] - pos[j][0]
                                dy = pos[i][1] - pos[j][1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < rad[i] + rad[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        value = np.sum(rad)
                        if value > best_value:
                            best_value = value
                            best_result = np.column_stack([pos, rad])
            except Exception:
                pass
    
    # If no valid result found, return a safe default
    if best_result is None:
        # Fallback: simple grid in a square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
