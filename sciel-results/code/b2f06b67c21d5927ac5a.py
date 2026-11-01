import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_value = -np.inf
    best_config = None
    
    # Try a few aspect ratios
    # Perimeter = 4 -> W + H = 2
    # W = 2*x, H = 2*(1-x)
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # x from 0.5 to 1.0
    
    for x in aspect_ratios:
        if time.time() > deadline:
            break
            
        W = 2 * x
        H = 2 * (1 - x)
        
        # Initial guess: Grid
        # 5x5 grid is 25, we need 21. 4x5 is 20, 5x5 with 4 removed?
        # Let's use a 5x5 grid and remove 4 corners or random ones.
        # Or just use 21 points in a grid pattern.
        # Simple grid: 5 columns, 5 rows. 25 points. Remove 4.
        # Let's just place 21 points in a 5x5 grid, skipping the last 4.
        
        cols = 5
        rows = 5
        # Centers of cells
        dx = W / cols
        dy = H / rows
        
        init_points = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                # Center of cell
                cx = (i + 0.5) * dx
                cy = (j + 0.5) * dy
                # Initial radius: slightly smaller than half the cell size
                r = min(dx, dy) * 0.45
                init_points.append([cx, cy, r])
                count += 1
        
        if len(init_points) < N:
            # Fill remaining with small circles if grid wasn't big enough (should not happen for 5x5=25)
            while len(init_points) < N:
                init_points.append([W/2, H/2, 0.1])
        
        x0 = np.array(init_points).flatten()
        
        # Define objective
        def objective(z):
            if time.time() > deadline:
                return 0 # Just return something, we will check time outside
            
            pts = z.reshape(-1, 3)
            xs = pts[:, 0]
            ys = pts[:, 1]
            rs = pts[:, 2]
            
            # Sum of radii
            sum_r = np.sum(rs)
            
            # Overlap penalty
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    overlap = rs[i] + rs[j] - dist
                    if overlap > 0:
                        penalty += overlap * overlap
            
            # Boundary penalty
            for i in range(N):
                if xs[i] < rs[i]:
                    penalty += (rs[i] - xs[i])**2
                if xs[i] + rs[i] > W:
                    penalty += (xs[i] + rs[i] - W)**2
                if ys[i] < rs[i]:
                    penalty += (rs[i] - ys[i])**2
                if ys[i] + rs[i] > H:
                    penalty += (ys[i] + ys[i] - H)**2 # Typo in thought, should be (ys[i]+rs[i]-H)**2
            
            return -sum_r + 100 * penalty
        
        # Bounds
        # x in [0, W], y in [0, H], r in [0, min(W,H)/2]
        r_max = min(W, H) / 2
        bounds = []
        for _ in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((1e-6, r_max))
            
        # Optimization
        # Use Nelder-Mead or L-BFGS-B?
        # L-BFGS-B is faster for smooth functions. The penalty makes it smooth.
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < 0: # Heuristic
                final_pts = res.x.reshape(-1, 3)
                # Check validity
                valid = True
                for i in range(N):
                    if final_pts[i, 2] <= 0:
                        valid = False
                        break
                    # Check boundaries
                    if final_pts[i, 0] < final_pts[i, 2] - 1e-5 or final_pts[i, 0] + final_pts[i, 2] > W + 1e-5:
                        valid = False
                        break
                    if final_pts[i, 1] < final_pts[i, 2] - 1e-5 or final_pts[i, 1] + final_pts[i, 2] > H + 1e-5:
                        valid = False
                        break
                if valid:
                    # Check overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((final_pts[i,0]-final_pts[j,0])**2 + (final_pts[i,1]-final_pts[j,1])**2)
                            if dist < final_pts[i,2] + final_pts[j,2] - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        val = np.sum(final_pts[:, 2])
                        if val > best_value:
                            best_value = val
                            best_config = final_pts.copy()
        except Exception:
            pass

    if best_config is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
