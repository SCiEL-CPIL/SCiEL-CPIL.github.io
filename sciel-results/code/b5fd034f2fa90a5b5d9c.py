import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_val = -1
    best_res = None
    
    # Try different aspect ratios
    # W + H = 2
    # W in [0.5, 1.5] is sufficient due to symmetry
    widths = np.linspace(0.7, 1.3, 13)
    
    for W in widths:
        H = 2.0 - W
        if H <= 0: continue
        
        # Initialize
        # Grid init
        side = int(np.ceil(np.sqrt(N)))
        # Create grid points
        xs = np.linspace(0.5/W, 1-0.5/W, side) # scaled to [0,1] then to [0,W]?
        # Better: just place in [r, W-r]
        
        # Let's use a standard grid init in the box
        init_x = np.zeros(N)
        init_y = np.zeros(N)
        init_r = np.zeros(N)
        
        # Simple grid
        cols = int(np.sqrt(N))
        rows = int(np.ceil(N / cols))
        # Adjust cols/rows to fit N
        # Just use a dense grid
        k = 0
        for i in range(cols):
            for j in range(rows):
                if k >= N: break
                # Center of cell
                cx = (i + 0.5) * W / cols
                cy = (j + 0.5) * H / rows
                # Max radius is half min(W/cols, H/rows)
                max_r = 0.5 * min(W/cols, H/rows) * 0.9
                init_x[k] = cx
                init_y[k] = cy
                init_r[k] = max_r
                k += 1
        
        x0 = np.concatenate([init_x, init_y, init_r])
        
        # Bounds
        # x in [0, W], y in [0, H], r in [0, min(W,H)/2]
        # Actually x in [r, W-r] is better enforced by penalty or bounds on x relative to r?
        # L-BFGS-B only supports simple bounds.
        # So we bound x in [0, W], y in [0, H], r in [0, max_r_global].
        # And add penalties for x < r, x > W-r, etc.
        
        max_r_global = min(W, H) / 2
        
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((1e-6, max_r_global))
            
        def objective(v):
            x = v[0:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            
            # Sum of radii
            val = -np.sum(r)
            
            # Penalties
            pen = 0
            # Boundary
            pen += np.sum(np.maximum(0, r - x)**2)
            pen += np.sum(np.maximum(0, x - (W - r))**2)
            pen += np.sum(np.maximum(0, r - y)**2)
            pen += np.sum(np.maximum(0, y - (H - r))**2)
            
            # Pairwise
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    req = r[i] + r[j]
                    if dist < req:
                        pen += (req - dist)**2
            
            return val + 100.0 * pen
        
        # Optimize
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
        
        # Check validity
        x = res.x[0:N]
        y = res.x[N:2*N]
        r = res.x[2*N:3*N]
        
        valid = True
        if np.any(r <= 0): valid = False
        if np.any(x < r - 1e-6) or np.any(x > W - r + 1e-6): valid = False
        if np.any(y < r - 1e-6) or np.any(y > H - r + 1e-6): valid = False
        
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-6:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_res = np.column_stack((x, y, r))
                
    if best_res is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        # Scale to perimeter 4?
        # The seed returns coordinates in [0,1] roughly?
        # The task says "bounding rectangle has perimeter at most 4".
        # If I return coords in [0, W]x[0, H] with W+H=2, the perimeter is 4.
        # The seed program returns coordinates that seem to be in a unit square?
        # Let's check the seed:
        # cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) ...]
        # If side=5 (sqrt(21)=4.58->5), coords are in [0,1].
        # If the rectangle is 1x1, perimeter is 4.
        # So the seed assumes a 1x1 square.
        # My solution should also output coordinates in a rectangle with perimeter 4.
        # If I use W=1, H=1, it matches.
        
        return np.array(cells[:N])
        
    return best_res
