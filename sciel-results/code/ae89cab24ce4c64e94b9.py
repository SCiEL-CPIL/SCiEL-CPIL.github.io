import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    best_val = -np.inf
    best_res = None

    # Vectorized objective function
    def objective(v, w, h, penalty_weight=10.0):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # 1. Boundary penalties
        # Left: x < r -> r - x
        # Right: x > w - r -> x - (w - r)
        # Bottom: y < r -> r - y
        # Top: y > h - r -> y - (h - r)
        
        pen_bound = 0.0
        # Vectorized max(0, val)^2
        pen_bound += np.sum(np.maximum(0, r - x)**2)
        pen_bound += np.sum(np.maximum(0, x - (w - r))**2)
        pen_bound += np.sum(np.maximum(0, r - y)**2)
        pen_bound += np.sum(np.maximum(0, y - (h - r))**2)
        
        # 2. Overlap penalties
        # Compute all pairwise distances
        # Use broadcasting: (N, 1, 1) vs (1, N, 1) etc.
        # dx[i, j] = x[i] - x[j]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Sum of radii for pairs
        sum_r = r[:, None] + r[None, :]
        
        # Violation is max(0, sum_r - dist)
        # We only care about i < j to avoid double counting, but summing all i!=j and dividing by 2 is equivalent
        # Or just mask the upper triangle
        # For speed, let's just sum all i!=j and divide by 2 in the penalty? No, penalty is sum of squares.
        # Let's use a mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        violations = np.maximum(0, sum_r - dist)
        # Zero out diagonal and lower triangle
        violations[~mask] = 0
        
        pen_overlap = np.sum(violations**2)
        
        total_penalty = pen_bound + pen_overlap
        
        # Maximize sum(r) => Minimize -sum(r)
        return -np.sum(r) + penalty_weight * total_penalty

    # Pre-compute indices for efficiency if needed, but N is small
    
    # Sweep aspect ratios
    # w + h = 2
    # w in [0.6, 1.4] seems reasonable
    for w in np.arange(0.6, 1.41, 0.1):
        h = 2.0 - w
        
        if time.time() - start_time > timeout:
            break
            
        # Try a few initializations
        inits = []
        
        # 1. Grid Initialization
        # Approximate grid size
        n_c = int(np.round(np.sqrt(N * w / h)))
        if n_c < 1: n_c = 1
        n_r = int(np.ceil(N / n_c))
        
        xs = []
        ys = []
        rs = []
        count = 0
        for i in range(n_r):
            for j in range(n_c):
                if count >= N: break
                x = (j + 0.5) * w / n_c
                y = (i + 0.5) * h / n_r
                # Start with a radius that fits the cell, slightly smaller
                r = min(w/n_c, h/n_r) / 2 * 0.95
                xs.append(x)
                ys.append(y)
                rs.append(r)
                count += 1
        
        def make_vec(xs, ys, rs):
            v = np.zeros(3*N)
            v[0::3] = xs
            v[1::3] = ys
            v[2::3] = rs
            return v
            
        v_grid = make_vec(xs, ys, rs)
        inits.append(v_grid)
        
        # 2. Jittered Grid
        for _ in range(5):
            v_rand = v_grid.copy()
            # Jitter positions
            v_rand[0::3] += np.random.uniform(-0.02, 0.02, N)
            v_rand[1::3] += np.random.uniform(-0.02, 0.02, N)
            # Clip to bounds (loose)
            v_rand[0::3] = np.clip(v_rand[0::3], 0.01, w - 0.01)
            v_rand[1::3] = np.clip(v_rand[1::3], 0.01, h - 0.01)
            inits.append(v_rand)
            
        # 3. Random Initialization with small radii
        for _ in range(3):
            v_rand = np.zeros(3*N)
            v_rand[0::3] = np.random.uniform(0.05, w - 0.05, N)
            v_rand[1::3] = np.random.uniform(0.05, h - 0.05, N)
            v_rand[2::3] = np.random.uniform(0.05, 0.1, N)
            inits.append(v_rand)

        for v0 in inits:
            if time.time() - start_time > timeout:
                break
                
            # Define bounds for optimizer
            # x in [0, w], y in [0, h], r in [0, min(w,h)/2] (theoretical max)
            max_r = min(w, h) / 2.0
            bounds = [(0, w), (0, h), (0, max_r)] * N
            
            # Use L-BFGS-B
            # maxiter increased, but we have a time limit
            try:
                res = minimize(
                    objective, 
                    v0, 
                    args=(w, h, 10.0), 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
                )
            except Exception:
                continue
                
            # Check validity
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            valid = True
            # Boundary check
            if np.any(x < r - 1e-5) or np.any(x > w - r + 1e-5) or \
               np.any(y < r - 1e-5) or np.any(y > h - r + 1e-5):
                valid = False
            
            # Overlap check
            if valid:
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                sum_r = r[:, None] + r[None, :]
                # Check i < j
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                if np.any(sum_r[mask] > dist[mask] + 1e-5):
                    valid = False
            
            if valid:
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_res = res.x.copy()

    if best_res is None:
        # Fallback to a valid configuration
        # Simple grid in a square (w=1, h=1)
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (j + 0.5) / side
                y = (i + 0.5) / side
                r = (1.0 / side) * 0.49 # Slightly smaller to ensure no touch
                cells.append([x, y, r])
        return np.array(cells[:N])
        
    # Format output
    x = best_res[0::3]
    y = best_res[1::3]
    r = best_res[2::3]
    
    out = np.zeros((N, 3))
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = r
    
    return out
