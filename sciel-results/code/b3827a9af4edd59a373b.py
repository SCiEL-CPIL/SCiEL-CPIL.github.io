"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    best_score = -np.inf
    best_result = None

    # Initial grid configuration
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                centers.append([0.5/side + i/side, 0.5/side + j/side])
                radii.append(0.45/side)
    centers = np.array(centers[:N])
    radii = np.array(radii[:N])
    
    def _evaluate(centers, radii):
        """Compute objective: sum of radii, penalize violations."""
        # Check boundary constraints
        for i in range(len(centers)):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9:
                return -np.inf
            if y - r < -1e-9 or y + r > 1 + 1e-9:
                return -np.inf
        
        # Check pairwise distances
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    return -np.inf
        
        return np.sum(radii)
    
    # Try multiple random restarts with gradient-based optimization
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        if restart == 0:
            # Use grid start
            c_init = centers.copy()
            r_init = radii.copy()
        else:
            # Random positions
            c_init = np.random.rand(N, 2) * 0.8 + 0.1
            # Random radii, small enough to likely fit
            r_init = np.random.uniform(0.01, 0.08, N)
        
        # Variables: [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.concatenate([c_init.flatten(), r_init])
        
        def objective(vars_):
            if time.time() > deadline - 3:
                return 1e10
            c = vars_[:2*N].reshape(N, 2)
            r = vars_[2*N:]
            
            # Boundary penalties
            pen = 0.0
            for i in range(N):
                x, y = c[i]
                ri = r[i]
                pen += max(0, ri - x) * 1000
                pen += max(0, ri - (1-x)) * 1000
                pen += max(0, ri - y) * 1000
                pen += max(0, ri - (1-y)) * 1000
            
            # Pairwise overlap penalties
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(c[i] - c[j])
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        pen += overlap * 1000
            
            return -np.sum(r) + pen
        
        # Bounds: x,y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # r
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20})
            
            c_opt = res.x[:2*N].reshape(N, 2)
            r_opt = res.x[2*N:]
            
            score = _evaluate(c_opt, r_opt)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([c_opt, r_opt])
                
        except Exception:
            continue
    
    # If no valid result found, return grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
