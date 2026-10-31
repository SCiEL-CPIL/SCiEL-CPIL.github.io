import numpy as np
from scipy.optimize import differential_evolution
import time

N = 26

def construct():
    start_time = time.time()
    timeout = 115

    # Vectorized objective function for Differential Evolution
    # pop shape: (M, 3*N)
    def objective_vectorized(pop):
        M = pop.shape[0]
        x = pop[:, :N]
        y = pop[:, N:2*N]
        r = pop[:, 2*N:3*N]
        
        # Boundary penalties
        pen_x_min = np.maximum(0, r - x)**2
        pen_x_max = np.maximum(0, x - (1 - r))**2
        pen_y_min = np.maximum(0, r - y)**2
        pen_y_max = np.maximum(0, y - (1 - r))**2
        bound_pen = np.sum(pen_x_min + pen_x_max + pen_y_min + pen_y_max, axis=1)
        
        # Overlap penalties
        x_diff = x[:, :, None] - x[:, None, :]
        y_diff = y[:, :, None] - y[:, None, :]
        d_sq = x_diff**2 + y_diff**2
        
        idx = np.arange(N)
        d_sq[:, idx, idx] = np.inf
        d = np.sqrt(d_sq)
        
        r_sum = r[:, :, None] + r[:, None, :]
        
        overlap = np.maximum(0, r_sum - d)
        overlap_pen = np.sum(overlap**2, axis=(1, 2))
        
        total_pen = bound_pen + overlap_pen
        
        return -np.sum(r, axis=1) + 100.0 * total_pen

    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
    for _ in range(N):
        bounds.append((0.0, 1.0))
    for _ in range(N):
        bounds.append((0.01, 0.5))

    try:
        result = differential_evolution(
            objective_vectorized, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-5,
            init='sobol',
            vectorized=True,
            polish=False
        )
        best_params = result.x
    except Exception as e:
        best_params = None

    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    cr = 0.5 / side - 0.01
                    cells.append([cx, cy, cr])
        return np.array(cells[:N])

    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Post-processing
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    for _ in range(50):
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        r_sum = np.add.outer(r, r)
        
        overlaps = np.maximum(0, r_sum - d)
        max_overlap = np.max(overlaps)
        
        if max_overlap < 1e-8:
            break
        r = r * 0.99
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)

    return np.column_stack((x, y, r))
