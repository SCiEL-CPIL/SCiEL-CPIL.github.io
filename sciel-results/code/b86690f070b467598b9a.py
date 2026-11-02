"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 142  # safety margin (150s limit)

    # Precompute indices for pairwise constraints
    pair_i, pair_j = np.triu_indices(N, k=1)
    n_pairs = len(pair_i)

    def compute_optimal_radii(x, y):
        """
        Solve LP to find optimal radii for fixed centers.
        Maximize sum(r) subject to:
        r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        r_i + r_j <= dist(i,j)
        r_i >= 0
        """
        # Objective: minimize -sum(r)
        c = -np.ones(N)
        
        # Constraints A_ub @ r <= b_ub
        # 1. Boundary constraints: 4 per circle
        # r_i <= x_i  ->  [0..1..0] r <= x_i
        # r_i <= 1-x_i -> [0..1..0] r <= 1-x_i
        # etc.
        # We can combine these into r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        # Let b_boundary = min(x_i, 1-x_i, y_i, 1-y_i)
        # Constraint: r_i <= b_boundary[i]
        
        b_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # 2. Pairwise constraints: r_i + r_j <= d_ij
        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        d_pairs = np.sqrt(dx**2 + dy**2)
        
        # Build A_ub matrix
        # Rows for boundary: Identity matrix
        # Rows for pairs: 1 at i, 1 at j
        A_ub = np.zeros((N + n_pairs, N))
        A_ub[:N, :] = np.eye(N)
        A_ub[N:, pair_i] = 1
        A_ub[N:, pair_j] = 1
        
        b_ub = np.concatenate([b_boundary, d_pairs])
        
        # Bounds: r_i >= 0
        bounds = [(0, None)] * N
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                radii = res.x
                # Clip small negative values to 0
                radii = np.maximum(radii, 0)
                return radii
            else:
                # Fallback to greedy if LP fails (should not happen)
                return greedy_radii(x, y)
        except Exception:
            return greedy_radii(x, y)

    def greedy_radii(x, y):
        """Fallback greedy radii calculation."""
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        return np.maximum(radii, 0)

    def objective(params):
        """Objective to minimize: -sum of optimal radii."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_optimal_radii(x, y)
        return -np.sum(radii)

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))  # 6
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x0[idx] = (i + 0.5) / side
                y0[idx] = (j + 0.5) / side
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x0[k] = 0.5
                y0[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def hex_init():
        """Hexagonal close-packing initialization."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        s = 1.0 / 5.5
        for row in range(6):
            for col in range(6):
                if idx >= N:
                    break
                x0[idx] = (col + 0.5) * s
                y0[idx] = (row + 0.5) * s * np.sqrt(3) / 2
                if row % 2 == 1:
                    x0[idx] += s / 2
                x0[idx] = np.clip(x0[idx], 0.02, 0.98)
                y0[idx] = np.clip(y0[idx], 0.02, 0.98)
                idx += 1
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    params0 = grid_init()
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)
    
    # Evaluate initializations
    initializations = [grid_init(), hex_init()]
    for init_params in initializations:
        obj_val = objective(init_params)
        if obj_val < best_obj:
            best_params = init_params
            best_obj = obj_val

    # Phase 1: Powell optimization on centers
    # Powell is derivative-free and suitable for the non-smooth LP objective
    if time.time() < deadline - 20:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 2: Refinement with smaller steps / Nelder-Mead
    if time.time() < deadline - 10:
        try:
            # Use Nelder-Mead for local refinement
            res = minimize(
                objective, best_params, method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 3: Final LP solve for the best centers
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_optimal_radii(x, y)
    
    # Shrink radii slightly for floating point safety
    radii *= 0.999999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
