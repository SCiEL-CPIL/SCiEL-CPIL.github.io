# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s kill

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y

    # Precompute the index arrays for each projection
    XY_diff = (X - Y + (n - 1)).flatten()  # 0 to 2*(n-1)
    X_flat = X.flatten()  # 0 to n-1
    XY_sum = (X + Y).flatten()  # 0 to 2*(n-1)
    X2Y = (X + 2 * Y).flatten()  # 0 to 3*(n-1)

    def compute_ratio(P):
        """Compute D/M ratio for a given distribution P (vectorized)."""
        P = np.maximum(P, 0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        P_flat = P.flatten()

        # H(X-Y)
        XY_dist = np.bincount(XY_diff, weights=P_flat, minlength=2 * n - 1)
        XY_nonzero = XY_dist[XY_dist > 1e-15]
        H_XY = -np.sum(XY_nonzero * np.log2(XY_nonzero)) if len(XY_nonzero) > 0 else 0.0

        # H(X)
        X_dist = np.bincount(X_flat, weights=P_flat, minlength=n)
        X_nonzero = X_dist[X_dist > 1e-15]
        H_X = -np.sum(X_nonzero * np.log2(X_nonzero)) if len(X_nonzero) > 0 else 0.0

        # H(X+Y)
        XY_sum_dist = np.bincount(XY_sum, weights=P_flat, minlength=2 * n - 1)
        XY_sum_nonzero = XY_sum_dist[XY_sum_dist > 1e-15]
        H_XY_sum = -np.sum(XY_sum_nonzero * np.log2(XY_sum_nonzero)) if len(XY_sum_nonzero) > 0 else 0.0

        # H(X+2Y)
        X2Y_dist = np.bincount(X2Y, weights=P_flat, minlength=3 * n - 1)
        X2Y_nonzero = X2Y_dist[X2Y_dist > 1e-15]
        H_X2Y = -np.sum(X2Y_nonzero * np.log2(X2Y_nonzero)) if len(X2Y_nonzero) > 0 else 0.0

        M = max(H_X, H_XY_sum, H_X2Y)
        if M <= 1e-9:
            return 0.0
        return H_XY / M

    def objective(params):
        """Negative of the ratio for minimization."""
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        return -compute_ratio(P)

    # Start with structured initializations
    np.random.seed(42)
    
    # Initial point 1: uniform on anti-diagonals (maximizes H(X-Y))
    P_best = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            k = abs(x - y)
            P_best[y, x] = 1.0 / (19.0 * (n - k))
    P_best /= P_best.sum()
    best_ratio = compute_ratio(P_best)

    # Try L-BFGS-B from multiple diverse starting points
    # Key insight: use log-parameterization and L-BFGS-B which handles
    # the smooth objective better than Nelder-Mead in high dimensions
    
    starting_points = []
    
    # 1. Anti-diagonal uniform (current best)
    starting_points.append(P_best.copy())
    
    # 2. Concentrate on cells where x-y has high entropy but x, x+y, x+2y are concentrated
    #    Try: put mass on a subset of anti-diagonals
    for subset_size in [3, 4, 5, 6, 7]:
        for offset in range(n):
            P_candidate = np.zeros((n, n), dtype=np.float64)
            for y in range(n):
                for x in range(n):
                    d = (x - y + n) % (2*n - 1)  # wrap
                    if d in range(offset, offset + subset_size) or d in range(2*n-1 - subset_size, 2*n-1):
                        P_candidate[y, x] = 1.0
            if P_candidate.sum() > 0:
                P_candidate /= P_candidate.sum()
                starting_points.append(P_candidate.copy())
    
    # 3. Random Dirichlet samples
    for _ in range(15):
        P_rand = np.random.dirichlet(np.ones(n*n))
        P_rand = P_rand.reshape(n, n)
        starting_points.append(P_rand)
    
    # 4. Structured: emphasize certain rows/columns
    #    Put mass where x and y are correlated in specific ways
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        P_struct = np.zeros((n, n), dtype=np.float64)
        for y in range(n):
            for x in range(n):
                # Bias toward cells where x-y is spread but x, x+y are concentrated
                P_struct[y, x] = np.exp(-alpha * (x - y)**2 / (2*n))
        P_struct /= P_struct.sum()
        starting_points.append(P_struct)
    
    # 5. Sparse patterns: put mass on specific cells
    for trial in range(10):
        P_sparse = np.zeros((n, n), dtype=np.float64)
        # Pick random cells
        for _ in range(20):
            ry = np.random.randint(0, n)
            rx = np.random.randint(0, n)
            P_sparse[ry, rx] = 1.0
        P_sparse /= P_sparse.sum()
        starting_points.append(P_sparse)
    
    # Optimize from each starting point using L-BFGS-B
    for i, P_init in enumerate(starting_points):
        if time.time() > deadline - 30:
            break
        
        P_init_safe = np.maximum(P_init, 1e-12)
        params_init = np.log(P_init_safe.flatten())
        
        try:
            # Use L-BFGS-B which is much faster in high dimensions
            result = minimize(objective, params_init, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000})
            
            P_opt = np.exp(result.x).reshape(n, n)
            P_opt = np.maximum(P_opt, 0)
            total = P_opt.sum()
            if total > 1e-9:
                P_opt /= total
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass
    
    # Final polish with coordinate descent on the best solution
    P_work = P_best.copy()
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
        
        improved = False
        for idx in range(n * n):
            old_val = P_work.flat[idx]
            # Try moving mass from this cell to others
            for target_idx in range(n * n):
                if target_idx == idx:
                    continue
                # Move small amount of mass
                for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    move_amt = old_val * frac
                    if move_amt < 1e-12:
                        continue
                    P_trial = P_work.copy()
                    P_trial.flat[idx] -= move_amt
                    P_trial.flat[target_idx] += move_amt
                    P_trial = np.maximum(P_trial, 0)
                    total = P_trial.sum()
                    if total < 1e-9:
                        continue
                    P_trial /= total
                    trial_ratio = compute_ratio(P_trial)
                    if trial_ratio > best_ratio + 1e-10:
                        P_work = P_trial.copy()
                        best_ratio = trial_ratio
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    P_best = P_work
    
    # Final normalization and clipping
    P_best = np.maximum(P_best, 0)
    P_best /= P_best.sum()

    return P_best

# EVOLVE-BLOCK-END
