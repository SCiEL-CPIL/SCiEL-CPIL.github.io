import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 600s kill

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)

    # Precompute bin indices (flattened)
    XY_diff = (X - Y + (n - 1)).flatten()   # 0..18
    X_flat = X.flatten()                     # 0..9
    XY_sum = (X + Y).flatten()               # 0..18
    X2Y = (X + 2 * Y).flatten()              # 0..27

    def compute_ratio(P):
        P = np.maximum(P, 0.0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        Pf = P.flatten()

        def entropy(indices, minlength):
            dist = np.bincount(indices, weights=Pf, minlength=minlength)
            nz = dist[dist > 1e-15]
            if len(nz) == 0:
                return 0.0
            return -np.sum(nz * np.log2(nz))

        H_diff = entropy(XY_diff, 2 * n - 1)
        H_x = entropy(X_flat, n)
        H_sum = entropy(XY_sum, 2 * n - 1)
        H_2y = entropy(X2Y, 3 * n - 1)

        M = max(H_x, H_sum, H_2y)
        if M <= 1e-9:
            return 0.0
        return H_diff / M

    def objective(params):
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        return -compute_ratio(P)

    def objective_row(row_params, row_idx, P_other_rows):
        # row_params are log-weights for the current row
        # P_other_rows is the 10x10 array with the current row zeroed out
        P_new = P_other_rows.copy()
        P_new[row_idx, :] = np.exp(row_params)
        
        # We need to normalize the whole thing for the ratio calculation
        # But the ratio is scale-invariant, so we can just compute it on P_new
        # However, compute_ratio normalizes internally.
        # To be efficient, we can check if the row sum is reasonable.
        row_sum = np.exp(row_params).sum()
        if row_sum <= 1e-9:
            return 1e10
        
        # We want to maximize the ratio of the full distribution.
        # The other rows are fixed.
        return -compute_ratio(P_new)

    # Generate diverse starting points
    def make_starting_points():
        starts = []
        # 1. Uniform on X-Y: P[y,x] ∝ 1/(n-|x-y|)
        P1 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P1[y, x] = 1.0 / (n - abs(x - y))
        P1 /= P1.sum()
        starts.append(np.log(np.maximum(P1, 1e-10)).flatten())

        # 2. Uniform
        P2 = np.ones((n, n)) / (n * n)
        starts.append(np.log(P2).flatten())

        # 3. "Two-Column" Strategy: Mass on x=0 and x=9
        P3 = np.zeros((n, n))
        for y in range(n):
            P3[y, 0] = 1.0
            P3[y, 9] = 1.0
        P3 /= P3.sum()
        starts.append(np.log(np.maximum(P3, 1e-10)).flatten())

        # 4. "Three-Column" Strategy: Mass on x=0, x=4, x=9
        P4 = np.zeros((n, n))
        for y in range(n):
            P4[y, 0] = 1.0
            P4[y, 4] = 1.0
            P4[y, 9] = 1.0
        P4 /= P4.sum()
        starts.append(np.log(np.maximum(P4, 1e-10)).flatten())

        # 5. Random Dirichlet starts
        np.random.seed(12345)
        for i in range(30):
            P_r = np.random.dirichlet(np.ones(n * n))
            P_r = P_r.reshape(n, n)
            starts.append(np.log(np.maximum(P_r, 1e-10)).flatten())

        # 6. Structured: weight by exp(-alpha * |x-y|)
        for alpha in [0.5, 1.0, 2.0, 3.0]:
            P_a = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    P_a[y, x] = np.exp(-alpha * abs(x - y))
            P_a /= P_a.sum()
            starts.append(np.log(np.maximum(P_a, 1e-10)).flatten())

        return starts

    starting_points = make_starting_points()

    best_ratio = 0.0
    P_best = np.ones((n, n)) / (n * n)

    # First pass: quick evaluation of all starting points
    for sp in starting_points:
        P = np.exp(sp).reshape(n, n)
        P = np.maximum(P, 0)
        P /= P.sum()
        r = compute_ratio(P)
        if r > best_ratio:
            best_ratio = r
            P_best = P.copy()

    # Differential Evolution for global exploration
    if time.time() < deadline - 120:
        try:
            bounds = [(-10, 10)] * (n * n)
            de_result = differential_evolution(objective, bounds, seed=42,
                                               maxiter=30, popsize=20,
                                               tol=1e-6, mutation=(0.5, 1.0),
                                               recombination=0.8,
                                               init='latinhypercube',
                                               polish=False)
            if de_result.fun < -best_ratio:
                P_opt = np.exp(de_result.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

    # Local refinement from best DE result and original best
    candidates = [np.log(np.maximum(P_best, 1e-10)).flatten()]
    if 'de_result' in locals():
        candidates.append(de_result.x)
    
    # Add top few starting points
    scored_starts = []
    for sp in starting_points:
        P = np.exp(sp).reshape(n, n)
        P = np.maximum(P, 0)
        P /= P.sum()
        r = compute_ratio(P)
        scored_starts.append((r, sp))
    scored_starts.sort(reverse=True, key=lambda x: x[0])
    for r, sp in scored_starts[:5]:
        candidates.append(sp)

    for params in candidates:
        if time.time() > deadline - 30:
            break

        # L-BFGS-B refinement
        try:
            result = minimize(objective, params, method='L-BFGS-B',
                             options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < -best_ratio:
                P_opt = np.exp(result.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

        # Block Coordinate Descent refinement
        if time.time() > deadline - 20:
            break
            
        P_cur = P_best.copy()
        params_cur = np.log(np.maximum(P_cur, 1e-10)).flatten().reshape(n, n)
        
        # Iterate rows
        for _ in range(20): # 20 sweeps
            if time.time() > deadline - 10:
                break
            for r in range(n):
                # Extract current row log-weights
                row_log_w = params_cur[r, :].copy()
                # Create P with this row zeroed out
                P_other = P_cur.copy()
                P_other[r, :] = 0
                
                # Define sub-objective
                def sub_obj(row_params):
                    return objective_row(row_params, r, P_other)
                
                try:
                    sub_res = minimize(sub_obj, row_log_w, method='L-BFGS-B',
                                      options={'maxiter': 50, 'gtol': 1e-10})
                    # Update P_cur
                    new_row = np.exp(sub_res.x)
                    # Renormalize the whole matrix? 
                    # The sub-objective used compute_ratio which normalizes internally.
                    # So the relative weights are what matter.
                    # We should update P_cur with the new row, then renormalize P_cur.
                    P_cur[r, :] = new_row
                    P_cur = np.maximum(P_cur, 0)
                    P_cur /= P_cur.sum()
                    params_cur = np.log(np.maximum(P_cur, 1e-10))
                except Exception:
                    pass
            
            # Check improvement
            r_cur = compute_ratio(P_cur)
            if r_cur > best_ratio:
                best_ratio = r_cur
                P_best = P_cur.copy()
            else:
                # If no improvement after a sweep, break early?
                # Or continue to see if it recovers?
                # Let's continue but break if stuck for 3 sweeps.
                # Simple: just break if no improvement in this sweep
                break

    # Final normalization and clipping
    P_best = np.maximum(P_best, 0)
    total = P_best.sum()
    if total <= 1e-9:
        P_best = np.ones((n, n)) / (n * n)
    else:
        P_best /= total

    return P_best
