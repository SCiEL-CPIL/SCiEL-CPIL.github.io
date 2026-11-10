import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    n = 8
    
    # Precompute indices for unordered pairs (i < j)
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]
    num_pairs = len(pair_indices)
    
    # Constants for the objective function
    # Score = (Product of all 56 ordered pair distances) / 8^8
    # After scaling to diameter D=2:
    # d_scaled = d * (2/D)
    # Product_scaled = Product(d) * (2/D)^56
    # log(Score) = log(Product(d)) + 56*log(2) - 56*log(D) - 8*log(8)
    # log(Product(d)) = 2 * sum_{i<j} log(d_ij)
    LOG_2 = np.log(2.0)
    LOG_8 = np.log(8.0)
    CONST_OFFSET = 56.0 * LOG_2 - 8.0 * LOG_8

    def objective(x):
        """
        x: 16-vector representing 8 points in 2D.
        Returns -log(Score) for minimization.
        """
        if time.time() > deadline:
            return 1e10
        
        points = x.reshape(n, 2)
        
        # Calculate pairwise distances for i < j
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Check for zero distances (distinct points required)
        if np.any(dists < 1e-12):
            return 1e10
            
        D = np.max(dists)
        if D < 1e-12:
            return 1e10
            
        # Sum of logs of distances
        sum_log_d = np.sum(np.log(dists))
        
        # log(Score) = CONST_OFFSET + 2 * sum_log_d - 56 * log(D)
        log_score = CONST_OFFSET + 2.0 * sum_log_d - 56.0 * np.log(D)
        
        return -log_score

    # --- Phase 0: Generate Diverse Initial Points ---
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0_oct = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    starts = [x0_oct]
    
    # 2. Two Squares Configurations (from current program's insight)
    # 4 points on circle r1, 4 on circle r2, rotated by theta
    r1_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    r2_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    theta_vals = [0.0, np.pi/8, np.pi/4, np.pi/3, np.pi/2]
    
    for r1 in r1_vals:
        for r2 in r2_vals:
            for theta in theta_vals:
                p1 = []
                p2 = []
                for k in range(4):
                    a1 = k * np.pi / 2
                    p1.append([r1 * np.cos(a1), r1 * np.sin(a1)])
                    a2 = theta + k * np.pi / 2
                    p2.append([r2 * np.cos(a2), r2 * np.sin(a2)])
                pts = np.array(p1 + p2)
                starts.append(pts.flatten())
                
    # 3. Random Perturbations of Octagon
    np.random.seed(42)
    for _ in range(20):
        noise = np.random.normal(0, 0.3, 2 * n)
        starts.append(x0_oct + noise)
        
    # 4. Fully Random Starts
    for _ in range(20):
        starts.append(np.random.uniform(-2, 2, 2 * n))

    best_val = float('inf')
    best_x = x0_oct

    # --- Phase 1: Fast Local Refinement (L-BFGS-B) ---
    # Quickly evaluate all starts to find promising basins
    for s in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                objective,
                s,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # --- Phase 2: Global Search (Differential Evolution) ---
    # Explore the 16D space to find better basins if time permits
    if time.time() < deadline - 40.0:
        bounds = [(-5.0, 5.0)] * (2 * n)
        try:
            de_res = differential_evolution(
                objective,
                bounds,
                maxiter=150,
                popsize=20,
                tol=1e-8,
                seed=42,
                init='latinhypercube',
                polish=False  # We will polish manually for better control
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x
        except Exception:
            pass

    # --- Phase 3: Basin Hopping ---
    # Escape local minima around the current best
    if time.time() < deadline - 20.0:
        try:
            def step(x):
                noise = np.random.normal(0, 0.1, len(x))
                return x + noise

            bh_res = basinhopping(
                objective,
                best_x,
                niter=30,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 50}},
                stepsize=0.2,
                seed=42
            )
            if bh_res.fun < best_val:
                best_val = bh_res.fun
                best_x = bh_res.x
        except Exception:
            pass

    # --- Phase 4: High-Precision Polish ---
    # Use Powell and Nelder-Mead for final precision
    if time.time() < deadline - 5.0:
        try:
            # Powell is robust for box-less optimization
            res_powell = minimize(
                objective,
                best_x,
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res_powell.fun < best_val:
                best_val = res_powell.fun
                best_x = res_powell.x
        except Exception:
            pass

        # Nelder-Mead for final touch
        if time.time() < deadline - 2.0:
            try:
                res_nm = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res_nm.fun < best_val:
                    best_val = res_nm.fun
                    best_x = res_nm.x
            except Exception:
                pass

    # --- Final Validation and Return ---
    best_points = best_x.reshape(n, 2)
    
    # Ensure distinctness
    min_d = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < min_d:
                min_d = d
    if min_d < 1e-8:
        # Nudge closest pair
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < 1e-8:
                    best_points[j] += np.array([1e-6, 0])

    # Scale to ensure coordinates are within bounds (|coord| < 1e6)
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e5:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
