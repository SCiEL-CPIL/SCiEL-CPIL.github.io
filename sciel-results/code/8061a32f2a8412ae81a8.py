import numpy as np
import time

def _precompute_tables():
    n = 10
    y_idx, x_idx = np.indices((n, n))
    
    # x - y range: -9 to 9 (19 values)
    diff = x_idx - y_idx
    diff_idx = diff + 9
    
    # x range: 0 to 9 (10 values)
    x_idx_flat = x_idx
    
    # x + y range: 0 to 18 (19 values)
    sum_idx = x_idx + y_idx
    
    # x + 2y range: 0 to 27 (28 values)
    sum2_idx = x_idx + 2 * y_idx
    
    return diff_idx, x_idx_flat, sum_idx, sum2_idx

# Pre-compute indices for fast evaluation
DIFF_IDX, X_IDX, SUM_IDX, SUM2_IDX = _precompute_tables()

def _fast_ratio(P):
    """
    Fast computation of D/M ratio.
    P is a 10x10 array, normalized or not (will be normalized internally if needed for stability, 
    but ratio is scale-invariant so we can skip normalization if P is positive).
    """
    # Check for invalid states quickly
    if not np.all(P > 0):
        # If any zero, entropy is still defined (0*log0=0), but if all zero, invalid.
        # The problem says "finite weights... entry < -1e-9 is invalid". 
        # Actually, the contract says: "Entries are clipped at 0 and P is normalized to sum 1."
        # And "any entry < -1e-9" is invalid. 
        # My optimizer works on log-probs, so exp(log_prob) is always > 0.
        # So P should always be > 0.
        pass
        
    total = np.sum(P)
    if total <= 1e-9:
        return 0.0
    
    # We don't strictly need to normalize P for entropy if we are careful, 
    # but entropy is scale invariant. 
    # H = - sum (p_i / total) * log2(p_i / total)
    # Let's just compute counts weighted by P.
    
    # H_{-1} : distribution of x - y
    dist_diff = np.zeros(19)
    np.add.at(dist_diff, DIFF_IDX.ravel(), P.ravel())
    
    # H_0 : distribution of x
    dist_x = np.zeros(10)
    np.add.at(dist_x, X_IDX.ravel(), P.ravel())
    
    # H_1 : distribution of x + y
    dist_sum = np.zeros(19)
    np.add.at(dist_sum, SUM_IDX.ravel(), P.ravel())
    
    # H_2 : distribution of x + 2y
    dist_sum2 = np.zeros(28)
    np.add.at(dist_sum2, SUM2_IDX.ravel(), P.ravel())
    
    def entropy_from_counts(counts):
        total = counts.sum()
        if total <= 1e-15:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
        
    H_diff = entropy_from_counts(dist_diff)
    H_x = entropy_from_counts(dist_x)
    H_sum = entropy_from_counts(dist_sum)
    H_sum2 = entropy_from_counts(dist_sum2)
    
    M = max(H_x, H_sum, H_sum2)
    
    if M <= 1e-9:
        return 0.0
        
    return H_diff / M

def _nelder_mead_minimize(func, x0, max_iter=1000, tol=1e-8):
    """
    Nelder-Mead algorithm.
    func: function to minimize
    x0: initial guess (1D array)
    Returns: (min_value, min_x)
    """
    n = len(x0)
    # Initialize simplex
    simplex = [x0.copy()]
    f0 = func(x0)
    simplex.append(x0 + 0.05 * np.ones(n)) # Small perturbation
    for i in range(1, n):
        simplex.append(x0.copy())
        simplex[-1][i-1] += 0.05
        
    # Evaluate initial simplex
    for i in range(n + 1):
        simplex[i] = simplex[i]
    # We need to store function values. 
    # Let's use a list of (x, f) tuples or parallel arrays.
    
    # Standard NM implementation
    points = np.zeros((n + 1, n))
    points[0] = x0
    for i in range(1, n + 1):
        points[i] = x0.copy()
        # Perturb one dimension
        dim = i - 1
        points[i][dim] += 0.05 if points[i][dim] == x0[dim] else -0.05
        
    values = np.array([func(p) for p in points])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        # Sort points by function value
        order = np.argsort(values)
        points = points[order]
        values = values[order]
        
        # Check convergence
        if np.max(values) - np.min(values) < tol:
            break
            
        # Centroid of all but the worst point
        centroid = np.mean(points[:-1], axis=0)
        worst = points[-1]
        
        # Reflection
        xr = centroid + alpha * (centroid - worst)
        fr = func(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - worst)
            fe = func(xe)
            if fe < fr:
                points[-1] = xe
                values[-1] = fe
            else:
                points[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            # Reflection accepted
            points[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            if fr < values[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - worst)
            fc = func(xc)
            
            if fc < min(fr, values[-1]):
                points[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    points[i] = points[0] + sigma * (points[i] - points[0])
                    values[i] = func(points[i])
                
    # Find best point
    best_idx = np.argmin(values)
    return values[best_idx], points[best_idx]

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600 - 15  # 15s safety margin
    
    # Parameters: log-probabilities for each of the 100 cells
    # We optimize w such that P = softmax(w)
    # To avoid issues with very large/small numbers, we can fix one parameter to 0 (gauge symmetry)
    # or just let NM handle it. NM works on the vector w.
    # However, softmax is invariant to adding a constant. 
    # Let's fix w[0,0] = 0 to break symmetry and reduce dimensionality to 99.
    
    def wrapper(w_flat):
        # w_flat is 99 elements (excluding w[0,0])
        w = np.zeros(100)
        w.flat[1:] = w_flat
        # Softmax
        w -= np.max(w)
        P = np.exp(w).reshape(10, 10)
        P /= np.sum(P)
        # Maximize ratio => Minimize negative ratio
        return -_fast_ratio(P)
        
    best_score = -np.inf
    best_w = np.zeros(99)
    
    # Random restarts
    n_restarts = 0
    while time.time() < deadline:
        # Generate random initial point
        # Scale matters. Uniform distribution corresponds to w = 0.
        # Let's try scales from 0.1 to 2.0
        scale = np.random.uniform(0.1, 2.0)
        w0 = np.random.randn(99) * scale
        
        # Run NM
        min_val, min_w = _nelder_mead_minimize(wrapper, w0, max_iter=200, tol=1e-6)
        
        score = -min_val
        if score > best_score:
            best_score = score
            best_w = min_w
            n_restarts += 1
            
        # Check time
        if time.time() > deadline:
            break
            
        # If we found a very good score, we might want to do more local search
        # But for now, just keep restarting.
        
    # Construct final P from best_w
    w = np.zeros(100)
    w.flat[1:] = best_w
    w -= np.max(w)
    P = np.exp(w).reshape(10, 10)
    P /= np.sum(P)
    
    # Final validation and clipping
    P = np.clip(P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        return np.ones((10, 10)) / 100.0
    P /= total
    
    return P
