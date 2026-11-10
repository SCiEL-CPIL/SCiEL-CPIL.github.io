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
    P is a 10x10 array.
    """
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

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600 - 30  # 30s safety margin
    
    # Strategy:
    # The current program uses Nelder-Mead on 99 dimensions. This is very slow and prone to getting stuck.
    # The "inspiration" suggests structured search.
    # A powerful structured family is P[y, x] = R[y] * C[x].
    # This reduces the problem to optimizing 10 + 10 = 20 parameters (minus 1 for scale invariance, so 19).
    # Even better, we can use a parametric form: R[y] = exp(a*y), C[x] = exp(b*x).
    # This is a 2-parameter family: P[y, x] = exp(a*y + b*x).
    # This corresponds to a geometric distribution on the grid.
    # Let's optimize a and b using gradient ascent (or Nelder-Mead in 2D, which is very fast).
    # Actually, let's use a slightly richer family: R[y] = exp(a*y + c*y^2), C[x] = exp(b*x + d*x^2).
    # 4 parameters. Still very low dimension.
    
    # Let's try a 4-parameter quadratic exponential family:
    # log P[y, x] = a*y + b*x + c*y^2 + d*x^2
    # Normalize P.
    
    y_vals = np.arange(10)
    x_vals = np.arange(10)
    Y, X = np.meshgrid(y_vals, x_vals, indexing='ij')
    
    def compute_P(params):
        a, b, c, d = params
        logP = a*Y + b*X + c*Y**2 + d*X**2
        # Subtract max for stability
        logP -= np.max(logP)
        P = np.exp(logP)
        P /= np.sum(P)
        return P
    
    def objective(params):
        P = compute_P(params)
        return -_fast_ratio(P)
    
    best_score = -np.inf
    best_params = np.zeros(4)
    
    # Use Nelder-Mead on the 4 parameters.
    # We can do many restarts because it's fast.
    
    def nelder_mead_2d(f, x0, max_iter=500):
        # Simple NM for low dimensions
        n = len(x0)
        simplex = [x0.copy()]
        # Initialize simplex
        for i in range(n):
            pt = x0.copy()
            pt[i] += 0.1
            simplex.append(pt)
        
        values = [f(pt) for pt in simplex]
        
        for it in range(max_iter):
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            if values[-1] - values[0] < 1e-10:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            # Reflection
            xr = centroid + (centroid - worst)
            fr = f(xr)
            
            if fr < values[0]:
                # Expansion
                xe = centroid + 2 * (centroid - worst)
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    values[-1] = fe
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                simplex[-1] = xr
                values[-1] = fr
            else:
                # Contraction
                xc = centroid + 0.5 * (worst - centroid)
                fc = f(xc)
                if fc < values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        values[i] = f(simplex[i])
                        
        best_idx = np.argmin(values)
        return values[best_idx], simplex[best_idx]
    
    n_restarts = 0
    while time.time() < deadline:
        # Random restart in 4D space
        # Scale parameters: a, b in [-1, 1], c, d in [-0.5, 0.5]
        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-1, 1)
        c = np.random.uniform(-0.5, 0.5)
        d = np.random.uniform(-0.5, 0.5)
        
        x0 = np.array([a, b, c, d])
        
        min_val, min_x = nelder_mead_2d(objective, x0, max_iter=200)
        
        score = -min_val
        if score > best_score:
            best_score = score
            best_params = min_x
            n_restarts += 1
            
        # Also try some structured initial points
        # Uniform: a=b=c=d=0
        # Corner: a=1, b=1, c=0, d=0
        # etc.
        
        if time.time() > deadline:
            break
            
    # If we have time, try a richer family:
    # P[y, x] = exp(a*y + b*x + e*y*x)
    # This adds interaction term.
    
    def compute_P_rich(params):
        a, b, c, d, e = params
        logP = a*Y + b*X + c*Y**2 + d*X**2 + e*Y*X
        logP -= np.max(logP)
        P = np.exp(logP)
        P /= np.sum(P)
        return P
        
    def objective_rich(params):
        P = compute_P_rich(params)
        return -_fast_ratio(P)
        
    if time.time() < deadline - 60:
        while time.time() < deadline - 60:
            a = np.random.uniform(-1, 1)
            b = np.random.uniform(-1, 1)
            c = np.random.uniform(-0.5, 0.5)
            d = np.random.uniform(-0.5, 0.5)
            e = np.random.uniform(-0.5, 0.5)
            
            x0 = np.array([a, b, c, d, e])
            
            min_val, min_x = nelder_mead_2d(objective_rich, x0, max_iter=200)
            
            score = -min_val
            if score > best_score:
                best_score = score
                # We need to store the type of parameterization used
                # For simplicity, let's just stick to the 4-param one for final P construction 
                # unless we find a much better one, in which case we need to handle both.
                # Let's just keep the best P directly.
                best_P = compute_P_rich(min_x)
                # Update best_params to indicate rich model? 
                # Or just store best_P.
                # Let's refactor to store best_P.
                
    # Refactor: Store best P directly
    # Re-run to capture best_P properly
    
    best_P = None
    best_score = -np.inf
    
    while time.time() < deadline:
        # 4-param model
        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-1, 1)
        c = np.random.uniform(-0.5, 0.5)
        d = np.random.uniform(-0.5, 0.5)
        x0 = np.array([a, b, c, d])
        min_val, min_x = nelder_mead_2d(objective, x0, max_iter=100)
        score = -min_val
        if score > best_score:
            best_score = score
            best_P = compute_P(min_x)
            
        if time.time() > deadline - 30:
            break
            
        # 5-param model
        e = np.random.uniform(-0.5, 0.5)
        x0_rich = np.array([a, b, c, d, e])
        min_val_rich, min_x_rich = nelder_mead_2d(objective_rich, x0_rich, max_iter=100)
        score_rich = -min_val_rich
        if score_rich > best_score:
            best_score = score_rich
            best_P = compute_P_rich(min_x_rich)
            
    if best_P is None:
        return np.ones((10, 10)) / 100.0
        
    P = np.clip(best_P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        return np.ones((10, 10)) / 100.0
    P /= total
    
    return P
