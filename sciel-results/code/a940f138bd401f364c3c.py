import numpy as np
import time

def _score_sparse(points, weights):
    """
    Calculate the score D/M for a sparse distribution.
    points: list of (x, y) tuples.
    weights: numpy array of same length.
    """
    n = len(points)
    if n == 0:
        return 0.0
    w = np.array(weights, dtype=np.float64)
    
    # Ensure positive and normalized
    w = np.clip(w, 1e-15, None)
    sum_w = w.sum()
    if sum_w < 1e-15:
        return 0.0
    w = w / sum_w
    
    # D = H(weights) because we ensure x-y is unique for each point
    D = -np.sum(w * np.log2(w))
    
    # Extract coordinates
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    def calc_entropy(vals, probs):
        # vals: array of values
        # probs: array of probabilities corresponding to vals
        # Aggregate probs by unique vals
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        agg_probs = np.zeros(len(unique_vals))
        np.add.at(agg_probs, inverse, probs)
        agg_probs = agg_probs[agg_probs > 1e-15]
        if len(agg_probs) == 0:
            return 0.0
        return -np.sum(agg_probs * np.log2(agg_probs))
    
    Hx = calc_entropy(xs, w)
    Hxy = calc_entropy(xs + ys, w)
    Hx2y = calc_entropy(xs + 2 * ys, w)
    
    M = max(Hx, Hxy, Hx2y)
    
    if M < 1e-9:
        return 0.0
        
    return D / M

def _optimize_weights(points, time_limit=0.5):
    """
    Optimize weights for a given set of points to maximize score.
    Uses a combination of analytic initialization and local search.
    Returns (score, weights).
    """
    n = len(points)
    if n == 0:
        return 0.0, np.array([])
    
    # Start with uniform weights
    w0 = np.ones(n) / n
    current_score = _score_sparse(points, w0)
    best_score = current_score
    best_w = w0.copy()
    
    # Local search using Nelder-Mead on log-weights for better conditioning?
    # Or just simple coordinate ascent / gradient-free on weights.
    # Given the small n (<=19), Nelder-Mead is fast.
    
    from scipy.optimize import minimize
    
    def objective(w):
        w_pos = np.clip(w, 1e-15, None)
        w_norm = w_pos / w_pos.sum()
        return -_score_sparse(points, w_norm)
        
    try:
        # Run a quick optimization
        start_t = time.time()
        res = minimize(objective, w0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if time.time() - start_t < time_limit:
             w_opt = np.clip(res.x, 1e-15, None)
             w_opt = w_opt / w_opt.sum()
             score_opt = _score_sparse(points, w_opt)
             if score_opt > best_score:
                 best_score = score_opt
                 best_w = w_opt
    except Exception:
        pass
        
    return best_score, best_w

def _get_valid_x(d):
    """
    Get valid x values for diagonal d (x - y = d).
    0 <= x <= 9 and 0 <= x - d <= 9
    => d <= x <= d + 9
    => max(0, d) <= x <= min(9, d + 9)
    """
    low = max(0, d)
    high = min(9, d + 9)
    return np.arange(low, high + 1)

def arithmetic_kakeya_distribution():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    n = 10
    diags = list(range(-9, 10))
    
    # Pre-compute valid x's
    valid_xs = {d: _get_valid_x(d) for d in diags}
    
    def state_to_points(state):
        points = []
        for i, d in enumerate(diags):
            x = state[i]
            y = x - d
            points.append((x, y))
        return points

    def random_state():
        state = []
        for d in diags:
            state.append(np.random.choice(valid_xs[d]))
        return state

    # Initialize
    best_score = 0.0
    best_state = random_state()
    best_weights = None
    
    points = state_to_points(best_state)
    s, w = _optimize_weights(points)
    best_score = s
    best_weights = w
    
    # Simulated Annealing on the discrete state
    current_state = best_state.copy()
    current_weights = best_weights.copy()
    current_score = best_score
    
    T = 0.05
    T_min = 0.001
    decay = 0.9995
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a random diagonal to perturb
        idx = np.random.randint(0, len(diags))
        d = diags[idx]
        valids = valid_xs[d]
        
        if len(valids) == 1:
            continue # Cannot change
        
        old_x = current_state[idx]
        # Pick a new x different from old
        mask = valids != old_x
        if not np.any(mask):
            continue
        new_x = np.random.choice(valids[mask])
        
        current_state[idx] = new_x
        points = state_to_points(current_state)
        
        # Optimize weights for the new configuration
        new_score, new_weights = _optimize_weights(points)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or (T > T_min and np.random.rand() < np.exp(delta / T)):
            # Accept move
            current_state[idx] = new_x
            current_weights = new_weights
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_state = current_state.copy()
                best_weights = current_weights.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject move
            current_state[idx] = old_x
            no_improve_count += 1
            
        # Cool down
        T = max(T * decay, T_min)
        
        # If stuck for a long time, restart from best with some random perturbations
        if no_improve_count > 200:
            current_state = best_state.copy()
            # Perturb a few random positions
            for _ in range(3):
                idx2 = np.random.randint(0, len(diags))
                if len(valid_xs[diags[idx2]]) > 1:
                    current_state[idx2] = np.random.choice(valid_xs[diags[idx2]])
            
            points = state_to_points(current_state)
            current_score, current_weights = _optimize_weights(points)
            no_improve_count = 0
            T = 0.05 # Reset temperature
        
    # Construct final 10x10 array
    P = np.zeros((n, n))
    points = state_to_points(best_state)
    for (x, y), w in zip(points, best_weights):
        P[y, x] = w
        
    # Final validation and normalization
    P = np.clip(P, 0, None)
    total = P.sum()
    if total < 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / total
        
    return P
