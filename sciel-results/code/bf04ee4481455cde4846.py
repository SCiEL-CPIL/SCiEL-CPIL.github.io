import numpy as np
import time

def _calc_entropy(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

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
    
    # D = H(weights) because x-y is unique for each point in the sparse support
    # (assuming points are chosen such that x-y is distinct)
    D = _calc_entropy(w)
    
    # Extract coordinates
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    def calc_proj_entropy(vals, probs):
        # Aggregate probs by unique vals
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        agg_probs = np.zeros(len(unique_vals))
        np.add.at(agg_probs, inverse, probs)
        return _calc_entropy(agg_probs)
    
    Hx = calc_proj_entropy(xs, w)
    Hxy = calc_proj_entropy(xs + ys, w)
    Hx2y = calc_proj_entropy(xs + 2 * ys, w)
    
    M = max(Hx, Hxy, Hx2y)
    
    if M < 1e-9:
        return 0.0
        
    return D / M

def _optimize_weights(points, max_iter=100):
    """
    Optimize weights for a given set of points to maximize score.
    Returns (score, weights).
    """
    n = len(points)
    if n == 0:
        return 0.0, np.array([])
    w0 = np.ones(n) / n
    
    def objective(w):
        w_pos = np.clip(w, 1e-15, None)
        w_norm = w_pos / w_pos.sum()
        return -_score_sparse(points, w_norm)
        
    try:
        # Nelder-Mead is robust for this small non-convex problem
        from scipy.optimize import minimize
        res = minimize(objective, w0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        w_opt = np.clip(res.x, 1e-15, None)
        w_opt = w_opt / w_opt.sum()
        score = _score_sparse(points, w_opt)
        return score, w_opt
    except Exception:
        return 0.0, w0

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
    num_diags = len(diags)
    
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

    # Initialize with multiple random starts and keep the best
    best_score = 0.0
    best_state = None
    best_weights = None
    
    # Fast optimization for search steps
    FAST_ITER = 50
    # Full optimization for final polish
    FULL_ITER = 1000

    # Initial phase: Random restarts with moderate optimization
    for _ in range(10):
        if time.time() > deadline - 120:
            break
        state = random_state()
        points = state_to_points(state)
        s, w = _optimize_weights(points, max_iter=FAST_ITER)
        if s > best_score:
            best_score = s
            best_state = state.copy()
            best_weights = w.copy()
            
    if best_state is None:
        best_state = random_state()
        points = state_to_points(best_state)
        best_score, best_weights = _optimize_weights(points, max_iter=FAST_ITER)
    
    # Simulated Annealing on the discrete state
    current_state = best_state.copy()
    current_weights = best_weights.copy()
    current_score = best_score
    
    T = 0.05
    T_min = 0.001
    decay = 0.9999  # Slower cooling
    
    no_improve_count = 0
    
    while time.time() < deadline - 60:
        # Pick a random diagonal to perturb
        idx = np.random.randint(0, num_diags)
        d = diags[idx]
        valids = valid_xs[d]
        
        if len(valids) == 1:
            continue  # Cannot change
        
        old_x = current_state[idx]
        # Pick a new x different from old
        mask = valids != old_x
        if not np.any(mask):
            continue
        new_x = np.random.choice(valids[mask])
        
        candidate_state = current_state.copy()
        candidate_state[idx] = new_x
        points = state_to_points(candidate_state)
        
        # Optimize weights for the new configuration (Fast)
        new_score, new_weights = _optimize_weights(points, max_iter=FAST_ITER)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or (T > T_min and np.random.rand() < np.exp(delta / T)):
            # Accept move
            current_state = candidate_state
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
            no_improve_count += 1
            
        # Cool down
        T = max(T * decay, T_min)
        
        # If stuck for a long time, restart from best with some random perturbations
        if no_improve_count > 200:
            current_state = best_state.copy()
            # Perturb a few random positions
            num_perturb = min(5, num_diags)
            for _ in range(num_perturb):
                idx2 = np.random.randint(0, num_diags)
                if len(valid_xs[diags[idx2]]) > 1:
                    current_state[idx2] = np.random.choice(valid_xs[diags[idx2]])
            
            points = state_to_points(current_state)
            current_score, current_weights = _optimize_weights(points, max_iter=FAST_ITER)
            no_improve_count = 0
            T = 0.05  # Reset temperature
        
        # Check time more frequently to avoid overshooting
        if time.time() > deadline - 90:
            break
    
    # Final high-precision polish: re-optimize with more iterations
    if time.time() < deadline - 10:
        points = state_to_points(best_state)
        polished_score, polished_weights = _optimize_weights(points, max_iter=FULL_ITER)
        if polished_score > best_score:
            best_score = polished_score
            best_weights = polished_weights
    
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
