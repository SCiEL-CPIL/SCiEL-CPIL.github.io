import numpy as np
import time

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    Uses vectorized operations with sort-and-search.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # Pairwise sums S_ij = s_i + s_j
    S = s[:, None] + s[None, :]
    # Thresholds T_kl = 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]
    # Pairwise weights W_ij = w_i * w_j
    W = w[:, None] * w[None, :]
    
    # Flatten for sorting and search
    S_flat = S.flatten()
    T_flat = T.flatten()
    W_flat = W.flatten()
    
    # Sort S_flat and compute cumulative weights
    order_S = np.argsort(S_flat, kind='stable')
    S_sorted = S_flat[order_S]
    W_sorted_S = W_flat[order_S]
    cum_W_S = np.cumsum(W_sorted_S)
    
    # For each T_flat, find how many S_sorted are strictly less than T_flat
    idxs_F = np.searchsorted(S_sorted, T_flat, side='left')
    
    # Create padded cumulative sum: [0, cum_W_S[0], cum_W_S[1], ...]
    padded_cum_W_S = np.concatenate(([0.0], cum_W_S))
    
    # F_flat[i] = sum of W for S < T_i
    F_flat = padded_cum_W_S[idxs_F]
    F = F_flat.reshape(n, n)
    
    P = np.sum(W * F)
    return float(P)

def _validate_and_normalize(s, w):
    s = np.asarray(s, dtype=np.float64).flatten()
    w = np.asarray(weights, dtype=np.float64).flatten()
    
    # Remove NaN/Inf
    mask = np.isfinite(s) & np.isfinite(w)
    s = s[mask]
    w = w[mask]
    
    if len(s) == 0:
        return None, None
        
    # Ensure s >= 0
    s = np.maximum(s, 0.0)
    
    # Ensure w >= 0
    w = np.maximum(w, 0.0)
    
    total_w = w.sum()
    if total_w <= 1e-9:
        return None, None
        
    # Sort by support
    sort_idx = np.argsort(s)
    s = s[sort_idx]
    w = w[sort_idx]
    
    # Merge duplicate support points
    unique_s, unique_indices, counts = np.unique(s, return_inverse=True, return_counts=True)
    if len(unique_s) < len(s):
        w_unique = np.zeros(len(unique_s))
        np.add.at(w_unique, unique_indices, w)
        s = unique_s
        w = w_unique
        total_w = w.sum()
        if total_w <= 1e-9:
            return None, None
        w = w / total_w
    else:
        w = w / total_w
        
    return s, w

def beat_the_average_measure() -> tuple:
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])
    best_score = _compute_P_fast(best_support, best_weights)
    
    def try_update(s, w):
        nonlocal best_support, best_weights, best_score
        if time.time() > deadline:
            return
        s, w = _validate_and_normalize(s, w)
        if s is None:
            return
        score = _compute_P_fast(s, w)
        if score > best_score:
            best_score = score
            best_support = s
            best_weights = w
    
    # 1. Binary {0, 1}: P(p) = p^2(1-p)(3-2p)
    # Optimize p in [0, 1]
    def obj_binary(p):
        return -p**2 * (1-p) * (3-2*p)
    
    # Golden section search for p
    phi = (1 + np.sqrt(5)) / 2
    a, b = 0.0, 1.0
    for _ in range(100):
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        if obj_binary(c) < obj_binary(d):
            a = c
        else:
            b = d
    p_opt = (a + b) / 2
    s_bin = np.array([0.0, 1.0])
    w_bin = np.array([p_opt, 1-p_opt])
    try_update(s_bin, w_bin)
    
    # 2. Mixture p * delta_0 + (1-p) * U[0, A]
    n_grid = 200
    def eval_mixture_uniform(p, A):
        grid = np.linspace(0, A, n_grid)
        w_grid = np.full(n_grid, (1-p)/n_grid)
        w_grid[0] += p
        return grid, w_grid
    
    # Optimize p for several A values
    for A in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        def obj_p_for_A(p):
            s, w = eval_mixture_uniform(p, A)
            return -_compute_P_fast(s, w)
        
        # Golden section search for p
        a, b = 0.0, 1.0
        for _ in range(100):
            c = b - (b - a) / phi
            d = a + (b - a) / phi
            if obj_p_for_A(c) < obj_p_for_A(d):
                a = c
            else:
                b = d
        p_opt = (a + b) / 2
        s_m, w_m = eval_mixture_uniform(p_opt, A)
        try_update(s_m, w_m)
    
    # 3. 3-point: p0*delta_0 + p1*delta_1 + p2*delta_A
    def eval_3point(p0, p1, A):
        p2 = 1 - p0 - p1
        if p0 < 0 or p1 < 0 or p2 < 0:
            return None, None
        s = np.array([0.0, 1.0, A])
        w = np.array([p0, p1, p2])
        return s, w
    
    # Grid search for A, then optimize p0, p1
    for A in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        # For each A, do a 2D grid search over p0, p1
        best_local_score = -1
        best_local_p0, best_local_p1 = 0.5, 0.25
        for p0 in np.linspace(0.1, 0.8, 20):
            for p1 in np.linspace(0.05, 0.6, 15):
                if p0 + p1 > 1:
                    continue
                s, w = eval_3point(p0, p1, A)
                if s is None:
                    continue
                score = _compute_P_fast(s, w)
                if score > best_local_score:
                    best_local_score = score
                    best_local_p0, best_local_p1 = p0, p1
        s_m, w_m = eval_3point(best_local_p0, best_local_p1, A)
        if s_m is not None:
            try_update(s_m, w_m)
    
    # 4. Fixed grid with optimized weights
    n_grid_opt = 50
    s_grid_opt = np.linspace(0, 2.0, n_grid_opt)
    
    def _project_to_simplex(v):
        n = len(v)
        if n == 0:
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        w = np.maximum(v - theta, 0)
        return w
    
    # Random restarts with local search
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        w_start = np.random.dirichlet(np.ones(n_grid_opt))
        # Simple hill climbing
        w_curr = w_start.copy()
        for step in range(50):
            if time.time() > deadline:
                break
            # Perturb
            w_pert = w_curr + np.random.normal(0, 0.05, n_grid_opt)
            w_pert = _project_to_simplex(w_pert)
            s, w_norm = _validate_and_normalize(s_grid_opt, w_pert)
            if s is None:
                continue
            score = _compute_P_fast(s, w_norm)
            score_curr = _compute_P_fast(s_grid_opt, w_curr)
            if score > score_curr:
                w_curr = w_pert
        s_m, w_m = _validate_and_normalize(s_grid_opt, w_curr)
        if s_m is not None:
            try_update(s_m, w_m)
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)
