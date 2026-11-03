"""Program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 50

# EVOLVE-BLOCK-START

def _autoconvolve(h):
    """Compute the full autoconvolution of h using FFT for speed."""
    n = len(h)
    conv_len = 2 * n - 1
    # Pad to next power of 2 for efficient FFT
    fsize = 1
    while fsize < conv_len:
        fsize <<= 1
    H = fft(h, fsize)
    conv = np.real(ifft(H * H, fsize))[:conv_len]
    return conv

def _score_raw(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -np.inf
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolve(h)
    l_inf = np.max(conv)
    if l_inf < 1e-15:
        return -np.inf
    return l2_sq / (l1 * l_inf)

def _epigraph_objective(x, h_support):
    """
    Epigraph formulation:
    Minimize t subject to:
      t >= 0
      conv_k <= 1/t for all k (i.e., t * conv_k <= 1)
      sum(h) = 1 (normalization)
    
    Actually, let's reformulate. We want to maximize:
      R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    With normalization ||h||_1 = 1, this becomes:
      R = ||h||_2^2 / ||h*h||_inf
    
    So we want to maximize ||h||_2^2 / ||h*h||_inf subject to sum(h)=1, h>=0.
    
    Equivalently, minimize ||h*h||_inf / ||h||_2^2, or set t = ||h*h||_inf and
    minimize t subject to ||h||_2^2 >= 1 (after scaling), sum(h)=1, h>=0.
    
    Let's use: minimize t such that:
      conv_k <= t for all k
      sum(h) = 1
      h >= 0
      ||h||_2^2 >= 1  (this is the objective we want to maximize, but it's nonlinear)
    
    Actually, let's just directly use the epigraph for the ratio.
    Let s = ||h||_2^2, t = ||h*h||_inf. We want to maximize s/t.
    Set u = s/t, so s >= u*t. This is bilinear, not convex.
    
    Alternative: fix the normalization sum(h)=1. Then maximize ||h||_2^2 / ||h*h||_inf.
    Let t = ||h*h||_inf. Minimize t / ||h||_2^2. This is still non-convex.
    
    Let's try a different approach: coordinate descent on the weights.
    """
    pass

def _coordinate_descent(h, max_iters=200, tol=1e-10):
    """
    Coordinate descent: for each index i, optimize h[i] while keeping others fixed.
    For a fixed h with h[i] variable, the score is:
      R = (sum_j h[j]^2) / (sum_j h[j] * max_k conv_k)
    
    Let S2 = sum_{j != i} h[j]^2, S1 = sum_{j != i} h[j]
    Then ||h||_2^2 = S2 + h[i]^2
    ||h||_1 = S1 + h[i]
    conv_k = sum_j h[j] * h[k-j] = (conv_k without h[i]) + h[i] * (sum_j h[k-j] where j != i)
    
    This is complex. Let's just do a simple 1D optimization for each coordinate.
    """
    h = np.maximum(h, 0.0).copy()
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h
    
    for iteration in range(max_iters):
        improved = False
        for i in range(N):
            # Optimize h[i] by trying a few values
            best_val = h[i]
            best_score = _score_raw(h)
            
            # Try geometric scaling
            for factor in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 0.1, 0.01]:
                h[i] = h[i] * factor if h[i] > 0 else 0.01
                score = _score_raw(h)
                if score > best_score:
                    best_score = score
                    best_val = h[i]
            
            h[i] = best_val
        
        # Check convergence
        if iteration > 0:
            # We could check if h changed significantly, but for now just run all iters
            pass
    
    return h

def _two_stage_optimization(h_init, deadline):
    """
    Two-stage block-coordinate descent:
    1. Identify the dominant autoconvolution peak index k*.
    2. For symmetric pairs (i, k*-i), solve a 1D subproblem to balance the peak.
    3. Iterate, refining the support.
    """
    h = np.maximum(h_init, 0.0).copy()
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h
    
    # Normalize
    h = h / l1
    
    for outer_iter in range(50):
        if time.time() > deadline:
            break
        
        conv = _autoconvolve(h)
        k_star = np.argmax(conv)
        peak_val = conv[k_star]
        
        if peak_val < 1e-15:
            break
        
        # For the dominant peak k*, the contribution is:
        # conv[k_star] = sum_i h[i] * h[k_star - i]
        # The pairs (i, k_star-i) contribute h[i]*h[k_star-i]
        
        # Try to reduce the peak by adjusting symmetric pairs
        # For each pair (i, k_star-i) where both are in [0, N-1]:
        changed = False
        for i in range(N):
            j = k_star - i
            if j < 0 or j >= N or j == i:
                continue
            
            # Current contribution of this pair to conv[k_star]:
            # h[i] * h[j]
            # We want to minimize h[i]*h[j] while maintaining h[i]+h[j] = const
            # to not change L1 norm too much.
            # Actually, for fixed sum s = h[i]+h[j], the product h[i]*h[j] is
            # minimized when one is 0 and the other is s.
            
            s = h[i] + h[j]
            if s < 1e-15:
                continue
            
            # Try moving weight to one side
            # Option 1: all to i
            h[i] = s
            h[j] = 0.0
            score1 = _score_raw(h)
            
            # Option 2: all to j
            h[i] = 0.0
            h[j] = s
            score2 = _score_raw(h)
            
            # Option 3: keep as is
            h[i] = h[i]  # already set
            h[j] = h[j]
            score0 = score1  # placeholder
            
            # Restore and try balanced
            h[i] = s / 2
            h[j] = s / 2
            score_bal = _score_raw(h)
            
            # Pick the best
            best_score = max(score1, score2, score_bal)
            if best_score == score1:
                h[i] = s
                h[j] = 0.0
            elif best_score == score2:
                h[i] = 0.0
                h[j] = s
            else:
                h[i] = s / 2
                h[j] = s / 2
            
            changed = True
        
        if not changed:
            break
    
    return h

def construct():
    """
    Construct 50 nonnegative heights on equal intervals of [-1/4, 1/4]
    to maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_score = -np.inf
    best_h = np.ones(N) / N  # Uniform initialization
    
    # Helper: compute the score for a given height vector
    def compute_score(h):
        return _score_raw(h)
    
    # Strategy 1: Start with various initializations and apply coordinate descent
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N) / N)
    
    # Exponential decay
    for decay in [0.5, 0.7, 0.8, 0.9]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = decay ** i
        h = h / np.sum(h)
        initializations.append(h.copy())
    
    # Sparse: only even indices
    h = np.zeros(N)
    for i in range(0, N, 2):
        h[i] = 1.0
    h = h / np.sum(h)
    initializations.append(h.copy())
    
    # Sparse: only odd indices
    h = np.zeros(N)
    for i in range(1, N, 2):
        h[i] = 1.0
    h = h / np.sum(h)
    initializations.append(h.copy())
    
    # Random sparse
    for _ in range(5):
        h = np.zeros(N)
        support = np.random.choice(N, size=25, replace=False)
        h[support] = 1.0
        h = h / np.sum(h)
        initializations.append(h.copy())
    
    # For each initialization, run two-stage optimization
    for h_init in initializations:
        if time.time() > deadline:
            break
        
        h_opt = _two_stage_optimization(h_init.copy(), deadline)
        score = compute_score(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Strategy 2: Coordinate descent on the best found so far
    if time.time() < deadline - 20:
        h_cd = _coordinate_descent(best_h.copy(), max_iters=100)
        score = compute_score(h_cd)
        if score > best_score:
            best_score = score
            best_h = h_cd.copy()
    
    # Strategy 3: SLSQP with epigraph formulation
    if time.time() < deadline - 30:
        # Minimize -log(score) or use epigraph
        # Let's try: minimize t subject to conv_k <= t for all k, sum(h)=1, h>=0
        # and maximize ||h||_2^2 (which we can fix by scaling)
        
        # Actually, let's use a simpler approach:
        # Maximize ||h||_2^2 - lambda * ||h*h||_inf for some lambda
        # This is non-smooth due to the max, but we can use subgradient methods
        
        # For now, let's try SLSQP on the epigraph form
        def epigraph_obj(x):
            h = np.maximum(x, 0.0)
            l1 = np.sum(h)
            if l1 < 1e-15:
                return 1e10
            h = h / l1
            conv = _autoconvolve(h)
            l_inf = np.max(conv)
            l2_sq = np.sum(h ** 2)
            if l_inf < 1e-15:
                return 1e10
            return -l2_sq / l_inf
        
        x0 = best_h.copy()
        try:
            result = minimize(epigraph_obj, x0, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            h_opt = np.maximum(result.x, 0.0)
            score = compute_score(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final check: ensure nonnegative and return
    best_h = np.maximum(best_h, 0.0)
    
    # If best_score is still very bad, fall back to uniform
    if best_score < 0.5:
        best_h = np.ones(N) / N
    
    return list(best_h)

# EVOLVE-BLOCK-END
