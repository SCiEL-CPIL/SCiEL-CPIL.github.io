import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import warnings

N = 50

def compute_ratio(h):
    """Compute ||h||^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return -1.0
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    conv_max = np.max(conv)
    if conv_max <= 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    best_score = 0.0
    best_h = None

    def update_best(h_candidate):
        nonlocal best_score, best_h
        if h_candidate is None:
            return
        h_candidate = np.asarray(h_candidate, dtype=float)
        if h_candidate.shape != (N,):
            return
        if np.any(h_candidate < -1e-12):
            return
        h_candidate = np.maximum(h_candidate, 0.0)
        if np.sum(h_candidate) <= 0:
            return
        s = compute_ratio(h_candidate)
        if s > best_score:
            best_score = s
            best_h = h_candidate.copy()

    # --- Phase 1: Sparse Support Search via Differential Evolution ---
    # The hypothesis is that the optimal solution might have a small support.
    # We search for the best weights on a fixed support of size K.
    
    def sparse_objective(w):
        """
        w: vector of length K, non-negative weights.
        We normalize w to sum to 1 inside the objective for scale invariance.
        """
        w = np.asarray(w, dtype=float)
        if np.any(w < 0):
            return 1e10
        s = np.sum(w)
        if s <= 1e-15:
            return 1e10
        w = w / s
        
        q = np.dot(w, w)
        if q <= 1e-15:
            return 1e10
            
        conv = np.convolve(w, w, mode='full')
        m = np.max(conv)
        
        if m <= 1e-15:
            return 1e10
            
        # Minimize M / Q
        return m / q

    # Try supports of various sizes
    # We fix the support to the first K indices for simplicity in DE, 
    # but we can also try random supports if time permits.
    # Given the 150s limit, let's stick to a few K values with DE.
    
    for K in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline - 20.0:
            break
            
        bounds = [(0, 1)] * K
        
        try:
            # Use DE to find the best weights for this support size
            # popsize=15, maxiter=50 is usually fast enough for small K
            res = differential_evolution(sparse_objective, bounds, 
                                         maxiter=50, popsize=15, 
                                         tol=1e-8, seed=42, 
                                         polish=False) # Polish can be slow, we'll polish later
            if res.success:
                w_opt = res.x
                s = np.sum(w_opt)
                if s > 0:
                    w_opt = w_opt / s
                    # Construct full vector
                    h_candidate = np.zeros(N)
                    h_candidate[:K] = w_opt
                    update_best(h_candidate)
        except Exception:
            pass

    # --- Phase 2: Local Refinement using Soft-Max Objective ---
    # If we found a good sparse solution, we can use it as a seed for a full-vector
    # optimization with a smooth approximation of the max operator.
    # This allows "spreading" the support slightly if it improves the ratio.
    
    def objective_softmax(w, beta=100.0):
        """
        w: vector of length N.
        v = softmax(w)
        Maximize ||v||^2 / (||v||_1 * soft_max(conv(v), beta))
        Since v is normalized to sum 1, ||v||_1 = 1.
        Minimize negative of ratio.
        """
        # Stable softmax
        w_max = np.max(w)
        v = np.exp(w - w_max)
        v_sum = np.sum(v)
        if v_sum <= 0:
            return 0.0
        v = v / v_sum
        
        l2_sq = np.dot(v, v)
        conv = np.convolve(v, v, mode='full')
        
        max_conv = np.max(conv)
        if max_conv <= 0:
            return 0.0
        
        # Soft max
        shifted = beta * (conv - max_conv)
        # Clip to avoid overflow in exp, though shifted is <= 0
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        if sum_exp == 0:
            return 0.0
        log_sum_exp = max_conv + np.log(sum_exp) / beta
        
        ratio = l2_sq / log_sum_exp
        if ratio <= 0:
            return 0.0
        return -ratio

    # Seed for local optimization
    if best_h is not None:
        # Convert best_h to log-space (with small epsilon for zeros)
        w_seed = np.log(np.maximum(best_h, 1e-15))
        
        # Try a few different betas for the soft-max approximation
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 5.0:
                break
            
            try:
                # Use L-BFGS-B for local refinement
                res = minimize(objective_softmax, w_seed, method='L-BFGS-B',
                               args=(beta,),
                               options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
                
                w_opt = res.x
                w_max = np.max(w_opt)
                v_opt = np.exp(w_opt - w_max)
                v_opt = v_opt / np.sum(v_opt)
                
                update_best(v_opt)
            except Exception:
                pass

    # --- Phase 3: Randomized Full-Vector Search ---
    # If the sparse search didn't find anything great, or just to explore more,
    # try a few random starts in the full N-dimensional space.
    
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Random sparse-like initialization
            w_rand = np.zeros(N)
            # Pick 3-5 random indices
            k = np.random.randint(3, 6)
            idx = np.random.choice(N, k, replace=False)
            w_rand[idx] = np.random.rand(k) + 0.1
            w_rand = np.log(w_rand)
            
            try:
                res = minimize(objective_softmax, w_rand, method='L-BFGS-B',
                               args=(100.0,),
                               options={'maxiter': 100, 'gtol': 1e-8})
                w_opt = res.x
                w_max = np.max(w_opt)
                v_opt = np.exp(w_opt - w_max)
                v_opt = v_opt / np.sum(v_opt)
                update_best(v_opt)
            except Exception:
                pass

    # --- Finalize ---
    if best_h is not None:
        # Scale the result. The ratio is scale-invariant, so any scale works.
        # Using 1e-6 ensures numerical stability if the evaluator is sensitive to magnitude.
        h_out = best_h * 1e-6
        h_out = np.maximum(h_out, 0.0)
        return h_out.tolist()
    else:
        # Fallback: Uniform
        h_out = np.ones(N) * 1e-6
        return h_out.tolist()
