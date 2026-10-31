import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_MARGIN = 8.0  # Return 8s before the 150s kill

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    The ratio is homogeneous of degree 0, so scaling h does not change the value.
    We assume h is non-negative.
    """
    h = np.asarray(h, dtype=np.float64)
    # Enforce non-negativity for safety
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    # Normalize to sum=1 to improve numerical stability and simplify ratio
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    # Full convolution
    # Using np.convolve for exactness. N=50 is small.
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    
    if conv_inf < 1e-15:
        return 0.0
        
    return h2_sq / conv_inf

def neg_ratio_normalized(x):
    """
    Objective for SLSQP on normalized vector x (sum=1, x>=0).
    """
    h = np.maximum(x, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    h_norm = h / s
    return -compute_ratio(h_norm)

def neg_ratio_smooth(x, alpha=5000.0):
    """
    Objective using a smooth surrogate for the max function in the denominator.
    We want to maximize R = sum(h^2) / max(conv).
    This is equivalent to minimizing log(max(conv)) - log(sum(h^2)).
    Using soft-max for max(conv) makes the function smooth.
    Higher alpha = tighter approximation to true max.
    """
    h = np.maximum(x, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    h_norm = h / s
    h2_sq = np.sum(h_norm**2)
    if h2_sq < 1e-15:
        return 1e10
    
    conv = np.convolve(h_norm, h_norm, mode='full')
    
    # Soft max with log-sum-exp trick for stability
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 1e10
        
    conv_shifted = conv - max_conv
    # Use a large alpha for tight approximation
    soft_max = max_conv + np.log(np.sum(np.exp(alpha * conv_shifted)) / alpha)
    
    # Minimize log(soft_max) - log(h2_sq)
    return np.log(soft_max) - np.log(h2_sq)

def neg_ratio_smooth_adaptive(x, alpha_base=100.0, alpha_max=20000.0, warmup=10):
    """
    Adaptive alpha smooth surrogate. Start with lower alpha for broader exploration,
    then increase alpha for tighter approximation.
    """
    h = np.maximum(x, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    h_norm = h / s
    h2_sq = np.sum(h_norm**2)
    if h2_sq < 1e-15:
        return 1e10
    
    conv = np.convolve(h_norm, h_norm, mode='full')
    
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 1e10
        
    conv_shifted = conv - max_conv
    
    # Adaptive alpha: increase based on how peaked the distribution is
    # If h is more concentrated, use higher alpha
    h_max = np.max(h_norm)
    concentration = h_max  # 1/N for uniform, 1 for spike
    # Map concentration to alpha
    alpha = alpha_base + (alpha_max - alpha_base) * (concentration - 1.0/N) / (1.0 - 1.0/N)
    alpha = np.clip(alpha, alpha_base, alpha_max)
    
    soft_max = max_conv + np.log(np.sum(np.exp(alpha * conv_shifted)) / alpha)
    
    # Minimize log(soft_max) - log(h2_sq)
    return np.log(soft_max) - np.log(h2_sq)

def coordinate_descent_refine(h, max_iter=200, tol=1e-10):
    """
    Coordinate descent on the ratio. For each coordinate, try to optimize
    it while keeping others fixed. This can escape from SLSQP local minima.
    """
    h = np.maximum(h, 0.0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    best_score = compute_ratio(h)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            h_orig = h[i]
            # Try a range of values for coordinate i
            # Keep sum fixed by adjusting another coordinate
            # Simple approach: try moving mass from other coordinates
            for j in range(N):
                if j == i:
                    continue
                if h[j] < 1e-12:
                    continue
                # Try transferring some mass from j to i
                for frac in [0.01, 0.05, 0.1, 0.2, 0.5]:
                    delta = h[j] * frac
                    if delta < 1e-14:
                        continue
                    h_new = h.copy()
                    h_new[i] = h_orig + delta
                    h_new[j] = h[j] - delta
                    if h_new[j] < 0:
                        continue
                    sc = compute_ratio(h_new)
                    if sc > best_score + tol:
                        best_score = sc
                        h = h_new.copy()
                        h_orig = h[i]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try increasing h[i] and decreasing all others proportionally
                if h_orig > 1e-12:
                    for factor in [1.1, 1.5, 2.0, 3.0]:
                        h_new = h.copy()
                        h_new[i] = h_orig * factor
                        # Scale others to keep sum=1
                        others_sum = 1.0 - h_new[i]
                        if others_sum < 1e-12:
                            continue
                        others = h_new.copy()
                        others[i] = 0
                        others_sum_orig = np.sum(others)
                        if others_sum_orig > 1e-12:
                            others = others * (others_sum / others_sum_orig)
                            others[i] = h_new[i]
                            h_new = others
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                best_score = sc
                                h = h_new.copy()
                                h_orig = h[i]
                                improved = True
                                break
        if not improved:
            break
    
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    
    def elapsed():
        return time.time() - t0
        
    def time_left():
        return deadline - time.time()

    # Initial best: single spike
    best_h = np.zeros(N, dtype=np.float64)
    best_h[N // 2] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []
    
    # Helper to add candidate
    def add_candidate(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0.0)
        if np.sum(h) < 1e-15:
            return
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))
        
    # --- Phase 1: Generate Structured Candidates (reduced, focus on quality) ---
    
    # 1. Single spike (already set)
    add_candidate(best_h)
    
    # 2. Two spikes - sample key spacings
    for i in range(N):
        for j in range(i+1, N, 3):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            add_candidate(h)
            if elapsed() > deadline - 10:
                return list(best_h)
                
    # 3. Three spikes - spread out
    for i in range(0, N, 4):
        for j in range(i+4, N, 4):
            for k in range(j+4, N, 4):
                h = np.zeros(N)
                h[i] = 1.0/3
                h[j] = 1.0/3
                h[k] = 1.0/3
                add_candidate(h)
                if elapsed() > deadline - 10:
                    return list(best_h)

    # 4. Sidon-like sets - these minimize collision in convolution
    def find_sidon_set_greedy(n, start_idx=0, step=1):
        """
        Greedy construction of a Sidon set.
        """
        A = []
        used_sums = set()
        for x in range(start_idx, N, step):
            valid = True
            new_sums = []
            for a in A:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if 2 * x in used_sums:
                valid = False
            if valid:
                A.append(x)
                used_sums.update(new_sums)
                used_sums.add(2 * x)
        return A

    for step in [1, 2]:
        for start in range(0, 3, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                h = np.zeros(N)
                for idx in A:
                    h[idx] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 10:
                    return list(best_h)

    # 5. Random Sparse Sets - fewer, more diverse
    rng = np.random.default_rng(42)
    for _ in range(30):
        k = rng.integers(3, 12)
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        add_candidate(h)
        if elapsed() > deadline - 10:
            return list(best_h)

    # 6. Uniform
    add_candidate(np.ones(N))
    
    # 7. Gaussian - smooth shapes can be good starting points for refinement
    x = np.arange(N)
    for sigma in [3, 7, 15]:
        h = np.exp(-0.5 * ((x - N/2) / sigma)**2)
        add_candidate(h)

    # 8. Structured Sparse Patterns
    for start in range(0, N, 7):
        for step in range(3, 10):
            indices = list(range(start, N, step))
            if len(indices) >= 2:
                h = np.zeros(N)
                h[indices] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 10:
                    return list(best_h)

    # 9. Weighted Sidon sets with varying weights
    for step in [1, 2]:
        for start in range(0, 3, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                # Try different weight patterns
                for weight_pattern in ['uniform', 'geometric', 'bimodal']:
                    h = np.zeros(N)
                    if weight_pattern == 'uniform':
                        for idx in A:
                            h[idx] = 1.0
                    elif weight_pattern == 'geometric':
                        weights = np.geomspace(0.5, 2.0, len(A))
                        for idx, w in zip(A, weights):
                            h[idx] = w
                    elif weight_pattern == 'bimodal':
                        mid = len(A) // 2
                        for i, idx in enumerate(A):
                            h[idx] = 2.0 if i in [0, len(A)-1] else 0.5
                    add_candidate(h)
                    if elapsed() > deadline - 10:
                        return list(best_h)

    # Sort candidates by score, keep top for refinement
    candidates.sort(key=lambda x: -x[0])
    
    # --- Phase 2: Refinement using smooth surrogate first, then exact ---
    
    top_candidates = candidates[:25]
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    bounds = [(0, 1)] * N
    
    for sc, h_init in top_candidates:
        if elapsed() > deadline - 3:
            break
            
        # Normalize h_init
        h_norm_init = h_init / np.sum(h_init)
        
        # Step 2a: Use adaptive smooth surrogate for fast convergence
        try:
            res_smooth = minimize(neg_ratio_smooth_adaptive, h_norm_init, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-14})
            
            if elapsed() > deadline - 2:
                break
                
            h_opt_s = res_smooth.x
            h_opt_s = np.maximum(h_opt_s, 0.0)
            if np.sum(h_opt_s) > 1e-15:
                h_opt_s = h_opt_s / np.sum(h_opt_s)
            
            sc_opt_s = compute_ratio(h_opt_s)
            if sc_opt_s > best_score:
                best_score = sc_opt_s
                best_h = h_opt_s.copy()
                
        except Exception:
            h_opt_s = h_norm_init
            
        # Step 2b: Refine with exact objective from smooth result
        if elapsed() > deadline - 2:
            continue
        try:
            # Use the smooth result as starting point for exact objective
            start_for_exact = h_opt_s if 'h_opt_s' in locals() else h_norm_init
            res = minimize(neg_ratio_normalized, start_for_exact, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-14})
            
            if elapsed() > deadline - 1:
                break
                
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # --- Phase 3: Coordinate Descent Refinement ---
    if elapsed() < deadline - 5:
        h_cd = coordinate_descent_refine(best_h, max_iter=100, tol=1e-10)
        sc_cd = compute_ratio(h_cd)
        if sc_cd > best_score:
            best_score = sc_cd
            best_h = h_cd.copy()

    # --- Phase 4: Final Polish with Nelder-Mead on best solution ---
    if elapsed() < deadline - 2:
        h_norm_best = best_h / np.sum(best_h)
        
        # Try multiple Nelder-Mead runs with different tolerances
        for i, (maxiter, xatol, fatol) in enumerate([(2000, 1e-13, 1e-13), (1000, 1e-11, 1e-11)]):
            if elapsed() > deadline - 1:
                break
            try:
                res = minimize(neg_ratio_normalized, h_norm_best, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol})
                h_opt = res.x
                h_opt = np.maximum(h_opt, 0.0)
                if np.sum(h_opt) > 1e-15:
                    h_opt = h_opt / np.sum(h_opt)
                sc_opt = compute_ratio(h_opt)
                if sc_opt > best_score:
                    best_score = sc_opt
                    best_h = h_opt.copy()
                    h_norm_best = h_opt  # Update starting point for next NM run
            except Exception:
                pass
        
        # Also try smooth surrogate polish with adaptive alpha
        if elapsed() < deadline - 1:
            try:
                res_s = minimize(neg_ratio_smooth_adaptive, h_norm_best, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-13})
                h_opt_s = res_s.x
                h_opt_s = np.maximum(h_opt_s, 0.0)
                if np.sum(h_opt_s) > 1e-15:
                    h_opt_s = h_opt_s / np.sum(h_opt_s)
                sc_opt_s = compute_ratio(h_opt_s)
                if sc_opt_s > best_score:
                    best_score = sc_opt_s
                    best_h = h_opt_s.copy()
            except Exception:
                pass

    # --- Phase 5: Final coordinate descent polish ---
    if elapsed() < deadline - 1:
        h_final = coordinate_descent_refine(best_h, max_iter=50, tol=1e-12)
        sc_final = compute_ratio(h_final)
        if sc_final > best_score:
            best_score = sc_final
            best_h = h_final.copy()

    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
