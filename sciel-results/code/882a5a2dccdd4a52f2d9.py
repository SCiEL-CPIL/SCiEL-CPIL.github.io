import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 110  # seconds, leaving margin for 120s limit
    
    N = 24
    # Dense sampling of the unit circle for objective estimation
    # 1000 points is usually sufficient to identify the peak location roughly
    # The evaluator does high-precision refinement, so we just need to find the right coefficients
    M = 2000
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    # Z[m, k] = exp(i * theta_m * k)
    # We can build this efficiently
    # Z = np.exp(1j * np.outer(thetas, np.arange(N)))
    # But we can also just compute c @ Z.T for each c
    
    # Precompute the basis for faster updates? 
    # P(z) = sum c_k z^k
    # If we flip bit k, P_new(z) = P_old(z) - 2*c_k*z^k
    # We can maintain P_values = c @ Z.T
    # But updating P_values for each flip requires vector ops anyway.
    # Given N=24, M=2000, c @ Z.T is very fast.
    
    # Let's just use the direct computation for simplicity and speed in Python loop overhead
    # We will use a simple local search with restarts.
    
    best_c = None
    best_score = np.inf
    
    # Precompute Z for objective calculation
    # Z shape (M, N)
    k_indices = np.arange(N)
    Z = np.exp(1j * np.outer(thetas, k_indices))
    
    def get_score(c):
        # c is a numpy array of shape (N,) with values 1, -1
        # P_vals = c @ Z.T -> shape (M,)
        # Note: Z.T is (N, M), so c @ Z.T is (M,)
        P_vals = c @ Z.T
        return np.max(np.abs(P_vals)) / np.sqrt(N)
    
    def local_search(c, max_iters=1000):
        c = c.copy()
        current_score = get_score(c)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try flipping each bit
            for k in range(N):
                # Flip bit k
                c_new = c.copy()
                c_new[k] = -c_new[k]
                new_score = get_score(c_new)
                if new_score < current_score - 1e-12:
                    c = c_new
                    current_score = new_score
                    improved = True
                    # Optionally break to restart the scan from 0, or continue
                    # Continuing is "best improvement" within the pass? No, this is "first improvement" if we break.
                    # Let's use "best improvement" for stability: check all, pick best.
                    # But N=24 is small, so checking all is fast.
                    # Let's stick to first improvement for speed, or best improvement for quality.
                    # Best improvement:
                    # Let's refactor to best improvement.
                    pass
        
        # Refactor to Best Improvement for better convergence
        # Actually, for such small N, we can just do a while loop that checks all bits, finds the best flip, applies it.
        
        return c, current_score
    
    # Better local search: Best Improvement
    def local_search_bi(c):
        c = c.copy()
        current_score = get_score(c)
        improved = True
        while improved:
            improved = False
            best_k = -1
            best_new_score = current_score
            for k in range(N):
                c_new = c.copy()
                c_new[k] = -c_new[k]
                new_score = get_score(c_new)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_k = k
            
            if best_k != -1 and best_new_score < current_score - 1e-12:
                c[best_k] = -c[best_k]
                current_score = best_new_score
                improved = True
        
        return c, current_score
    
    # Main loop with random restarts
    rng = np.random.default_rng(42)
    
    # Initial candidate: All 1s? Or random?
    # Let's try a few specific good starts if known, but random is safer.
    # Golay-Rudin-Shapiro is a good start too.
    
    # Generate Golay-Rudin-Shapiro
    p, q = np.array([1]), np.array([1])
    while len(p) < N:
        p_new = np.concatenate([p, q])
        q_new = np.concatenate([p, -q])
        p, q = p_new, q_new
    grs = p[:N]
    
    candidates = [grs]
    # Add some random candidates
    for _ in range(10):
        candidates.append(rng.choice([-1, 1], size=N))
        
    for c_init in candidates:
        if time.time() - start_time > time_limit:
            break
            
        c_opt, score_opt = local_search_bi(c_init)
        if score_opt < best_score:
            best_score = score_opt
            best_c = c_opt.copy()
            
        # If we find the known optimum, we can stop early?
        # The known optimum is ~1.2583758.
        # If best_score < 1.2584, it's likely optimal.
        if best_score < 1.2584:
            # Verify with a more precise check?
            # The evaluator does the precise check. We just return the coefficients.
            # We can break if we are confident.
            pass

    # If we haven't found a good one, continue with random restarts until time limit
    while time.time() - start_time < time_limit:
        c_init = rng.choice([-1, 1], size=N)
        c_opt, score_opt = local_search_bi(c_init)
        if score_opt < best_score:
            best_score = score_opt
            best_c = c_opt.copy()
            
        if best_score < 1.2584:
            # Likely found the global optimum
            # We can do a few more to be sure, or break.
            # Let's break if we hit the target.
            break

    # Ensure best_c is returned as a list of ints
    if best_c is None:
        return [1]*N
    return [int(x) for x in best_c]
