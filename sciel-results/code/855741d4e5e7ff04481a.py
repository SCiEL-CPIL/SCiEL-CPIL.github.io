    # ... after projection ...
    
    def get_R(s):
        return get_autocorr(s.astype(float))

    R = get_R(best_s)
    max_val = np.max(np.abs(R))
    best_max = max_val
    best_s = best_s.copy()
    
    t_limit_ls = start_time + DEADLINE - 5 # 5s safety
    
    while time.time() < t_limit_ls:
        # Find peak lag
        abs_R = np.abs(R)
        k_star = np.argmax(abs_R) + 1 # lag index 1..N-1
        
        # Find bits contributing to R(k_star)
        # R(k) = sum_{j=0}^{N-1-k} s_j s_{j+k}
        # Contribution of s_i:
        # If i < N-k, s_i is paired with s_{i+k}
        # If i >= k, s_i is paired with s_{i-k}
        
        # We want to flip s_i to reduce |R(k_star)|
        # New R(k) = R(k) - 2 s_i (s_{i-k} + s_{i+k}) (with bounds)
        # Let delta_i = -2 s_i (s_{i-k} + s_{i+k})
        # New |R(k)| = |R(k) + delta_i|
        
        # We can compute potential deltas for all i
        # But flipping one bit changes ALL R(k).
        # So we must check the *global* max after flip.
        
        # Strategy: Try flipping the bit with the largest potential reduction in the current peak.
        # Calculate delta for all i for the current peak lag k_star.
        
        s_curr = best_s
        # Compute terms
        # term[i] = s_{i-k} + s_{i+k}
        term = np.zeros(N)
        # i-k valid if i>=k
        term[k:] += s_curr[:N-k]
        # i+k valid if i < N-k
        term[:N-k] += s_curr[k:]
        
        # delta_i = -2 s_i term[i]
        deltas = -2 * s_curr * term
        
        # We want to minimize |R(k_star) + delta_i|
        # But we also care about other lags.
        # Heuristic: Pick i that minimizes |R(k_star) + delta_i|
        
        target = R[k_star-1]
        new_vals = np.abs(target + deltas)
        i_candidate = np.argmin(new_vals)
        
        # Try flipping i_candidate
        s_try = s_curr.copy()
        s_try[i_candidate] *= -1
        
        R_try = get_R(s_try)
        new_max = np.max(np.abs(R_try))
        
        if new_max < best_max - 1e-10:
            best_s = s_try
            best_max = new_max
            R = R_try
        else:
            # If no improvement, maybe try next best candidate?
            # Or just break to avoid getting stuck?
            # Let's try a few more candidates from the top list
            # To keep it simple, we'll just break or continue with random?
            # Let's just continue the loop, it will pick the same candidate again if R doesn't change?
            # No, R doesn't change if we don't flip.
            # So we will be stuck.
            # Solution: If no improvement, pick a random bit instead.
            
            if np.random.rand() < 0.1:
                 # Random escape
                 i_rand = np.random.randint(N)
                 s_try = s_curr.copy()
                 s_try[i_rand] *= -1
                 R_try = get_R(s_try)
                 new_max = np.max(np.abs(R_try))
                 if new_max < best_max:
                     best_s = s_try
                     best_max = new_max
                     R = R_try
                 else:
                     # Stuck, break?
                     # Let's add a counter
                     pass
            else:
                 # Stuck on this peak, maybe try to reduce a different peak?
                 # For simplicity, break if we've tried this candidate and it failed?
                 # We need a way to know we tried it.
                 # Let's just use random flips if greedy fails.
                 pass
            
            # To prevent infinite loop on same candidate:
            # We can shuffle the candidate list.
            # But for simplicity and time, let's just do random flips if greedy fails.
            
    return best_s
