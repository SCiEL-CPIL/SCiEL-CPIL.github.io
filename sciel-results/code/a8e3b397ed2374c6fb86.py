<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation
    num_continuous_starts = 5
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 20.0:
            break
        
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        try:
            res = minimize(_objective_sum_sq, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6, 'maxls': 20})
        except Exception:
            continue
        
        h_bin = np.sign(res.x)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        if best_h is not None and time.time() < deadline - 10.0:
            time_for_sa = min(4.0, max(1.0, (deadline - 10.0 - time.time()) / max(1, num_continuous_starts - i)))
            
            h_sa, val_sa = _local_search_sa_fast(best_h.copy(), time_limit=time_for_sa)
            
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
    
    if best_h is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - 5.0 - time.time()
        if remaining_time > 2.0:
            h_final, val_final = _local_search_sa_fast(best_h.copy(), time_limit=remaining_time)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
    
    if best_h is not None and time.time() < deadline - 1.0:
        h_polish, val_polish = _greedy_polish(best_h.copy(), time_limit=min(2.0, deadline - 1.0 - time.time()))
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
        
    if best_h is None:
        return np.ones(N)
        
    return best_h
=======
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Generate diverse initializations using Gray code
    gray_inits = []
    n_gray_patterns = 8
    for i in range(n_gray_patterns):
        if time.time() >= deadline - 30.0:
            break
        # Generate Gray code-based initialization
        n_bits = max(8, N // (i + 1))  # Vary the pattern density
        init = _generate_gray_code_init(n_bits, 1)[0]
        # Add some randomness
        if np.random.random() < 0.3:
            flip_mask = np.random.random(N) < 0.1
            init[flip_mask] *= -1
        gray_inits.append(init)
    
    # Also add some random initializations
    for i in range(3):
        if time.time() >= deadline - 30.0:
            break
        gray_inits.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Phase 2: Parallel tempering search
    if time.time() < deadline - 15.0:
        pt_time = min(12.0, deadline - 15.0 - time.time())
        h_pt, val_pt = _parallel_tempering_search(gray_inits, time_limit=pt_time, n_chains=min(4, len(gray_inits)))
        if val_pt < best_val:
            best_val = val_pt
            best_h = h_pt
    
    # Phase 3: Continuous relaxation from best found
    if best_h is not None and time.time() < deadline - 10.0:
        time_for_opt = min(8.0, deadline - 10.0 - time.time())
        if time_for_opt > 2.0:
            try:
                res = minimize(_objective_sum_sq, best_h.copy(), method='L-BFGS-B', 
                               bounds=[(-1.0, 1.0)] * N, 
                               options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 30})
                h_bin = np.sign(res.x)
                zero_mask = (h_bin == 0)
                if np.any(zero_mask):
                    h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
                
                ac = _compute_autocorr(h_bin)
                val = _max_sidelobe(ac)
                
                if val < best_val:
                    best_val = val
                    best_h = h_bin.copy()
            except Exception:
                pass
    
    # Phase 4: SA refinement
    if best_h is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - 5.0 - time.time()
        if remaining_time > 2.0:
            h_final, val_final = _local_search_sa_fast(best_h.copy(), time_limit=remaining_time)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
    
    # Phase 5: Greedy polish
    if best_h is not None and time.time() < deadline - 1.0:
        h_polish, val_polish = _greedy_polish(best_h.copy(), time_limit=min(2.0, deadline - 1.0 - time.time()))
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
        
    if best_h is None:
        return np.ones(N)
        
    return best_h
>>>>>>> REPLACE
