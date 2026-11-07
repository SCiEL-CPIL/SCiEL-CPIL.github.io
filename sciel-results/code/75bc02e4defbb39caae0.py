<<<<<<< SEARCH
    # Phase 3: Random perturbations around best (larger scale to explore more)
    if time.time() < deadline - 10.0:
        best_log = np.log(np.maximum(best_h, 1e-9))
        
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            
            # Random smooth perturbation with varied scale
            scale = rng.choice([0.15, 0.25, 0.35, 0.5])
            perturb = rng.randn(N)
            kernel_size = rng.choice([3, 5, 7, 9])
            kernel = np.ones(kernel_size) / kernel_size
            perturb = np.convolve(perturb, kernel, mode='same')
            perturb = perturb * scale
            
            start_log = best_log + perturb
            v_cand, h_cand = try_optimize(start_log, use_smooth=True, maxiter=2000, maxfun=10000)
            
            if h_cand is not None and v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
    
    # Phase 4: Coordinate descent refinement
    if time.time() < deadline - 8.0:
        h_cd = best_h.copy()
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for j in range(N):
                old_val = h_cd[j]
                # Try small perturbations
                for delta in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]:
                    h_trial = h_cd.copy()
                    h_trial[j] = max(0.0, old_val + delta * old_val if old_val > 0 else abs(delta))
                    v_trial = compute_value(h_trial)
                    if v_trial > best_val:
                        h_cd = h_trial
                        best_val = v_trial
                        best_h = h_trial.copy()
                        improved = True
                        break
            if not improved:
                break
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        best_log = np.log(np.maximum(best_h, 1e-9))
        v_cand, h_cand = try_optimize(best_log, use_smooth=False, maxiter=10000, maxfun=50000)
        if h_cand is not None and v_cand > best_val:
            best_val = v_cand
            best_h = h_cand.copy()
=======
    # Phase 3: Random perturbations around best (larger scale to explore more)
    if time.time() < deadline - 15.0:
        best_log = np.log(np.maximum(best_h, 1e-9))
        
        for _ in range(20):
            if time.time() > deadline - 10.0:
                break
            
            # Random smooth perturbation with varied scale
            scale = rng.choice([0.1, 0.2, 0.3, 0.4])
            perturb = rng.randn(N)
            kernel_size = rng.choice([3, 5, 7, 9])
            kernel = np.ones(kernel_size) / kernel_size
            perturb = np.convolve(perturb, kernel, mode='same')
            perturb = perturb * scale
            
            start_log = best_log + perturb
            v_cand, h_cand = try_optimize(start_log, use_smooth=True, maxiter=1500, maxfun=8000, beta=100.0)
            
            if h_cand is not None and v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
    
    # Phase 4: Coordinate descent refinement with finer steps
    if time.time() < deadline - 10.0:
        h_cd = best_h.copy()
        max_iter = 200
        for _ in range(max_iter):
            if time.time() > deadline - 6.0:
                break
            improved = False
            for j in range(N):
                old_val = h_cd[j]
                # Try small perturbations with finer granularity
                for delta in [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]:
                    h_trial = h_cd.copy()
                    h_trial[j] = max(0.0, old_val + delta * old_val if old_val > 0 else abs(delta))
                    v_trial = compute_value(h_trial)
                    if v_trial > best_val + 1e-10:
                        h_cd = h_trial
                        best_val = v_trial
                        best_h = h_trial.copy()
                        improved = True
                        break
            if not improved:
                break
    
    # Phase 5: Final high-precision polish with very high beta
    if time.time() < deadline - 5.0:
        best_log = np.log(np.maximum(best_h, 1e-9))
        v_cand, h_cand = try_optimize(best_log, use_smooth=True, maxiter=5000, maxfun=25000, beta=1000.0)
        if h_cand is not None and v_cand > best_val:
            best_val = v_cand
            best_h = h_cand.copy()
        
        # One more with direct
        if time.time() < deadline - 2.0:
            v_cand, h_cand = try_optimize(best_log, use_smooth=False, maxiter=3000, maxfun=15000)
            if h_cand is not None and v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
>>>>>>> REPLACE
