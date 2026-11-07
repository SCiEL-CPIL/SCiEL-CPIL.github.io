<<<<<<< SEARCH
        # Candidate spacing strategies
        candidates = []
        
        # 1. Uniform in f1
        u_uniform = np.linspace(f1_min_val, f1_max_val, n_evals)
        candidates.append(u_uniform)
        
        # 2. Uniform in sqrt(f1)
        sqrt_u_uniform = np.linspace(np.sqrt(f1_min_val), np.sqrt(f1_max_val), n_evals)
        u_sqrt = sqrt_u_uniform ** 2
        candidates.append(u_sqrt)
        
        # 3. Uniform in f1^0.25
        q_u_uniform = np.linspace(f1_min_val**0.25, f1_max_val**0.25, n_evals)
        u_q = q_u_uniform ** 4
        candidates.append(u_q)
        
        # 4. Spacing proportional to derivative of HV contribution?
        # Marginal gain of adding a point at u is roughly (11-u) * |dv/du| * du?
        # |dv/du| = 1/(2*sqrt(u)).
        # So gain ~ (11-u)/(2*sqrt(u)).
        # This is higher for small u.
        # So we should cluster points at small u (low f1, high f2).
        # This suggests uniform in something like f1^a where a < 1?
        # Uniform in sqrt(u) is uniform in f1^0.5.
        # Let's try uniform in u^0.2.
        
        # Evaluate HV for each candidate
        best_u = None
        best_hv = -1
        for u_cand in candidates:
            hv_val = calc_hv(u_cand)
            if hv_val > best_hv:
                best_hv = hv_val
                best_u = u_cand
=======
        # Greedy iterative refinement: start with uniform spacing, then iteratively
        # move points to maximize hypervolume.
        
        # Start with uniform in f1
        u_current = np.linspace(f1_min_val, f1_max_val, n_evals)
        
        # Helper to compute HV for a set of u values (sorted ascending)
        def compute_hv(u_vals):
            u_sorted = np.sort(u_vals)
            v_vals = 1 - np.sqrt(u_sorted)
            v_prev = 11.0
            hv = 0.0
            for i in range(len(u_sorted)):
                hv += (11.0 - u_sorted[i]) * (v_prev - v_vals[i])
                v_prev = v_vals[i]
            return hv
        
        # Greedy refinement: iteratively try to improve each point
        best_hv = compute_hv(u_current)
        
        # Number of refinement iterations
        n_refine_iters = 50
        
        for refine_iter in range(n_refine_iters):
            if time.time() - start_time > deadline - 10:
                break
                
            improved = False
            for i in range(n_evals):
                if time.time() - start_time > deadline - 10:
                    break
                    
                # Try moving point i to different positions
                # Sample candidate positions from a local neighborhood and global candidates
                candidates_for_i = []
                
                # Local neighborhood: perturb current position
                u_i = u_current[i]
                perturbations = [0.001, 0.005, 0.01, 0.05, 0.1]
                for p in perturbations:
                    candidates_for_i.append(u_i - p)
                    candidates_for_i.append(u_i + p)
                
                # Global candidates: uniform and power-law spacings
                n_global = 20
                u_global_uniform = np.linspace(f1_min_val, f1_max_val, n_global)
                u_global_sqrt = np.linspace(np.sqrt(f1_min_val), np.sqrt(f1_max_val), n_global) ** 2
                u_global_025 = np.linspace(f1_min_val**0.25, f1_max_val**0.25, n_global) ** 4
                candidates_for_i.extend(u_global_uniform.tolist())
                candidates_for_i.extend(u_global_sqrt.tolist())
                candidates_for_i.extend(u_global_025.tolist())
                
                # Filter to valid range
                candidates_for_i = [c for c in candidates_for_i if f1_min_val <= c <= f1_max_val]
                
                best_candidate = u_i
                best_candidate_hv = best_hv
                
                for c in candidates_for_i:
                    # Replace point i with candidate
                    u_test = u_current.copy()
                    u_test[i] = c
                    hv_test = compute_hv(u_test)
                    if hv_test > best_candidate_hv:
                        best_candidate_hv = hv_test
                        best_candidate = c
                
                if best_candidate != u_i:
                    u_current[i] = best_candidate
                    best_hv = best_candidate_hv
                    improved = True
            
            if not improved:
                break
>>>>>>> REPLACE

