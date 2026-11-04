<<<<<<< SEARCH
    def solve_sub_problem(max_time=30.0, target_rank=15):
        """Solve for 2x5x3 multiplication rank."""
        start = time.time()
        
        # Try target rank
        r = target_rank
        
        # Try a few random initializations
        for seed in range(20):
            if time.time() - start > max_time:
                break
            
            np.random.seed(seed)
            # Scale initialization to help convergence
            U = np.random.randn(nm_sub, r) * 0.5
            V = np.random.randn(mp_sub, r) * 0.5
            W = np.random.randn(pn_sub, r) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() - start > max_time:
                    break
                
                # Update U
                # M_U shape: (mp_sub * pn_sub, r)
                # M_U[(j, k), l] = V[j, l] * W[k, l]
                M_U = np.zeros((mp_sub * pn_sub, r))
                for j in range(m_sub):
                    for k in range(p_sub):
                        idx = j * p_sub + k
                        M_U[idx, :] = V[j, :] * W[k, :]
                
                t_U = np.ones(mp_sub * pn_sub)
                # Solve M_U @ U[a, :] = t_U for each a
                # Since M_U is same for all a, we can solve once and broadcast?
                # No, U[a, :] is the unknown.
                # M_U X = t_U.
                # X = lstsq(M_U, t_U)
                try:
                    sol_U = np.linalg.lstsq(M_U, t_U, rcond=None)[0]
                    # Check if solution is good
                    if np.allclose(M_U @ sol_U, t_U, atol=1e-8):
                        U[:, :] = sol_U
                except:
                    pass

                # Update V
                M_V = np.zeros((nm_sub * pn_sub, r))
                for i in range(n_sub):
                    for k in range(p_sub):
                        idx = i * p_sub + k
                        M_V[idx, :] = U[i, :] * W[k, :]
                
                t_V = np.ones(nm_sub * pn_sub)
                try:
                    sol_V = np.linalg.lstsq(M_V, t_V, rcond=None)[0]
                    if np.allclose(M_V @ sol_V, t_V, atol=1e-8):
                        V[:, :] = sol_V
                except:
                    pass

                # Update W
                M_W = np.zeros((nm_sub * mp_sub, r))
                for i in range(n_sub):
                    for j in range(m_sub):
                        idx = i * m_sub + j
                        M_W[idx, :] = U[i, :] * V[j, :]
                
                t_W = np.ones(nm_sub * mp_sub)
                try:
                    sol_W = np.linalg.lstsq(M_W, t_W, rcond=None)[0]
                    if np.allclose(M_W @ sol_W, t_W, atol=1e-8):
                        W[:, :] = sol_W
                except:
                    pass
                
                # Check convergence
                # Compute residual
                R_sub = np.einsum('ad,bd,cd->abc', U, V, W)
                T_sub = np.ones((nm_sub, mp_sub, pn_sub))
                # Note: T_sub indices are (i, j, k) in sub-space?
                # Our U, V, W are indexed by (a, d), (b, d), (c, d).
                # a = i*m_sub + j
                # b = j*p_sub + k
                # c = k*n_sub + i
                # So R_sub[a, b, c] should be 1.
                # Let's check the specific entries.
                max_diff = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            a = i * m_sub + j
                            b = j * p_sub + k
                            c = k * n_sub + i
                            max_diff = max(max_diff, abs(R_sub[a, b, c] - 1.0))
                
                if max_diff < 1e-6:
                    converged = True
                    break
            
            if converged:
                return U, V, W, r
        
        return None, None, None, None
=======
    def solve_sub_problem(max_time=30.0, target_rank=15):
        """Solve for 2x5x3 multiplication rank using proper ALS."""
        start = time.time()
        r = target_rank
        
        # Precompute target tensor entries for sub-problem
        target_sub = {}
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    target_sub[(a, b, c)] = 1.0
        
        # Try multiple random initializations
        for seed in range(50):
            if time.time() - start > max_time:
                break
            
            np.random.seed(seed + 1000)
            # Initialize with random values
            U = np.random.randn(nm_sub, r) * 0.3
            V = np.random.randn(mp_sub, r) * 0.3
            W = np.random.randn(pn_sub, r) * 0.3
            
            # ALS iterations with proper linear solves
            converged = False
            for it in range(300):
                if time.time() - start > max_time:
                    break
                
                # Update U: for each row a, solve min ||V^T ⊗ W^T * U[a,:] - t_U[a,:]||
                # where t_U[a, (j,k)] = 1 if (a,b,c) is in target
                for a in range(nm_sub):
                    # Build RHS: t_U[a, (j,k)] = 1 if there exists i,k such that a=i*m+j and target[(a, j*p+k, k*n+i)]=1
                    rhs = np.zeros(mp_sub * pn_sub)
                    for j in range(m_sub):
                        for k in range(p_sub):
                            idx = j * p_sub + k
                            # Check if (a, b, c) is in target for some i
                            # a = i*m + j => i = (a-j)//m if divisible
                            if (a - j) % m_sub == 0:
                                i = (a - j) // m_sub
                                b = j * p_sub + k
                                c = k * n_sub + i
                                if (a, b, c) in target_sub:
                                    rhs[idx] = 1.0
                    
                    # Build matrix: M_U[idx, d] = V[j,d] * W[k,d]
                    M_U = np.zeros((mp_sub * pn_sub, r))
                    for j in range(m_sub):
                        for k in range(p_sub):
                            idx = j * p_sub + k
                            M_U[idx, :] = V[j, :] * W[k, :]
                    
                    try:
                        sol = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                        U[a, :] = sol
                    except:
                        pass
                
                # Update V: for each row b, solve min ||U^T ⊗ W^T * V[b,:] - t_V[b,:]||
                for b in range(mp_sub):
                    rhs = np.zeros(nm_sub * pn_sub)
                    for i in range(n_sub):
                        for k in range(p_sub):
                            idx = i * p_sub + k
                            # b = j*p + k => j = (b-k)//p if divisible
                            if (b - k) % p_sub == 0:
                                j = (b - k) // p_sub
                                a = i * m_sub + j
                                c = k * n_sub + i
                                if (a, b, c) in target_sub:
                                    rhs[idx] = 1.0
                    
                    M_V = np.zeros((nm_sub * pn_sub, r))
                    for i in range(n_sub):
                        for k in range(p_sub):
                            idx = i * p_sub + k
                            M_V[idx, :] = U[i, :] * W[k, :]
                    
                    try:
                        sol = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                        V[b, :] = sol
                    except:
                        pass
                
                # Update W: for each row c, solve min ||U^T ⊗ V^T * W[c,:] - t_W[c,:]||
                for c in range(pn_sub):
                    rhs = np.zeros(nm_sub * mp_sub)
                    for i in range(n_sub):
                        for j in range(m_sub):
                            idx = i * m_sub + j
                            # c = k*n + i => k = (c-i)//n if divisible
                            if (c - i) % n_sub == 0:
                                k = (c - i) // n_sub
                                a = i * m_sub + j
                                b = j * p_sub + k
                                if (a, b, c) in target_sub:
                                    rhs[idx] = 1.0
                    
                    M_W = np.zeros((nm_sub * mp_sub, r))
                    for i in range(n_sub):
                        for j in range(m_sub):
                            idx = i * m_sub + j
                            M_W[idx, :] = U[i, :] * V[j, :]
                    
                    try:
                        sol = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                        W[c, :] = sol
                    except:
                        pass
                
                # Check convergence
                R_sub = np.einsum('ad,bd,cd->abc', U, V, W)
                max_diff = 0
                for (a, b, c), val in target_sub.items():
                    max_diff = max(max_diff, abs(R_sub[a, b, c] - val))
                
                if max_diff < 1e-8:
                    converged = True
                    break
            
            if converged:
                return U, V, W, r
        
        return None, None, None, None
>>>>>>> REPLACE

