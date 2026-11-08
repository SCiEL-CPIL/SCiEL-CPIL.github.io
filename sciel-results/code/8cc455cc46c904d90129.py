<<<<<<< SEARCH
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Try to find lower rank decompositions using optimization
    # We'll try ranks from 16 up to 27
    for target_rank in range(16, rank_28):
        if time.time() > deadline:
            break
        
        found = False
        # Number of trials per rank
        num_trials = 50 if target_rank <= 20 else 20
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Use deterministic seeding for reproducibility
            np.random.seed(42 + target_rank * 100 + trial)
            
            # Initialize with small random values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Use Adam-like optimizer
            lr = 0.02
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            max_steps = 3000 if target_rank <= 20 else 1500
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                residual = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = i * m + j
                            vj = j * p + k
                            wk = k * n + i
                            val = np.dot(U[ui, :] * V[vj, :], W[wk, :])
                            r = val - 1.0
                            residual += 0.5 * r * r
                            prod = V[vj, :] * W[wk, :]
                            grad_U[ui, :] += r * prod
                            prod2 = U[ui, :] * W[wk, :]
                            grad_V[vj, :] += r * prod2
                            prod3 = U[ui, :] * V[vj, :]
                            grad_W[wk, :] += r * prod3
                
                # Adam update
                t = step + 1
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                if residual < 1e-10:
                    found = True
                    break
            
            if found:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify
                if verify_decomposition(U_snap, V_snap, W_snap):
                    best_rank = target_rank
                    best_U = U_snap.copy()
                    best_V = V_snap.copy()
                    best_W = W_snap.copy()
                    break
        
        if found:
            break
=======
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Precompute index arrays for vectorized gradient computation
    # For each (i, j, k) triple, we need ui, vj, wk
    idx_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                idx_list.append((ui, vj, wk))
    
    idx_U = np.array([x[0] for x in idx_list])  # shape (n*m*p,)
    idx_V = np.array([x[1] for x in idx_list])  # shape (n*m*p,)
    idx_W = np.array([x[2] for x in idx_list])  # shape (n*m*p,)
    num_triples = n * m * p  # 32
    
    def compute_grads(U, V, W):
        """Vectorized gradient computation."""
        # val[t] = dot(U[idx_U[t]] * V[idx_V[t]], W[idx_W[t]])
        U_sel = U[idx_U]  # (num_triples, rank)
        V_sel = V[idx_V]  # (num_triples, rank)
        W_sel = W[idx_W]  # (num_triples, rank)
        
        prod = U_sel * V_sel  # (num_triples, rank)
        val = np.sum(prod * W_sel, axis=1)  # (num_triples,)
        r = val - 1.0  # (num_triples,)
        residual = 0.5 * np.sum(r * r)
        
        # grad_U[ui] = sum over t where idx_U[t]==ui of r[t] * V_sel[t] * W_sel[t]
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Use np.add.at for efficient accumulation
        rVW = r[:, None] * V_sel * W_sel  # (num_triples, rank)
        rUW = r[:, None] * U_sel * W_sel  # (num_triples, rank)
        rUV = r[:, None] * U_sel * V_sel  # (num_triples, rank)
        
        np.add.at(grad_U, idx_U, rVW)
        np.add.at(grad_V, idx_V, rUW)
        np.add.at(grad_W, idx_W, rUV)
        
        return grad_U, grad_V, grad_W, residual
    
    # Try to find lower rank decompositions using optimization
    # We'll try ranks from 16 up to 27
    for target_rank in range(16, rank_28):
        if time.time() > deadline:
            break
        
        found = False
        # Number of trials per rank - more trials for lower ranks
        num_trials = 30 if target_rank <= 20 else 15
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Use deterministic seeding for reproducibility
            np.random.seed(42 + target_rank * 100 + trial)
            
            # Multiple initialization strategies
            if trial % 3 == 0:
                # Small random
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            elif trial % 3 == 1:
                # Larger random
                U = np.random.randn(nm, target_rank) * 1.0
                V = np.random.randn(mp, target_rank) * 1.0
                W = np.random.randn(pn, target_rank) * 1.0
            else:
                # Initialize from rank-28 solution truncated
                U = U_28[:, :target_rank].copy()
                V = V_28[:, :target_rank].copy()
                W = W_28[:, :target_rank].copy()
            
            # Use Adam-like optimizer
            lr = 0.01
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            # Adaptive max steps based on remaining time
            remaining = deadline - time.time()
            max_steps = min(5000, int(remaining * 50)) if target_rank <= 20 else min(3000, int(remaining * 30))
            
            best_res_in_trial = float('inf')
            best_U_trial = U.copy()
            best_V_trial = V.copy()
            best_W_trial = W.copy()
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Compute gradients (vectorized)
                grad_U, grad_V, grad_W, residual = compute_grads(U, V, W)
                
                if residual < best_res_in_trial:
                    best_res_in_trial = residual
                    best_U_trial = U.copy()
                    best_V_trial = V.copy()
                    best_W_trial = W.copy()
                
                # Adam update
                t = step + 1
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                if residual < 1e-10:
                    found = True
                    break
            
            if not found and best_res_in_trial < 1e-8:
                # Very close - try polishing
                U = best_U_trial
                V = best_V_trial
                W = best_W_trial
                
                # High-precision polish
                lr2 = 0.001
                m_U2 = np.zeros_like(U)
                v_U2 = np.zeros_like(U)
                m_V2 = np.zeros_like(V)
                v_V2 = np.zeros_like(V)
                m_W2 = np.zeros_like(W)
                v_W2 = np.zeros_like(W)
                
                for step2 in range(2000):
                    if time.time() > deadline:
                        break
                    grad_U, grad_V, grad_W, residual = compute_grads(U, V, W)
                    
                    t = step2 + 1
                    m_U2 = beta1 * m_U2 + (1 - beta1) * grad_U
                    v_U2 = beta2 * v_U2 + (1 - beta2) * grad_U ** 2
                    U -= lr2 * (m_U2 / (1 - beta1 ** t)) / (np.sqrt(v_U2 / (1 - beta2 ** t)) + eps)
                    
                    m_V2 = beta1 * m_V2 + (1 - beta1) * grad_V
                    v_V2 = beta2 * v_V2 + (1 - beta2) * grad_V ** 2
                    V -= lr2 * (m_V2 / (1 - beta1 ** t)) / (np.sqrt(v_V2 / (1 - beta2 ** t)) + eps)
                    
                    m_W2 = beta1 * m_W2 + (1 - beta1) * grad_W
                    v_W2 = beta2 * v_W2 + (1 - beta2) * grad_W ** 2
                    W -= lr2 * (m_W2 / (1 - beta1 ** t)) / (np.sqrt(v_W2 / (1 - beta2 ** t)) + eps)
                    
                    if residual < 1e-12:
                        break
            
            if found or best_res_in_trial < 1e-8:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify
                if verify_decomposition(U_snap, V_snap, W_snap):
                    best_rank = target_rank
                    best_U = U_snap.copy()
                    best_V = V_snap.copy()
                    best_W = W_snap.copy()
                    found = True
                    break
        
        if found:
            break
>>>>>>> REPLACE

Now let me also improve the verification function to be vectorized for speed:

