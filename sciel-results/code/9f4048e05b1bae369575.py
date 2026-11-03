<<<<<<< SEARCH
def _try_structured_search(n, m, p, deadline):
    """
    Search for lower-rank decompositions using structured approaches.
    Returns (U, V, W) if found, else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Strategy: Try to find a decomposition by solving a linear system
    # For rank r, we have r*(nm+mp+pn) unknowns and nm*mp*pn equations
    # But the tensor is sparse, so we only need to satisfy the nonzero entries
    
    # Let's try a randomized approach:
    # 1. Initialize with random vectors
    # 2. Use gradient descent or iterative projection to satisfy constraints
    
    np.random.seed(42)
    
    for target_rank in range(119, 80, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Use iterative method: for each target entry, adjust the vectors
            # to make the product closer to 1
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current values for all target entries
                max_error = 0
                for (a, b, c) in target_entries:
                    current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    error = 1.0 - current_val
                    max_error = max(max_error, abs(error))
                
                if max_error < 1e-6:
                    break
                
                # Gradient step: for each (a,b,c), the gradient w.r.t. U[a,:] is V[b,:]*W[c,:]
                # Update: U[a,:] += learning_rate * error * V[b,:]*W[c,:]
                lr = 0.01
                for (a, b, c) in target_entries:
                    current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    error = 1.0 - current_val
                    
                    # Gradient for U[a,:]
                    grad_U = V[b, :] * W[c, :]
                    U[a, :] += lr * error * grad_U
                    
                    # Gradient for V[b,:]
                    grad_V = U[a, :] * W[c, :]
                    V[b, :] += lr * error * grad_V
                    
                    # Gradient for W[c,:]
                    grad_W = U[a, :] * V[b, :]
                    W[c, :] += lr * error * grad_W
                
                # Also need to ensure invalid entries are 0
                # This is harder - we need to add constraints
                # For now, let's just check if we're making progress
    
    return None
=======
def _try_structured_search(n, m, p, deadline):
    """
    Search for lower-rank decompositions using structured approaches.
    Returns (U, V, W) if found, else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Build target tensor entries (valid positions)
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Build invalid entries (positions that must be 0)
    invalid_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Add all other (b', c') pairs for this (a) that are not the valid one
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k2 in range(p):
                        b2 = j2 * p + k2
                        # c must satisfy: c = k2 * n + i2 where i2 is from a
                        # Actually, let's just enumerate all invalid (b,c) for each a
                # Simpler: enumerate all (b,c) and mark valid ones
            # For each a, collect all (b,c) where the entry should be 0
    # Let's rebuild invalid entries more carefully
    valid_set = set()
    for (a, b, c) in target_entries:
        valid_set.add((a, b, c))
    
    invalid_entries = []
    for a in range(nm):
        i = a // m
        j = a % m
        for b in range(mp):
            j2 = b // p
            k = b % p
            for c in range(pn):
                k2 = c // n
                i2 = c % n
                if (a, b, c) in valid_set:
                    continue
                # Check if this is truly invalid (i != i2 or k != k2)
                if i != i2 or k != k2:
                    invalid_entries.append((a, b, c))
    
    np.random.seed(42)
    
    # Try ranks from just below schoolbook down
    for target_rank in range(schoolbook_rank - 1, max(80, schoolbook_rank - 15), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Use Adam-like optimization
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2 = 0.9, 0.999
            t = 0
            lr = 0.05
            
            for iteration in range(300):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                # For valid entries, we want sum(U[a]*V[b]*W[c]) = 1
                for (a, b, c) in target_entries:
                    current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    error = 1.0 - current_val
                    gU[a, :] += error * (V[b, :] * W[c, :])
                    gV[b, :] += error * (U[a, :] * W[c, :])
                    gW[c, :] += error * (U[a, :] * V[b, :])
                
                # For invalid entries, we want sum(U[a]*V[b]*W[c]) = 0
                # Use a smaller weight for these
                for (a, b, c) in invalid_entries:
                    current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    gU[a, :] += 0.1 * current_val * (V[b, :] * W[c, :])
                    gV[b, :] += 0.1 * current_val * (U[a, :] * W[c, :])
                    gW[c, :] += 0.1 * current_val * (U[a, :] * V[b, :])
                
                # Adam update
                t += 1
                m_U = beta1 * m_U + (1 - beta1) * gU
                v_U = beta2 * v_U + (1 - beta2) * gU ** 2
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + 1e-8)
                
                m_V = beta1 * m_V + (1 - beta1) * gV
                v_V = beta2 * v_V + (1 - beta2) * gV ** 2
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + 1e-8)
                
                m_W = beta1 * m_W + (1 - beta1) * gW
                v_W = beta2 * v_W + (1 - beta2) * gW ** 2
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + 1e-8)
                
                # Check convergence every 50 iterations
                if iteration % 50 == 49:
                    max_valid_error = 0
                    for (a, b, c) in target_entries:
                        current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        max_valid_error = max(max_valid_error, abs(1.0 - current_val))
                    max_invalid_error = 0
                    for (a, b, c) in invalid_entries:
                        current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        max_invalid_error = max(max_invalid_error, abs(current_val))
                    
                    if max_valid_error < 1e-6 and max_invalid_error < 1e-6:
                        # Verify fully
                        if _validate_decomposition(n, m, p, U, V, W):
                            return U.tolist(), V.tolist(), W.tolist()
    
    return None
>>>>>>> REPLACE

