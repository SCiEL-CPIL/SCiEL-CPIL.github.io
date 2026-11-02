<<<<<<< SEARCH
def _try_rank54_exploit_n2(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition by exploiting the n=2 structure.
    
    Key insight: For n=2, both output rows i=0 and i=1 share the same B matrix.
    The constraints are:
      For each (j,k): sum_c U[0*m+j,c]*V[j*p+k,c]*W[k*n+0,c] = 1
      For each (j,k): sum_c U[1*m+j,c]*V[j*p+k,c]*W[k*n+1,c] = 1
    
    We can try to construct U and V with structure that allows both rows to
    be satisfied simultaneously with rank 54 (one less than 55).
    
    Strategy: Start from the rank-55 Strassen construction and try to merge
    one pair of columns by finding a linear combination that satisfies both rows.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    # Start from rank-55 and try to reduce by 1
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r55 = 55
    r54 = 54
    
    # Try to find if we can merge two columns of the rank-55 solution
    # by finding a linear combination that works
    # This is a hard problem, so we try random perturbations
    
    best = None
    
    # Approach 1: Try random U,V of rank 54 and solve for W
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        # Initialize U and V with structure similar to Strassen
        U = np.zeros((n*m, r54))
        V = np.zeros((m*p, r54))
        
        # Use the Strassen pattern but with rank 54
        # Try different sparsity patterns
        strat = attempt % 4
        
        if strat == 0:
            # Sparse random
            for row in range(n*m):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                U[row, cols] = rng.choice([-1, 0, 1], size=nnz)
            for row in range(m*p):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                V[row, cols] = rng.choice([-1, 0, 1], size=nnz)
                
        elif strat == 1:
            # Based on Strassen structure with one column removed
            U[:, :r54] = U0[:, :r54] * rng.randint(0, 2, size=(n*m, r54))
            V[:, :r54] = V0[:, :r54] * rng.randint(0, 2, size=(m*p, r54))
            # Add some randomness
            U += rng.randint(-1, 2, size=(n*m, r54)) * 0.1
            V += rng.randint(-1, 2, size=(m*p, r54)) * 0.1
            
        elif strat == 2:
            # Random ±1 with moderate sparsity
            U = rng.randint(0, 2, size=(n*m, r54)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r54)) * 2 - 1
            U[rng.random((n*m, r54)) < 0.3] = 0
            V[rng.random((m*p, r54)) < 0.3] = 0
            
        else:
            # Block structure
            block_size = r54 // 5
            for g in range(5):
                start = g * block_size
                end = start + block_size if g < 4 else r54
                if end > start:
                    U[:, start:end] = rng.randint(0, 2, size=(n*m, end-start)) * 2 - 1
                    V[:, start:end] = rng.randint(0, 2, size=(m*p, end-start)) * 2 - 1
                    U[:, start:end][rng.random((n*m, end-start)) < 0.4] = 0
                    V[:, start:end][rng.random((m*p, end-start)) < 0.4] = 0
        
        # Solve for W
        W = np.zeros((p*n, r54))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r54))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ sol, b, atol=1e-10):
                        feasible = False
                        break
                    W[k*n+i, :] = sol
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            # Verify full solution
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best
=======
def _try_rank54_structured(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition using structured patterns derived from Strassen.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r54 = 54
    best = None
    
    # Strategy: Try removing one column from the rank-55 solution and solving for the rest?
    # No, the other columns are fixed. We need to find a new set of 54 columns.
    
    # We will try a set of structured modifications to the Strassen U and V.
    # We assume the rank-54 solution is close to the rank-55 solution.
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
            
        # Create U and V by taking the first 54 columns of U0, V0
        # and perturbing them slightly or permuting.
        # A better idea: The rank-54 solution might be obtained by merging two Strassen blocks.
        
        U = U0[:, :r54].copy()
        V = V0[:, :r54].copy()
        
        # Apply a random sparse perturbation to U and V
        # This allows us to explore the neighborhood of the Strassen solution
        if attempt % 2 == 0:
             # Perturb U
             mask = rng.random((n*m, r54)) < 0.05
             U[mask] += rng.choice([-1, 1], size=np.sum(mask))
        else:
             # Perturb V
             mask = rng.random((m*p, r54)) < 0.05
             V[mask] += rng.choice([-1, 1], size=np.sum(mask))
        
        # Solve for W
        W = np.zeros((p*n, r54))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r54))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ sol, b, atol=1e-8):
                        feasible = False
                        break
                    W[k*n+i, :] = sol
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            # Check if W is close to integers/half-integers
            W_rounded = np.round(W * 2) / 2
            if np.allclose(W, W_rounded, atol=1e-5):
                # Verify with rounded W
                ok = True
                for i in range(n):
                    if not ok: break
                    for j in range(m):
                        if not ok: break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W_rounded[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-5:
                                ok = False
                                break
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W_rounded.tolist()}
                    break
    
    return best
>>>>>>> REPLACE

