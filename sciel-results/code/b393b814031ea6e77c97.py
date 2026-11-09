def _search_rank(rank, time_limit):
    n, m, p = DIMENSIONS
    rows_u = n * m
    rows_v = m * p
    rows_w = p * n
    
    end_time = time.time() + time_limit
    rng = np.random.default_rng()
    
    # Precompute the list of all triples (u, v, w) and their targets
    triples = []
    for u in range(rows_u):
        i, j = divmod(u, m)
        for v in range(rows_v):
            j2, k = divmod(v, p)
            if j != j2:
                # Inconsistent in j
                for w in range(rows_w):
                    triples.append((u, v, w, 0))
            else:
                # Consistent in j
                for w in range(rows_w):
                    k2, i2 = divmod(w, n)
                    if k2 == k and i2 == i:
                        triples.append((u, v, w, 1))
                    else:
                        triples.append((u, v, w, 0))
                        
    # We only need to solve for the inconsistent ones?
    # No, we need to satisfy all.
    # But the consistent ones are 42. Inconsistent are 1722.
    # We can set up the matrix A for the linear system.
    # A has shape (1764, rows_w * rank)
    # This is large but manageable.
    
    # Actually, we can just check the error after solving.
    # To speed up, we can use a subset of equations?
    # No, we need exact solution.
    
    while time.time() < end_time:
        # Random U, V
        U = np.random.randint(-2, 3, size=(rows_u, rank)) # integers
        V = np.random.randint(-2, 3, size=(rows_v, rank))
        
        # Build linear system for W
        # Variables: W_flat of size rows_w * rank
        # Eq: sum_r U[u,r] V[v,r] W[w,r] = target
        # Note: W[w,r] is variable index w*rank + r
        
        # We can use numpy.linalg.lstsq
        # But building the full 1764 x 336 matrix every time might be slow?
        # 1764 * 336 ~ 600k floats. 1764 * 336 * 8 bytes ~ 5MB.
        # Solving 1764x336 is fast.
        
        # Optimization: Only use a subset of inconsistent equations for the search?
        # If we find a W that satisfies all, we are done.
        
        A_mat = np.zeros((len(triples), rows_w * rank))
        b_vec = np.zeros(len(triples))
        
        for idx, (u, v, w, target) in enumerate(triples):
            b_vec[idx] = target
            for r in range(rank):
                # Coefficient for W[w, r]
                A_mat[idx, w * rank + r] = U[u, r] * V[v, r]
                
        # Solve A_mat W_flat = b_vec
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W = W_flat.reshape(rows_w, rank)
            
            # Check exactness
            # The residual from lstsq is approximate. We need exact integer/half-integer check.
            # Since U, V are integers, W from lstsq might be float.
            # We need W to be half-integers?
            # The task says "integer/half-integer entries".
            # If the solution is rational, we can scale?
            # But the rank is fixed.
            
            # Check if W is close to half-integers
            # And if the product is exactly 1/0.
            
            # Calculate error exactly
            error = 0
            for idx, (u, v, w, target) in enumerate(triples):
                s = 0
                for r in range(rank):
                    s += U[u, r] * V[v, r] * W[w, r]
                if abs(s - target) > 1e-9:
                    error += 1
                    break
            if error == 0:
                # Convert to Fractions
                # Check if W entries are half-integers
                W_int = (W * 2).round()
                if np.allclose(W, W_int / 2.0):
                    Wf = [[Fraction(int(W_int[i, r])) / 2 for r in range(rank)] for i in range(rows_w)]
                    Uf = [[Fraction(int(U[i, r])) for r in range(rank)] for i in range(rows_u)]
                    Vf = [[Fraction(int(V[i, r])) for r in range(rank)] for i in range(rows_v)]
                    return {"U": Uf, "V": Vf, "W": Wf}
        except np.linalg.LinAlgError:
            continue
            
    return None
