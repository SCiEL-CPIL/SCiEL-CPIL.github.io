<<<<<<< SEARCH
def _search_sub_decomposition(n, m, p, max_rank, coeff_set, timeout=60):
    """Exhaustive search for rank-minimal decomposition of n x m x p tensor.

    coeff_set: list of allowed coefficient values.
    Returns (rank, U_list, V_list, W_list) or None if not found within max_rank.
    """
    import time
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n

    # For a given rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1

    # Strategy: fix U and V, then W is determined by linear equations.
    # For each (k, i), W[k*n+i, c] appears in equations for all j.
    # Actually, let's use a different approach: enumerate small coefficient matrices.

    # For small ranks, we can enumerate U and V entries from coeff_set,
    # then solve for W.

    for r in range(1, max_rank + 1):
        if time.time() - start_time > timeout:
            return None

        # Enumerate U: nm x r matrix with entries from coeff_set
        # This is huge, so we need a smarter approach.
        # Instead, use the fact that for rank-1: U[i*m+j,0]=a_ij, V[j*p+k,0]=b_jk, W[k*n+i,0]=c_ki
        # and a_ij * b_jk * c_ki = 1 for all i,j,k.

        # General approach: random search with verification
        # But for small p (2 or 3), we can be more systematic.

        # Let's try: fix U entries, fix V entries, solve for W via linear system.
        # For fixed U and V, the equations for W are:
        # For each (k, i): sum_j [sum_c U[i*m+j,c] * V[j*p+k,c]] * W[k*n+i,c] = 1
        # Wait, that's not right. The equation is:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for ALL i,j,k
        # This is a system where W entries are variables.

        # Actually, for fixed U and V, the tensor equation becomes:
        # For each (i,j,k): sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        # Let A_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c]. Then:
        # sum_c A_c[i,j,k] * W[k*n+i,c] = 1
        # This is a linear system in the W variables. There are p*n unknowns (W entries per column)
        # but they appear coupled across columns c.

        # Better: treat each column c independently? No, W[k*n+i,c] is specific to column c.
        # The equation for a specific (i,j,k) is:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is one equation involving all r W-entries for index (k,i).

        # Let's set up the linear system: variables are W[k*n+i,c] for all k,i,c.
        # Total variables: p*n*r
        # Equations: n*m*p (one for each (i,j,k))

        # For small r, we can enumerate U and V from a small set and solve.
        # But nm*r and mp*r can be large. Let's limit coefficient set size.

        # Alternative: use the structure. For the sub-problems with small p,
        # the known optimal ranks are:
        # 3x4x2: rank 8 (it's known that m(3,4,2) = 8)
        # 3x4x3: rank 12 (m(3,4,3) = 12? Let me think... m(3,3,3)=23, m(4,4,4)=49)
        # Actually for 3x4x2, the schoolbook rank is 24, but it can be reduced.

        # Let me use a randomized approach with verification.
        best = None
        num_trials = 5000
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
            # Random U, V with small integer entries
            U_rand = np.random.choice(coeff_set, size=(nm, r))
            V_rand = np.random.choice(coeff_set, size=(mp, r))

            # Set up linear system for W
            # Variables: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
            # Number of variables: p * n * r
            # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1

            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A[eq_idx, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq_idx += 1

            try:
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is exact
                if np.allclose(A @ W_flat, b, atol=1e-8):
                    W_rand = W_flat.reshape((p * n, r))
                    # Verify
                    if _check_decomposition(U_rand, V_rand, W_rand, n, m, p):
                        # Check entries are "nice" (integer or half-integer)
                        W_rounded = np.round(W_rand * 2) / 2
                        if np.allclose(W_rounded, W_rand, atol=1e-8):
                            U_list = U_rand.tolist()
                            V_list = V_rand.tolist()
                            W_list = W_rounded.tolist()
                            if best is None or r < best[0]:
                                best = (r, U_list, V_list, W_list)
            except:
                pass

        if best is not None:
            return best

    return None
=======
def _search_sub_decomposition(n, m, p, max_rank, coeff_set, timeout=60):
    """Exhaustive search for rank-minimal decomposition of n x m x p tensor.

    coeff_set: list of allowed coefficient values.
    Returns (rank, U_list, V_list, W_list) or None if not found within max_rank.
    """
    import time
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n

    # For a given rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1

    # Strategy: fix U and V, then W is determined by linear equations.
    for r in range(1, max_rank + 1):
        if time.time() - start_time > timeout:
            return None

        best = None
        num_trials = 20000
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
            # Random U, V with small integer entries
            # Use a sparse random matrix to increase chance of full rank and solvability
            U_rand = np.random.choice(coeff_set, size=(nm, r))
            V_rand = np.random.choice(coeff_set, size=(mp, r))
            
            # Ensure U and V are not all zero
            if np.all(U_rand == 0) or np.all(V_rand == 0):
                continue

            # Set up linear system for W
            # Variables: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
            # Number of variables: p * n * r
            # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1

            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A[eq_idx, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq_idx += 1

            try:
                # Use lstsq to find minimum norm solution
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is exact
                if np.allclose(A @ W_flat, b, atol=1e-8):
                    W_rand = W_flat.reshape((p * n, r))
                    # Verify
                    if _check_decomposition(U_rand, V_rand, W_rand, n, m, p):
                        # Check entries are "nice" (integer or half-integer)
                        W_rounded = np.round(W_rand * 2) / 2
                        if np.allclose(W_rounded, W_rand, atol=1e-8):
                            U_list = U_rand.tolist()
                            V_list = V_rand.tolist()
                            W_list = W_rounded.tolist()
                            if best is None or r < best[0]:
                                best = (r, U_list, V_list, W_list)
            except:
                pass

        if best is not None:
            return best

    return None
>>>>>>> REPLACE

