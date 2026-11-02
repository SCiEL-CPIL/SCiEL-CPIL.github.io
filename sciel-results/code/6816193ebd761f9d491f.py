<<<<<<< SEARCH
def _try_rank17_search(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition with enhanced strategies."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: remove each schoolbook term and perturb with varying scales
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(10):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed)
            scale = 0.01 + 0.01 * (seed % 5)
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=200)
            if result is not None:
                return result
    
    # Strategy 2: remove Strassen terms and perturb
    for term_to_remove in range(14):
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(5):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed + 50)
            scale = 0.01 + 0.01 * (seed % 5)
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=200)
            if result is not None:
                return result
    
    # Strategy 3: random initializations with varying scales
    for seed in range(30):
        if time.time() > deadline:
            break
        np.random.seed(seed * 7 + 123)
        scale = 0.3 + 0.2 * (seed % 5)
        U_init = np.random.randn(4, 17) * scale
        V_init = np.random.randn(10, 17) * scale
        W_init = np.random.randn(10, 17) * scale
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=150)
        if result is not None:
            return result
    
    # Strategy 4: Try to merge two rank-1 terms into one by finding linear dependencies
    # Check if any pair of columns in U, V, W are linearly dependent
    for i in range(17):
        if time.time() > deadline:
            break
        for j in range(i+1, 17):
            if time.time() > deadline:
                break
            # Check if columns i and j can be merged
            Ui, Uj = U18_a[:, i], U18_a[:, j]
            Vi, Vj = V18_a[:, i], V18_a[:, j]
            Wi, Wj = W18_a[:, i], W18_a[:, j]
            
            # Try to find alpha, beta such that alpha*Ui + beta*Uj is useful
            # This is heuristic; skip for now
            pass
    
    return None
=======
def _try_rank17_exact_search(n, m, p, T, deadline):
    """
    Search for a rank-17 decomposition using exact rational arithmetic.
    We generate candidate support patterns by modifying the rank-18 supports
    and solve the resulting linear system.
    """
    from fractions import Fraction
    
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Extract supports (indices of non-zero entries) from rank-18
    # Supports are lists of indices for U, V, W for each rank-1 term
    supports_U = []
    supports_V = []
    supports_W = []
    for r in range(18):
        su = [i for i in range(n*m) if U18[i][r] != 0]
        sv = [i for i in range(m*p) if V18[i][r] != 0]
        sw = [i for i in range(p*n) if W18[i][r] != 0]
        supports_U.append(su)
        supports_V.append(sv)
        supports_W.append(sw)
    
    # Get the list of non-zero entries in T: (u, v, w)
    T_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_entries.append((u, v, w))
    
    def solve_linear_system(supports_U, supports_V, supports_W, T_entries):
        """
        Solve for coefficients x_r such that sum_r x_r * U_r * V_r * W_r = T.
        U_r, V_r, W_r are determined by supports.
        We assume U_r[u]=1 if u in supports_U[r], else 0. Same for V, W.
        This is a linear system A x = b.
        A is len(T_entries) x 17.
        b is len(T_entries) x 1.
        """
        R = 17
        A = [[Fraction(0)] * R for _ in range(len(T_entries))]
        b = [Fraction(1)] * len(T_entries)
        
        for eq_idx, (u, v, w) in enumerate(T_entries):
            for r in range(R):
                if u in supports_U[r] and v in supports_V[r] and w in supports_W[r]:
                    A[eq_idx][r] = Fraction(1)
        
        # Solve A x = b using Gaussian elimination with Fractions
        # We have 10 equations and 17 unknowns. We look for a solution with small denominators.
        # We can set 7 variables to 0 and solve for the other 10?
        # Or use least squares? No, exact.
        # Let's try to find a solution where most coefficients are 0, 1, or -1.
        
        # Simple approach: iterate over subsets of 10 variables to be non-zero? Too many.
        # Instead, we can parameterize the null space.
        # Since it's small, we can just try to solve for the first 10 variables assuming the rest are 0?
        # That might not have a solution.
        
        # Let's try a heuristic: 
        # 1. Identify rows in A that are all zero. If any, no solution.
        for row in A:
            if all(val == 0 for val in row):
                return None
        
        # 2. Try to solve for a specific subset of variables.
        # We'll try to solve for variables 0..9, setting 10..16 to 0.
        # If the submatrix A[:, 0:10] is invertible, we get a unique solution.
        # We can try different subsets of 10 columns.
        
        # To keep it fast, we'll just try the first 10 columns.
        # If that fails, we'll try a few random subsets.
        
        import itertools
        
        # Try solving for columns 0-9
        A_sub = [row[:10] for row in A]
        
        def det_matrix(matrix):
            # Calculate determinant using Bareiss algorithm or similar for Fractions
            # For 10x10, standard Gaussian elimination is fine.
            n = len(matrix)
            M = [row[:] for row in matrix]
            det = Fraction(1)
            for i in range(n):
                # Find pivot
                pivot_row = -1
                for r in range(i, n):
                    if M[r][i] != 0:
                        pivot_row = r
                        break
                if pivot_row == -1:
                    return Fraction(0)
                
                if pivot_row != i:
                    M[i], M[pivot_row] = M[pivot_row], M[i]
                    det = -det
                
                det *= M[i][i]
                
                # Eliminate below
                for r in range(i+1, n):
                    if M[r][i] != 0:
                        factor = M[r][i] / M[i][i]
                        for c in range(i, n):
                            M[r][c] -= factor * M[i][c]
            
            return det
        
        det_A = det_matrix(A_sub)
        if det_A != 0:
            # Solve A_sub x = b
            # Use Cramer's rule or Gaussian elimination
            # Gaussian elimination is better.
            M = [row[:] + [b[i]] for i, row in enumerate(A_sub)]
            n = 10
            for i in range(n):
                pivot_row = -1
                for r in range(i, n):
                    if M[r][i] != 0:
                        pivot_row = r
                        break
                if pivot_row == -1:
                    return None
                if pivot_row != i:
                    M[i], M[pivot_row] = M[pivot_row], M[i]
                
                for r in range(i+1, n):
                    if M[r][i] != 0:
                        factor = M[r][i] / M[i][i]
                        for c in range(i, n+1):
                            M[r][c] -= factor * M[i][c]
            
            # Back substitution
            x = [Fraction(0)] * n
            for i in range(n-1, -1, -1):
                s = M[i][n]
                for j in range(i+1, n):
                    s -= M[i][j] * x[j]
                x[i] = s / M[i][i]
            
            # Check if solution is "simple"
            # Denominators should be small
            for val in x:
                if val.denominator > 100:
                    return None
            
            # Construct full solution
            x_full = x + [Fraction(0)] * 7
            
            # Verify
            for eq_idx, (u, v, w) in enumerate(T_entries):
                s = Fraction(0)
                for r in range(10):
                    if u in supports_U[r] and v in supports_V[r] and w in supports_W[r]:
                        s += x_full[r]
                if s != b[eq_idx]:
                    return None
            
            return x_full
        
        return None

    # Generate candidate supports
    # Base: remove one term from rank-18
    for term_to_remove in range(18):
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        base_sup_U = [supports_U[i] for i in idx]
        base_sup_V = [supports_V[i] for i in idx]
        base_sup_W = [supports_W[i] for i in idx]
        
        # Try to "fix" the system by adding one extra non-zero entry to one of the vectors
        # We iterate over which term to augment and which vector to augment
        for term_idx in range(17):
            for vec_type in ['U', 'V', 'W']:
                # Get current supports for this term
                if vec_type == 'U':
                    current_sup = base_sup_U[term_idx][:]
                    dim = n * m
                elif vec_type == 'V':
                    current_sup = base_sup_V[term_idx][:]
                    dim = m * p
                else:
                    current_sup = base_sup_W[term_idx][:]
                    dim = p * n
                
                # Try adding each possible new index
                for new_idx in range(dim):
                    if new_idx in current_sup:
                        continue
                    
                    new_sup = current_sup + [new_idx]
                    
                    # Create modified supports
                    mod_sup_U = [s[:] for s in base_sup_U]
                    mod_sup_V = [s[:] for s in base_sup_V]
                    mod_sup_W = [s[:] for s in base_sup_W]
                    
                    if vec_type == 'U':
                        mod_sup_U[term_idx] = new_sup
                    elif vec_type == 'V':
                        mod_sup_V[term_idx] = new_sup
                    else:
                        mod_sup_W[term_idx] = new_sup
                    
                    sol = solve_linear_system(mod_sup_U, mod_sup_V, mod_sup_W, T_entries)
                    
                    if sol is not None:
                        # Construct U, V, W
                        U = [[0]*17 for _ in range(n*m)]
                        V = [[0]*17 for _ in range(m*p)]
                        W = [[0]*17 for _ in range(p*n)]
                        
                        for r in range(17):
                            for u in mod_sup_U[r]:
                                U[u][r] = float(sol[r])
                            for v in mod_sup_V[r]:
                                V[v][r] = float(sol[r])
                            for w in mod_sup_W[r]:
                                W[w][r] = float(sol[r])
                        
                        # Check with float
                        if _check_decomposition(U, V, W, T):
                            return U, V, W
                        
                        # If float check fails, try to return exact fractions if possible
                        # But the interface expects numbers.
                        # We can return the float values if they are close to integers/halves
                        U_out = [[0]*17 for _ in range(n*m)]
                        V_out = [[0]*17 for _ in range(m*p)]
                        W_out = [[0]*17 for _ in range(p*n)]
                        
                        for r in range(17):
                            val = sol[r]
                            val_f = float(val)
                            # Snap to nearest 0.5 or 1
                            snapped = round(val_f * 2) / 2
                            if abs(val_f - snapped) < 1e-9:
                                val_f = snapped
                                
                            for u in mod_sup_U[r]:
                                U_out[u][r] = val_f
                            for v in mod_sup_V[r]:
                                V_out[v][r] = val_f
                            for w in mod_sup_W[r]:
                                W_out[w][r] = val_f
                                
                        if _check_decomposition(U_out, V_out, W_out, T):
                            return U_out, V_out, W_out
                            
    return None
>>>>>>> REPLACE

Update `construct` to use the new search function.

