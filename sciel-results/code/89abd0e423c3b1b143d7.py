import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Build the multiplication tensor T[a, b, c]
    # a = i*m + j (i: row of A, j: col of A = row of B)
    # b = j*p + k (j: row of B, k: col of B = row of C)
    # c = k*n + i (k: row of C, i: col of C = row of A)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Schoolbook decomposition as fallback (rank = 45)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        T_recon = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_recon, T, atol=tol)
    
    def try_als(r, seed, U_init=None, V_init=None, W_init=None, max_iters=300):
        """Alternating least squares for rank r decomposition."""
        np.random.seed(seed)
        if U_init is not None:
            U = U_init.copy()
        else:
            U = np.random.randn(n * m, r) * 0.1
        if V_init is not None:
            V = V_init.copy()
        else:
            V = np.random.randn(m * p, r) * 0.1
        if W_init is not None:
            W = W_init.copy()
        else:
            W = np.random.randn(p * n, r) * 0.1
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Update U
            TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (m*p, p*n, r)
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (n*m, r)
            G_U = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            
            try:
                U_new = np.linalg.lstsq(G_U, h_U.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None
            
            # Update V
            UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (n*m, p*n, r)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (m*p, r)
            G_V = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            
            try:
                V_new = np.linalg.lstsq(G_V, h_V.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None
            
            # Update W
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (n*m, m*p, r)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (p*n, r)
            G_W = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            
            try:
                W_new = np.linalg.lstsq(G_W, h_W.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None
            
            U, V, W = U_new, V_new, W_new
            
            # Calculate error
            T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < 1e-10:
                break
                
            if err > prev_err * 1.1:
                break
            prev_err = err
        
        # Try rounding to half-integers
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return (U_r, V_r, W_r)
        
        return None
    
    def get_strassen_3x3x3():
        """Get Strassen's rank-7 decomposition for 3x3x3 matrix multiplication.
        
        Returns U, V, W matrices of shape (9, 7) each for the 3x3x3 problem.
        """
        # Strassen's 7 products for 3x3x3
        # Standard Strassen for 2x2 extended to 3x3
        # For 3x3, we can use the known rank-7 decomposition
        
        # One known rank-7 decomposition for 3x3x3:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A01-A11)(B10+B11)  
        # M3 = (A00+A01)B10
        # M4 = A11(B01-B00)
        # M5 = (A10+A11)B00
        # M6 = (A00-A10)(B10+B11)
        # M7 = (A01+A10)(B00-B11)
        
        # But for 3x3, we need a proper rank-7 or higher decomposition
        # Actually, the known minimal rank for 3x3x3 is 19 (not 7)
        # Strassen's method gives rank 27 for 3x3, not 7
        # The rank-7 is only for 2x2x2
        
        # For 3x3x3, let's use a structured initialization instead
        # We'll create a partial structure based on the 2x2x2 Strassen blocks
        
        U = np.zeros((9, 7))
        V = np.zeros((9, 7))
        W = np.zeros((9, 7))
        
        # 2x2x2 Strassen core (top-left 2x2 block)
        # A indices: 0,1,2 (row), 3,4,5 (col) -> but for 3x3: a = i*3+j
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        # A[2,0]=6, A[2,1]=7, A[2,2]=8
        
        # B[0,0]=0, B[0,1]=1, B[0,2]=2
        # B[1,0]=3, B[1,1]=4, B[1,2]=5
        # B[2,0]=6, B[2,1]=7, B[2,2]=8
        
        # C[0,0]=0, C[0,1]=1, C[0,2]=2
        # C[1,0]=3, C[1,1]=4, C[1,2]=5
        # C[2,0]=6, C[2,1]=7, C[2,2]=8
        
        # For the 2x2x2 Strassen (using A[0:2,0:2], B[0:2,0:2], C[0:2,0:2]):
        # M1 = (A00+A11)(B00+B11) -> A indices {0,4}, B indices {0,4}, C indices {0,4}
        # M2 = (A01-A11)(B10+B11) -> A indices {1,4}, B indices {3,4}, C indices {1,4}
        # M3 = (A00+A01)B10 -> A indices {0,1}, B indices {3}, C indices {2,5}
        # M4 = A11(B01-B00) -> A indices {4}, B indices {1,0}, C indices {5,2}
        # M5 = (A10+A11)B00 -> A indices {3,4}, B indices {0}, C indices {3,7}
        # M6 = (A00-A10)(B10+B11) -> A indices {0,3}, B indices {3,4}, C indices {0,3}
        # M7 = (A01+A10)(B00-B11) -> A indices {1,3}, B indices {0,4}, C indices {1,7}
        
        # This is getting complicated. Let's just use random initialization
        # with a bias toward small values to help convergence.
        
        return None
    
    # Try ranks from 19 up to schoolbook
    # Theoretical lower bound for 3x3x5 is at least 19
    ranks_to_try = list(range(19, rank_sb + 1))
    
    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000, 1001, 1002, 1003, 1004, 1005, 2024, 2025, 2026]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            break
        
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            
            # Use small random initialization
            result = try_als(r, seed)
            if result is not None:
                U_f, V_f, W_f = result
                if r < best_rank:
                    best_rank = r
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found valid for this rank, move to next rank
    
    return best_result
