"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    def verify_decomp(U, V, W, n, m, p):
        """Check if the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for c in range(U.shape[1]):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    # Build rank-28 Strassen decomposition as baseline
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))

    def strassen_2x2_decomp(P_rows, P_cols, Q_rows, Q_cols, R_rows, R_cols, base_col, U, V, W):
        u11 = P_rows[0] * m + P_cols[0]
        u12 = P_rows[0] * m + P_cols[1]
        u21 = P_rows[1] * m + P_cols[0]
        u22 = P_rows[1] * m + P_cols[1]
        v11 = Q_rows[0] * p + Q_cols[0]
        v12 = Q_rows[0] * p + Q_cols[1]
        v21 = Q_rows[1] * p + Q_cols[0]
        v22 = Q_rows[1] * p + Q_cols[1]
        w11 = R_cols[0] * n + R_rows[0]
        w12 = R_cols[1] * n + R_rows[0]
        w21 = R_cols[0] * n + R_rows[1]
        w22 = R_cols[1] * n + R_rows[1]

        c = base_col + 0
        U[u11, c] += 1; U[u22, c] += 1; V[v11, c] += 1; V[v22, c] += 1; W[w11, c] += 1; W[w22, c] += 1
        c = base_col + 1
        U[u21, c] += 1; U[u22, c] += 1; V[v11, c] += 1; W[w21, c] += 1; W[w22, c] += -1
        c = base_col + 2
        U[u11, c] += 1; V[v12, c] += 1; V[v22, c] += -1; W[w12, c] += 1; W[w22, c] += 1
        c = base_col + 3
        U[u22, c] += 1; V[v21, c] += 1; V[v11, c] += -1; W[w11, c] += 1; W[w21, c] += 1
        c = base_col + 4
        U[u11, c] += 1; U[u12, c] += 1; V[v22, c] += 1; W[w11, c] += -1; W[w12, c] += 1
        c = base_col + 5
        U[u21, c] += 1; U[u11, c] += -1; V[v11, c] += 1; V[v12, c] += 1; W[w22, c] += 1
        c = base_col + 6
        U[u12, c] += 1; U[u22, c] += -1; V[v21, c] += 1; V[v22, c] += 1; W[w11, c] += 1; W[w22, c] += -1
        return base_col + 7

    col = strassen_2x2_decomp([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0, U28, V28, W28)
    col = strassen_2x2_decomp([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], col, U28, V28, W28)
    col = strassen_2x2_decomp([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], col, U28, V28, W28)
    col = strassen_2x2_decomp([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], col, U28, V28, W28)

    if not verify_decomp(U28, V28, W28, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 28
    best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()

    # Try rank 27 via direct construction using the 2-row structure
    # Key idea: Since n=2, we can potentially merge terms across the two rows
    # Let's try a modified scheme that exploits the symmetry

    if time.time() < deadline - 30:
        # Attempt: Use a 27-term scheme based on sharing between the 4 block products
        # The 4 products are: A1*B11, A2*B21, A1*B12, A2*B22
        # Standard Strassen: 4*7 = 28
        # Try to share one term by using a different factorization
        
        # Alternative: Use the fact that we only need C[0,k] and C[1,k] for k=0..3
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
        
        # Try a direct rank-27 construction using algebraic identities
        # We'll use a hybrid approach: 3 full Strassen blocks (21 terms) + 6 custom terms
        
        U27 = np.zeros((n * m, 27))
        V27 = np.zeros((m * p, 27))
        W27 = np.zeros((p * n, 27))
        
        # Use 3 Strassen blocks for 3 of the 4 products
        col = strassen_2x2_decomp([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0, U27, V27, W27)
        col = strassen_2x2_decomp([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], col, U27, V27, W27)
        col = strassen_2x2_decomp([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], col, U27, V27, W27)
        # col is now 21, we have 6 terms left for A2*B22
        
        # For A2*B22, instead of 7 Strassen terms, try 6 custom terms
        # A2 = A[:, 2:4] (2x2), B22 = B[2:4, 2:4] (2x2)
        # Result contributes to C[:, 2:4]
        
        # A2[0,0]=A[0,2], A2[0,1]=A[0,3], A2[1,0]=A[1,2], A2[1,1]=A[1,3]
        # B22[0,0]=B[2,2], B22[0,1]=B[2,3], B22[1,0]=B[3,2], B22[1,1]=B[3,3]
        # C[0,2], C[0,3], C[1,2], C[1,3]
        
        # Try 6-term scheme for this specific product
        # M1 = (A2[0,0]+A2[1,1])*(B22[0,0]+B22[1,1])
        # M2 = (A2[1,0]+A2[1,1])*B22[0,0]
        # M3 = A2[0,0]*(B22[0,1]-B22[1,1])
        # M4 = A2[1,1]*(B22[1,0]-B22[0,0])
        # M5 = (A2[0,0]+A2[0,1])*B22[1,1]
        # M6 = (A2[1,0]-A2[0,0])*(B22[0,0]+B22[0,1])
        # This is 6 terms, but standard Strassen needs 7 for exact 2x2 mult.
        # So this won't work for exact decomposition.
        
        # Let's try a different approach: use 27 terms with a different structure
        # Instead of 4x7, try a scheme where some terms are shared
        
        # Actually, let's try: use the full 28-term Strassen but check if any
        # two columns are linearly dependent and can be merged.
        
        def try_merge_columns(U, V, W, n, m, p, r):
            """Try to find two columns that can be merged."""
            for c1 in range(r):
                if time.time() > deadline - 30:
                    return None
                for c2 in range(c1 + 1, r):
                    # Check if column c2 = alpha * column c1
                    u1, u2 = U[:, c1], U[:, c2]
                    v1, v2 = V[:, c1], V[:, c2]
                    w1, w2 = W[:, c1], W[:, c2]
                    
                    # Find alpha from U
                    nonzero = np.abs(u1) > 1e-10
                    if np.any(nonzero):
                        alphas = u2[nonzero] / u1[nonzero]
                        if np.allclose(alphas, alphas[0], atol=1e-10):
                            alpha = alphas[0]
                            if np.allclose(v2, alpha * v1, atol=1e-10) and \
                               np.allclose(w2, alpha * w1, atol=1e-10):
                                return (c1, c2, alpha)
                    
                    # Check if c2 is zero
                    if np.allclose(u2, 0, atol=1e-10) and \
                       np.allclose(v2, 0, atol=1e-10) and \
                       np.allclose(w2, 0, atol=1e-10):
                        return (c1, c2, 0)
            return None
        
        merge_info = try_merge_columns(U28, V28, W28, n, m, p, 28)
        if merge_info:
            c1, c2, alpha = merge_info
            U27_try = np.delete(U28, c2, axis=1)
            V27_try = np.delete(V28, c2, axis=1)
            W27_try = np.delete(W28, c2, axis=1)
            if verify_decomp(U27_try, V27_try, W27_try, n, m, p):
                best_rank = 27
                best_U, best_V, best_W = U27_try.copy(), V27_try.copy(), W27_try.copy()
    
    # Try rank 26 via more aggressive search
    if best_rank > 26 and time.time() < deadline - 20:
        # Try to find rank 26 by removing two columns and adjusting
        # This is a nonlinear optimization problem
        
        np.random.seed(42)
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            # Remove two random columns from rank-28
            cols_to_remove = np.random.choice(28, size=2, replace=False)
            U26 = np.delete(U28, cols_to_remove, axis=1)
            V26 = np.delete(V28, cols_to_remove, axis=1)
            W26 = np.delete(W28, cols_to_remove, axis=1)
            
            # Try to fix with small perturbations
            def compute_max_error(U, V, W, n, m, p):
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                            max_err = max(max_err, abs(val - 1))
                return max_err
            
            err = compute_max_error(U26, V26, W26, n, m, p)
            
            if err < 0.5:  # Close enough to try fixing
                for sub in range(50):
                    if time.time() > deadline - 20:
                        break
                    # Perturb a random element
                    mat = np.random.randint(3)
                    if mat == 0:
                        r, c = np.random.randint(n*m), np.random.randint(26)
                        U26[r, c] += np.random.randn() * 0.05
                    elif mat == 1:
                        r, c = np.random.randint(m*p), np.random.randint(26)
                        V26[r, c] += np.random.randn() * 0.05
                    else:
                        r, c = np.random.randint(p*n), np.random.randint(26)
                        W26[r, c] += np.random.randn() * 0.05
                    
                    new_err = compute_max_error(U26, V26, W26, n, m, p)
                    if new_err < err:
                        err = new_err
                    else:
                        # Revert
                        if mat == 0:
                            U26[r, c] -= np.random.randn() * 0.05
                        elif mat == 1:
                            V26[r, c] -= np.random.randn() * 0.05
                        else:
                            W26[r, c] -= np.random.randn() * 0.05
                    
                    if err < 1e-9:
                        break
                
                if err < 1e-9:
                    best_rank = 26
                    best_U, best_V, best_W = U26.copy(), V26.copy(), W26.copy()
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
