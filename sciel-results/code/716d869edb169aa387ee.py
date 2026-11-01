import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)

    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-8)

    def mat_to_list(mat):
        return mat.tolist()

    def round_to_half(mat):
        return np.round(mat * 2) / 2.0

    def try_lm_rank(r):
        """
        Attempt to find a rank r decomposition using Levenberg-Marquardt.
        """
        if time.time() > deadline - 10:
            return None

        # Initialize with a structured pattern:
        # Use Strassen-like blocks for the first 2 columns of B, 
        # and schoolbook for the rest, then pad/truncate to rank r.
        
        # Strassen rank 7 for 2x2x2
        # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
        # We need to map this to the global indices.
        
        U_init = np.zeros((n*m, r))
        V_init = np.zeros((m*p, r))
        W_init = np.zeros((p*n, r))
        
        # Fill first 7 columns with Strassen for j=0,1
        # Strassen terms for 2x2x2:
        # c=0: (A00+A11)(B00+B11) -> U: A00,A11; V: B00,B11; W: C00,C11
        # c=1: (A10+A11)B00 -> U: A10,A11; V: B00; W: C10
        # c=2: A00(B10-B11) -> U: A00; V: B10,B11; W: C01
        # c=3: A11(B01-B00) -> U: A11; V: B01,B00; W: C11
        # c=4: (A00+A10)B11 -> U: A00,A10; V: B11; W: C01
        # c=5: (A10-A00)(B00+B01) -> U: A10,A00; V: B00,B01; W: C10,C11
        # c=6: (A01-A11)(B10+B11) -> U: A01,A11; V: B10,B11; W: C01,C11
        
        # Indices: U[i*m+k], V[k*p+j], W[j*n+i]
        
        # c=0
        U_init[0, 0] = 1; U_init[3, 0] = 1
        V_init[0, 0] = 1; V_init[1, 0] = 1
        W_init[0, 0] = 1; W_init[2, 0] = 1
        
        # c=1
        U_init[2, 1] = 1; U_init[3, 1] = 1
        V_init[0, 1] = 1
        W_init[1, 1] = 1
        
        # c=2
        U_init[0, 2] = 1
        V_init[2, 2] = 1; V_init[3, 2] = -1
        W_init[1, 2] = 1
        
        # c=3
        U_init[3, 3] = 1
        V_init[1, 3] = 1; V_init[0, 3] = -1
        W_init[2, 3] = 1
        
        # c=4
        U_init[0, 4] = 1; U_init[2, 4] = 1
        V_init[3, 4] = 1
        W_init[1, 4] = 1
        
        # c=5
        U_init[2, 5] = 1; U_init[0, 5] = -1
        V_init[0, 5] = 1; V_init[1, 5] = 1
        W_init[1, 5] = 1; W_init[3, 5] = 1
        
        # c=6
        U_init[1, 6] = 1; U_init[3, 6] = -1
        V_init[2, 6] = 1; V_init[3, 6] = 1
        W_init[1, 6] = 1; W_init[3, 6] = 1
        
        # For j>=2, use schoolbook terms if r allows
        # Schoolbook term for (i,k,j) is at index i*m+k*p+j? No, index is just sequential.
        # In schoolbook, col = i*m*k + k*p + j? No, col increments for every i,k,j.
        # Let's just add schoolbook terms for the remaining capacity.
        
        col = 7
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    if col >= r:
                        break
                    # Only add if not already covered by Strassen (j<2)
                    if j >= 2:
                        U_init[i*m + k, col] = 1
                        V_init[k*p + j, col] = 1
                        W_init[j*n + i, col] = 1
                        col += 1
            if col >= r:
                break
            if col >= r:
                break
        
        # If r < 28, we have to be more clever or the initialization is poor.
        # But let's try LM anyway.
        
        def residuals(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W = x[n*m*r + m*p*r : ].reshape(p*n, r)
            
            u_r = U.T.reshape(n, m, r)
            v_r = V.T.reshape(m, p, r)
            w_r = W.T.reshape(p, n, r)
            T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
            return (T - 1.0).ravel()

        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=10000)
            if sol.cost < 1e-12:
                U = sol.x[:n*m*r].reshape(n*m, r)
                V = sol.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = sol.x[n*m*r + m*p*r : ].reshape(p*n, r)
                
                # Round to half-integers
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                
                if verify(U_r, V_r, W_r) and is_half_integer(U_r) and is_half_integer(V_r) and is_half_integer(W_r):
                    return {
                        "U": mat_to_list(U_r),
                        "V": mat_to_list(V_r),
                        "W": mat_to_list(W_r)
                    }
        except Exception:
            pass
        
        return None

    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = 14.
    # Upper bound is 28.
    # We know 25 is possible from inspiration.
    # Let's try 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14
    
    for r in range(24, 13, -1):
        if time.time() > deadline - 5:
            break
        sol = try_lm_rank(r)
        if sol is not None:
            return sol
            
    return best
# EVOLVE-BLOCK-END
