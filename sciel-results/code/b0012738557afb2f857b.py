# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Target rank to try
    # Schoolbook is 32.
    # Try to find a lower rank.
    
    def check_decomposition(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        R = U.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (val - 1.0)**2
        return error

    def try_rank(R, iterations=50):
        # Random initialization
        for _ in range(iterations):
            # Initialize with small random integers
            U = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
            
            # Solve for W
            # System: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # This is linear in W for fixed U, V.
            # A * W_flat = b
            # A is (n*m*p) x (p*n*R)
            # b is (n*m*p)
            
            num_eqs = n * m * p
            num_unk = p * n * R
            
            A = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation idx
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        for r in range(R):
                            # W index is k*n + i, column r
                            # In W_flat, index is (k*n+i)*R + r
                            w_idx = (k*n + i) * R + r
                            A[idx, w_idx] = U[i*m+j, r] * V[j*p+k, r]
                        idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W = W_flat.reshape(p*n, R)
            
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check error
            err = check_decomposition(U_rounded, V_rounded, W_rounded)
            if err < 1e-9:
                return U_rounded, V_rounded, W_rounded
                
        return None

    # Try ranks from 24 to 31
    for R in range(24, 32):
        res = try_rank(R, iterations=20)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
