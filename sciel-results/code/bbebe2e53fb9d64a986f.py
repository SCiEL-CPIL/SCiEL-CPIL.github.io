"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _laderman_3x3_3x3():
    """Return U, V, W matrices for 3x3x3 matrix multiplication with rank 23.
    
    U: shape (9, 23) - rows indexed by i*3+j
    V: shape (9, 23) - rows indexed by j*3+k  
    W: shape (9, 23) - rows indexed by k*3+i
    """
    n = 3
    rank = 23
    
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Laderman's algorithm factors for 3x3x3
    factors = [
        (0, 0, 0, 1),
        (1, 3, 1, 1),
        (2, 6, 2, 1),
        (3, 1, 3, 1),
        (4, 4, 4, 1),
        (5, 7, 5, 1),
        (6, 2, 6, 1),
        (7, 5, 7, 1),
        (8, 8, 8, 1),
        (0, 4, 4, 1),
        (1, 7, 5, 1),
        (2, 1, 3, 1),
        (3, 8, 8, 1),
        (4, 2, 6, 1),
        (5, 5, 7, 1),
        (6, 3, 1, 1),
        (7, 0, 0, 1),
        (8, 6, 2, 1),
        (0, 6, 2, -1),
        (1, 1, 3, -1),
        (2, 5, 7, -1),
        (3, 7, 5, -1),
        (4, 0, 0, -1),
    ]
    
    for t, (u_idx, v_idx, w_idx, coeff) in enumerate(factors):
        U[u_idx, t] = coeff
        V[v_idx, t] = coeff
        W[w_idx, t] = coeff
    
    return U, V, W

def _verify_tensor(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(U.shape[1]):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_reconstruction(U, V, W, n, m, p):
    """Verify by reconstructing matrix products for random inputs"""
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_reconstructed = np.zeros((n, p))
        for t in range(U.shape[1]):
            u_col = U[:, t]
            v_col = V[:, t]
            w_col = W[:, t]
            
            a_t = 0.0
            for i in range(n):
                for j in range(m):
                    a_t += u_col[i*m+j] * A[i, j]
            
            b_t = 0.0
            for j in range(m):
                for k in range(p):
                    b_t += v_col[j*p+k] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_reconstructed[i, k] += a_t * b_t * w_col[k*n+i]
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Try Laderman's rank-23 decomposition for two 3x3x3 blocks -> rank 46
    U_3x3, V_3x3, W_3x3 = _laderman_3x3_3x3()
    rank_3x3 = U_3x3.shape[1]  # 23
    
    # Verify the 3x3x3 decomposition
    if _verify_tensor(U_3x3, V_3x3, W_3x3, 3, 3, 3):
        # Build 3x3x6 from two 3x3x3 blocks
        rank = 2 * rank_3x3  # 46
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # First block: columns 0-2 of B
        for t in range(rank_3x3):
            U[:, t] = U_3x3[:, t]
            for j in range(m):
                for k in range(3):
                    V[j*p + k, t] = V_3x3[j*3 + k, t]
            for k in range(3):
                for i in range(n):
                    W[k*n + i, t] = W_3x3[k*3 + i, t]
        
        # Second block: columns 3-5 of B
        for t in range(rank_3x3):
            idx = rank_3x3 + t
            U[:, idx] = U_3x3[:, t]
            for j in range(m):
                for k in range(3):
                    V[j*p + (k+3), idx] = V_3x3[j*3 + k, t]
            for k in range(3):
                for i in range(n):
                    W[(k+3)*n + i, idx] = W_3x3[k*3 + i, t]
        
        # Verify full decomposition
        if _verify_tensor(U, V, W, n, m, p):
            if _verify_reconstruction(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: schoolbook decomposition (rank 54)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
