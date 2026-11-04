"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, trials=10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False

    for _ in range(trials):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        
        # Compute C from decomposition
        # C[k, i] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] ?
        # Wait, the decomposition is T = sum_t U_t (otimes) V_t (otimes) W_t.
        # T is a tensor in R^{nm} x R^{mp} x R^{pn}.
        # The contraction is:
        # C_flat = sum_t (U^T * A_flat) * (V^T * B_flat) * W_t ?
        # No.
        # Let u_t be the t-th column of U.
        # Let v_t be the t-th column of V.
        # Let w_t be the t-th column of W.
        # The term is u_t (otimes) v_t (otimes) w_t.
        # The matrix multiplication tensor T satisfies:
        # T_{(i,j), (j,k), (k,i)} = 1 if indices match, 0 otherwise.
        # The decomposition approximates T.
        # So for any A, B:
        # sum_{i,j,k} A_{ij} B_{jk} T_{(i,j),(j,k),(k,i)} = C_{ki}
        # sum_t (sum_{i,j} A_{ij} U_{ij,t}) (sum_{j,k} B_{jk} V_{jk,t}) W_{ki,t} = C_{ki}
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # u_t dot A_flat
        # U is (nm, rank). So U.T is (rank, nm).
        # u_t is a vector of length nm.
        # sum_{ij} A_{ij} U_{ij,t} = A_flat dot U[:, t]
        
        a_u = A_flat @ U  # shape (rank,)
        b_v = B_flat @ V  # shape (rank,)
        
        # C_flat should be sum_t a_u[t] * b_v[t] * W[:, t]
        # W is (pn, rank).
        # c_calc = sum_t (a_u[t] * b_v[t]) * W[:, t]
        # c_calc = (a_u * b_v) @ W
        c_calc = (a_u * b_v) @ W
        c_calc = c_calc.reshape(p, n)
        
        # Check if c_calc matches C
        # Note: W index is k*n + i. So c_calc[k, i] is the element.
        if not np.allclose(c_calc, C, atol=1e-9):
            return False
    return True

def _build_3x3x3_rank23():
    """
    Returns U, V, W for 3x3x3 matrix multiplication with rank 23.
    If verification fails, returns None.
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Hardcoded 23-rank decomposition for 3x3x3
    # This is a known decomposition.
    # I will define the terms.
    
    # Term 0: A00 B00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A00 B01
    U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    # Term 2: A01 B00
    U[1, 2] = 1; V[0, 2] = 1; W[0, 2] = 1
    # Term 3: A01 B01
    U[1, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    # Term 4: A02 B00
    U[2, 4] = 1; V[0, 4] = 1; W[0, 4] = 1
    # Term 5: A02 B01
    U[2, 5] = 1; V[1, 5] = 1; W[1, 5] = 1
    # Term 6: A10 B00
    U[3, 6] = 1; V[0, 6] = 1; W[0, 6] = 1
    # Term 7: A10 B01
    U[3, 7] = 1; V[1, 7] = 1; W[1, 7] = 1
    # Term 8: A11 B00
    U[4, 8] = 1; V[0, 8] = 1; W[0, 8] = 1
    # Term 9: A11 B01
    U[4, 9] = 1; V[1, 9] = 1; W[1, 9] = 1
    # Term 10: A12 B00
    U[5, 10] = 1; V[0, 10] = 1; W[0, 10] = 1
    # Term 11: A12 B01
    U[5, 11] = 1; V[1, 11] = 1; W[1, 11] = 1
    # Term 12: A20 B00
    U[6, 12] = 1; V[0, 12] = 1; W[0, 12] = 1
    # Term 13: A20 B01
    U[6, 13] = 1; V[1, 13] = 1; W[1, 13] = 1
    # Term 14: A21 B00
    U[7, 14] = 1; V[0, 14] = 1; W[0, 14] = 1
    # Term 15: A21 B01
    U[7, 15] = 1; V[1, 15] = 1; W[1, 15] = 1
    # Term 16: A22 B00
    U[8, 16] = 1; V[0, 16] = 1; W[0, 16] = 1
    # Term 17: A22 B01
    U[8, 17] = 1; V[1, 17] = 1; W[1, 17] = 1
    
    # Terms 18-22: Strassen-like terms for the remaining 5 multiplications
    # This part is tricky. The standard 23-rank decomposition is not simply 18 schoolbook + 5 Strassen.
    # It is a specific 23-term scheme.
    # Since I am not 100% sure of the coefficients for terms 18-22, I will return None to trigger fallback.
    # If I return None, the caller will use schoolbook for 3x3x3 (rank 27).
    # Then total rank is 27+27+18 = 72. No improvement.
    
    # To ensure an improvement, I should use a decomposition I am sure of.
    # I am sure of the schoolbook decomposition.
    # I am NOT sure of the 23-rank coefficients.
    
    return None

def _build_schoolbook(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to build a better decomposition
    # Split 8 into 3 + 3 + 2
    
    U1, V1, W1 = _build_3x3x3_rank23()
    if U1 is None:
        U1, V1, W1 = _build_schoolbook(3, 3, 3)
        
    U2, V2, W2 = _build_3x3x3_rank23()
    if U2 is None:
        U2, V2, W2 = _build_schoolbook(3, 3, 3)
        
    U3, V3, W3 = _build_schoolbook(3, 3, 2)
    
    # Concatenate
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    # Verify
    if _verify_decomposition(U, V, W, 3, 3, 8, trials=5):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to full schoolbook
        U_full, V_full, W_full = _build_schoolbook(3, 3, 8)
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
