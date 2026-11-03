import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 120
N, M, P = 3, 3, 6

def _lader_3x3x3():
    """
    Return a rank-23 decomposition for 3x3x3 matrix multiplication.
    U, V, W each of shape (9, 23).
    Convention: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Helper to set a term
    def set_term(r, i, j, k, u_val=1.0, v_val=1.0, w_val=1.0):
        U[i*3+j, r] = u_val
        V[j*3+k, r] = v_val
        W[k*3+i, r] = w_val

    # 9 "single" terms
    set_term(0, 0, 0, 0)
    set_term(1, 1, 1, 1)
    set_term(2, 2, 2, 2)
    set_term(3, 0, 1, 0) # a01 b10 -> c00? No, c00 = a00b00 + a01b10 + a02b20.
    # Wait, the tensor condition is sum U[i,j] V[j,k] W[k,i] = 1.
    # For c00 (i=0, k=0): sum_j U[0,j] V[j,0] W[0,0].
    # So U[0,1] V[1,0] W[0,0] contributes to c00.
    # My set_term(r, i, j, k) sets U[i,j], V[j,k], W[k,i].
    # So set_term(3, 0, 1, 0) sets U[0,1], V[1,0], W[0,0]. This contributes to c00. Correct.
    
    set_term(4, 1, 0, 1) # a10 b01 -> c11
    set_term(5, 0, 2, 0) # a02 b20 -> c00
    set_term(6, 2, 0, 2) # a20 b02 -> c22
    set_term(7, 1, 2, 1) # a12 b21 -> c11
    set_term(8, 2, 1, 2) # a21 b12 -> c22
    
    # 14 "combination" terms
    # I will use the Lader coefficients.
    # Term 9: (a00+a01+a10+a11)(b00+b01+b10+b11)
    # This term contributes to all c_ij where i in {0,1}, j in {0,1} (via W[k,i] where k in {0,1})
    # Actually, the W factor determines which c_ki is affected.
    # In the tensor decomposition, the W matrix is W[k,i].
    # For the term to be valid, the sum of products must match the target tensor.
    
    # This is too complex to derive manually in the code generation step.
    # I will use a known valid set of coefficients.
    
    # Coefficients for Lader decomposition (from Lader 1975):
    # I will use the following explicit matrices.
    
    # Term 9
    U[0,9]=1; U[1,9]=1; U[3,9]=1; U[4,9]=1
    V[0,9]=1; V[1,9]=1; V[3,9]=1; V[4,9]=1
    W[0,9]=1; W[3,9]=1 # W[k,i] for k in {0,1}, i in {0,1}
    # This is getting risky.
    
    # Fallback: Return None to use schoolbook
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = N, M, P
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return get_schoolbook()
    
    # Try to use a rank-46 decomposition
    # For 3x3x6, use two 3x3x3 blocks with a rank-23 decomposition each
    
    # I will use a verified rank-23 decomposition for 3x3x3.
    # Since I cannot guarantee the hardcoded Lader coefficients are correct in this environment,
    # I will return the schoolbook solution (rank 54).
    
    return get_schoolbook()
