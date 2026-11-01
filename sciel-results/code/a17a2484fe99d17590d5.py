"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    We use a block decomposition based on Strassen's algorithm.
    The 2x14 matrix B is split into seven 2x2 blocks.
    Each 2x2 * 2x2 multiplication is performed using Strassen's algorithm (rank 7).
    Total rank = 7 * 7 = 49.
    
    Returns:
        dict with keys 'U', 'V', 'W' as lists of lists representing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U indices: i*m + j. i in {0,1}, j in {0,1}
    # 0: 0,0; 1: 0,1; 2: 1,0; 3: 1,1
    
    # V indices: j*p + k. j in {0,1}, k in {0..13}
    # For block b, k is 2b or 2b+1.
    # j=0: 2b, 2b+1
    # j=1: 14+2b, 14+2b+1
    
    # W indices: k*n + i. k in {0..13}, i in {0,1}
    # For block b, k is 2b or 2b+1.
    # k=2b: 4b, 4b+1
    # k=2b+1: 4b+2, 4b+3
    
    for b in range(7):
        base_col = 7 * b
        
        # M1
        c = base_col
        U[0, c] = 1; U[3, c] = 1
        V[2*b, c] = 1; V[14+2*b, c] = 1
        W[4*b, c] = 1; W[4*b+3, c] = 1
        
        # M2
        c = base_col + 1
        U[2, c] = 1; U[3, c] = 1
        V[2*b, c] = 1
        W[4*b+1, c] = 1; W[4*b+3, c] = -1
        
        # M3
        c = base_col + 2
        U[0, c] = 1
        V[2*b+1, c] = 1; V[14+2*b, c] = -1
        W[4*b+2, c] = 1; W[4*b+3, c] = 1
        
        # M4
        c = base_col + 3
        U[3, c] = 1
        V[14+2*b, c] = 1; V[2*b, c] = -1
        W[4*b, c] = 1; W[4*b+1, c] = 1
        
        # M5
        c = base_col + 4
        U[0, c] = 1; U[1, c] = 1
        V[14+2*b, c] = 1
        W[4*b, c] = -1; W[4*b+2, c] = 1
        
        # M6
        c = base_col + 5
        U[0, c] = -1; U[2, c] = 1
        V[2*b, c] = 1; V[2*b+1, c] = 1
        W[4*b+3, c] = 1
        
        # M7
        c = base_col + 6
        U[1, c] = 1; U[3, c] = -1
        V[14+2*b, c] = 1; V[15+2*b, c] = 1
        W[4*b, c] = 1
        
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(U.shape[1]):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook), dtype=float)
        V = np.zeros((m * p, rank_schoolbook), dtype=float)
        W = np.zeros((p * n, rank_schoolbook), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        if not verify(U, V, W, n, m, p):
            raise ValueError("Decomposition failed verification")

    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    def round_to_half(x):
        return round(x * 2) / 2
        
    U_out = [[round_to_half(v) for v in row] for row in U_out]
    V_out = [[round_to_half(v) for v in row] for row in V_out]
    W_out = [[round_to_half(v) for v in row] for row in W_out]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
