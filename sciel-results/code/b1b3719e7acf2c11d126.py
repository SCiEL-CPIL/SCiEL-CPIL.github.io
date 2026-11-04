# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays with shape (4, 7).
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Indices: 0:00, 1:01, 2:10, 3:11 for 2x2 matrices
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    Uses vectorized NumPy operations.
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Reshape U2, V2, W2 to (2, 2, 7) for block indices
    # U2[i, j, r] corresponds to A[i,j] in term r
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    
    # We want to fill U[2*i1+i2, 2*k1+k2, r1*7+r2] = U2_3d[i1, k1, r1] * U2_3d[i2, k2, r2]
    # Note: U2 is indexed by [row, col, term] in the 2x2 sense?
    # In _strassen_2x2, U[0,0]=1 (A00), U[3,0]=1 (A11).
    # Index 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1).
    # So U2[i*2+j, r].
    # Reshape to (2, 2, 7) gives U2_3d[i, j, r].
    
    # We need to map 4x4 indices (i, k) to 2x2 block indices (i1, i2) and (k1, k2).
    # i = 2*i1 + i2, k = 2*k1 + k2.
    # The term index r = r1*7 + r2.
    
    # We can use broadcasting to compute the Kronecker product of the matrices
    # but we need to interleave the indices correctly.
    
    # Let's just use the loop but vectorize the inner part?
    # The loop is small enough.
    
    # Actually, we can construct the full Kronecker product of U2 and U2, then permute.
    # U_kron = np.kron(U2, U2) gives shape (4*4, 7*7) = (16, 49).
    # The row index of U_kron is i*4 + k?
    # np.kron(A, B) where A is (2,7) and B is (4,7)? No.
    # We have U2 of shape (4, 7).
    # We want U of shape (16, 49).
    # U[i, k, r1, r2] = U2[i1, k1, r1] * U2[i2, k2, r2].
    
    # Let's stick to the explicit loop for clarity and correctness, as it is fast.
    for i1 in range(2):
        for i2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    i = 2 * i1 + i2
                    k = 2 * k1 + k2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = 2 * j1 + j2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U[i * m + k, r] = U2[i1 * 2 + k1, r1] * U2[i2 * 2 + k2, r2]
                                    V[k * p + j, r] = V2[k1 * 2 + j1, r1] * V2[k2 * 2 + j2, r2]
                                    W[j * n + i, r] = W2[j1 * 2 + i1, r1] * W2[j2 * 2 + i2, r2]
    
    U_list = [list(map(int, row)) for row in U]
    V_list = [list(map(int, row)) for row in V]
    W_list = [list(map(int, row)) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    result = _kronecker_decomposition(n, m, p)
    
    if time.time() < deadline:
        valid, max_diff = _verify_decomp(n, m, p, result["U"], result["V"], result["W"])
        if not valid:
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": result["U"], "V": result["V"], "W": result["W"]}
# EVOLVE-BLOCK-END
