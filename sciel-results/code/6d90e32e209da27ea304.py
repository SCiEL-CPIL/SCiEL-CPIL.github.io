"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomp(U1, V1, W1, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two tensor decompositions.
    M_{n1,m1,p1} ⊗ M_{n2,m2,p2} = M_{n1*n2, m1*m2, p1*p2}
    Rank: rank1 * rank2
    """
    rank1 = U1.shape[1]
    rank2 = U2_rank = V1.shape[1]  # should be same
    total_rank = rank1 * rank2
    
    U = np.zeros((n1 * m1 * n2 * m2, total_rank))  # This is wrong, let me fix
    # Actually U should be (n1*n2 * m1*m2, total_rank)
    # Wait no. For M_{n,m,p}, U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for t1 in range(rank1):
        for t2 in range(rank2):
            t = t1 * rank2 + t2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u_val = U1[i1 * m1 + j1, t1] * U2[i2 * m2 + j2, t2]
                            if u_val != 0:
                                U[(i1 * n2 + i2) * (m1 * m2) + j1 * m2 + j2, t] = u_val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v_val = V1[j1 * p1 + k1, t1] * V2[j2 * p2 + k2, t2]
                            if v_val != 0:
                                V[(j1 * m2 + j2) * (p1 * p2) + k1 * p2 + k2, t] = v_val
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w_val = W1[k1 * n1 + i1, t1] * W2[k2 * n2 + i2, t2]
                            if w_val != 0:
                                W[(k1 * p2 + k2) * (n1 * n2) + i1 * n2 + i2, t] = w_val
    
    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen (Kronecker product).
    Rank 49.
    """
    U1, V1, W1 = _strassen_2x2()
    U, V, W = _kron_decomp(U1, V1, W1, 2, 2, 2, 2, 2, 2)
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()
    
    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]
    
    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)
    
    return np.allclose(sums, 1.0, atol=1e-9)

def _decomp_4x4x1():
    """
    Decomposition of M_{4,4,1}. Rank 16 (schoolbook).
    """
    n, m, p = 4, 4, 1
    return _schoolbook_nm_p(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Decompose 9 as 4 + 4 + 1
    U1, V1, W1 = _decomp_4x4x4()  # rank 49
    U2, V2, W2 = _decomp_4x4x4()  # rank 49
    U3, V3, W3 = _decomp_4x4x1()  # rank 16
    
    total_rank = U1.shape[1] + U2.shape[1] + U3.shape[1]
    U = np.hstack([U1, U2, U3])
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Place V1 in rows for k=0..3
    for j in range(m):
        for k in range(4):
            V[j * p + k, :49] = V1[j * 4 + k, :]
    
    # Place V2 in rows for k=4..7
    for j in range(m):
        for k in range(4):
            V[j * p + (4 + k), 49:98] = V2[j * 4 + k, :]
    
    # Place V3 in rows for k=8
    for j in range(m):
        V[j * p + 8, 98:114] = V3[j * 1 + 0, :]
    
    # Place W1 in rows for k=0..3
    for k in range(4):
        for i in range(n):
            W[k * n + i, :49] = W1[k * 4 + i, :]
    
    # Place W2 in rows for k=4..7
    for k in range(4):
        for i in range(n):
            W[(4 + k) * n + i, 49:98] = W2[k * 4 + i, :]
    
    # Place W3 in rows for k=8
    for i in range(n):
        W[8 * n + i, 98:114] = W3[0 * 4 + i, :]
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
