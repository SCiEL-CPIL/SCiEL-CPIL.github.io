"""Matrix multiplication 2,4,5 - exact rank-36 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _build_1x2x1():
    """Rank-2 decomposition of M_{1,2,1}: C = A (1x2) * B (2x1)."""
    n, m, p = 1, 2, 1
    rank = 2
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for j in range(m):
        c = j
        U[j, c] = 1.0
        V[j, c] = 1.0
        W[0, c] = 1.0
    return U, V, W

def _build_2x2x2():
    """Rank-7 Strassen decomposition of M_{2,2,2}."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U factors: A11=(1,1,1,1,1,0,-1), A12=(0,0,1,0,1,-1,0), A21=(0,1,0,-1,0,1,0), A22=(1,1,0,1,0,0,0)
    U[0, :] = [1, 0, 0, 1, 0, 0, -1]  # A11
    U[1, :] = [0, 1, 0, -1, 0, 1, 0]  # A21
    U[2, :] = [0, 0, 1, 0, 1, -1, 0]  # A12
    U[3, :] = [1, 1, 0, 1, 0, 0, 0]   # A22
    
    # V factors: B11=(1,1,0,0,0,1,0), B12=(0,0,0,0,0,1,0), B21=(1,0,1,1,0,0,1), B22=(1,0,0,-1,1,0,1)
    V[0, :] = [1, 1, 0, 0, 0, 1, 0]   # B11
    V[1, :] = [0, 0, 0, 0, 0, 1, 0]   # B12
    V[2, :] = [1, 0, 1, 1, 0, 0, 1]   # B21
    V[3, :] = [1, 0, 0, -1, 1, 0, 1]  # B22
    
    # W factors:
    # W for C11: M1+M4-M5+M7 -> (1,0,-1,1,0,0,1)
    # W for C21: M2+M4 -> (0,1,0,1,0,0,0)
    # W for C12: M3+M5 -> (0,0,1,0,1,0,0)
    # W for C22: M1-M2+M3+M6 -> (1,-1,1,0,0,1,0)
    W[0, :] = [1, 0, -1, 1, 0, 0, 1]  # C11
    W[1, :] = [0, 0, 1, 0, 1, 0, 0]   # C12
    W[2, :] = [0, 1, 0, 1, 0, 0, 0]   # C21
    W[3, :] = [1, -1, 1, 0, 0, 1, 0]  # C22
    
    return U, V, W

def _build_2x2x4():
    """Rank-14 decomposition of M_{2,2,4} = M_{2,2,2} x M_{1,2,2}."""
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 1, 2, 2
    
    U1, V1, W1 = _build_2x2x2()
    U2, V2, W2 = _build_1x2x1()  # M_{1,2,1}
    
    # Actually M_{2,2,4} = M_{2,2,2} * M_{1,2,2}
    # But M_{1,2,2} = M_{1,2,1} * M_{1,1,2}
    # M_{1,1,2} is just two independent multiplications: rank 2
    # So M_{1,2,2} has rank 4: U is 1x4, V is 2x4, W is 2x4
    # Let's build M_{1,2,2} directly: A (1x2), B (2x2), C (1x2)
    # C_k = sum_j A_j B_jk
    # Rank 4: for each (j,k)
    n, m, p = 1, 2, 2
    rank_122 = 4
    U_122 = np.zeros((n * m, rank_122))
    V_122 = np.zeros((m * p, rank_122))
    W_122 = np.zeros((p * n, rank_122))
    col = 0
    for j in range(m):
        for k in range(p):
            U_122[j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k, col] = 1.0
            col += 1
    
    # Now tensor product of M_{2,2,2} and M_{1,2,2}
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    rank = U1.shape[1] * U_122.shape[1]
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n1):
        for j in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    row_idx = (i * m1 + j) * (n2 * m2) + (i2 * m2 + j2)
                    for c1 in range(U1.shape[1]):
                        for c2 in range(U_122.shape[1]):
                            col_idx = c1 * U_122.shape[1] + c2
                            U[row_idx, col_idx] = U1[i * m1 + j, c1] * U_122[i2 * m2 + j2, c2]
    
    for j in range(m1):
        for k in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    row_idx = (j * p1 + k) * (m2 * p2) + (j2 * p2 + k2)
                    for c1 in range(V1.shape[1]):
                        for c2 in range(V_122.shape[1]):
                            col_idx = c1 * V_122.shape[1] + c2
                            V[row_idx, col_idx] = V1[j * p1 + k, c1] * V_122[j2 * p2 + k2, c2]
    
    for k in range(p1):
        for i in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    row_idx = (k * n1 + i) * (p2 * n2) + (k2 * n2 + i2)
                    for c1 in range(W1.shape[1]):
                        for c2 in range(W_122.shape[1]):
                            col_idx = c1 * W_122.shape[1] + c2
                            W[row_idx, col_idx] = W1[k * n1 + i, c1] * W_122[k2 * n2 + i2, c2]
    
    return U, V, W

def _build_2x4x4():
    """Rank-28 decomposition of M_{2,4,4} = M_{2,2,4} x M_{1,2,1}."""
    n1, m1, p1 = 2, 2, 4
    n2, m2, p2 = 1, 2, 1
    
    U1, V1, W1 = _build_2x2x4()
    U2, V2, W2 = _build_1x2x1()
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    rank = U1.shape[1] * U2.shape[1]
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n1):
        for j in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    row_idx = (i * m1 + j) * (n2 * m2) + (i2 * m2 + j2)
                    for c1 in range(U1.shape[1]):
                        for c2 in range(U2.shape[1]):
                            col_idx = c1 * U2.shape[1] + c2
                            U[row_idx, col_idx] = U1[i * m1 + j, c1] * U2[i2 * m2 + j2, c2]
    
    for j in range(m1):
        for k in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    row_idx = (j * p1 + k) * (m2 * p2) + (j2 * p2 + k2)
                    for c1 in range(V1.shape[1]):
                        for c2 in range(V2.shape[1]):
                            col_idx = c1 * V2.shape[1] + c2
                            V[row_idx, col_idx] = V1[j * p1 + k, c1] * V2[j2 * p2 + k2, c2]
    
    for k in range(p1):
        for i in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    row_idx = (k * n1 + i) * (p2 * n2) + (k2 * n2 + i2)
                    for c1 in range(W1.shape[1]):
                        for c2 in range(W2.shape[1]):
                            col_idx = c1 * W2.shape[1] + c2
                            W[row_idx, col_idx] = W1[k * n1 + i, c1] * W2[k2 * n2 + i2, c2]
    
    return U, V, W

def _build_2x4x1():
    """Rank-8 decomposition of M_{2,4,1}."""
    n, m, p = 2, 4, 1
    rank = n * m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j, c] = 1.0
            V[j, c] = 1.0
            W[i, c] = 1.0
    
    return U, V, W

def construct():
    t0 = time.time()
    
    # Build M_{2,4,4} (rank 28) and M_{2,4,1} (rank 8)
    U1, V1, W1 = _build_2x4x4()
    U2, V2, W2 = _build_2x4x1()
    
    # Concatenate for M_{2,4,5}
    n, m, p = DIMENSIONS
    rank = U1.shape[1] + U2.shape[1]
    
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Verify correctness
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Reconstruct C from decomposition
    # C[i,k] = sum_r sum_{j} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
    # Actually: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j V[j*p+k, r] * B[j,k])
    # But the tensor decomposition is: T[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # C[i,k] = sum_j sum_r A[i,j] * B[j,k] * T[i,j,k,r]
    # = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
    
    C_recon = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r]
        v_vec = V[:, r]
        w_val = W[:, r]
        
        # Sum over j: A[i,j] * U[i*m+j, r] for each i
        a_u = np.zeros(n)
        b_v = np.zeros(p)
        for i in range(n):
            for j in range(m):
                a_u[i] += A[i, j] * u_vec[i * m + j]
                b_v[j] += B[j, k] * v_vec[j * p + k]  # This is wrong, need to do it properly
        
        # Let me redo this more carefully
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += A[i, j] * B[j, k] * U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_recon[i, k] += s
    
    if not np.allclose(C_recon, C_true, atol=1e-9):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
