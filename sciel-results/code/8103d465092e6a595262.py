import numpy as np

def build_3x3x2_block():
    """Build rank-13 decomposition for 3x3x2 matrix multiplication.
    
    U: 9x13, V: 6x13, W: 6x13
    U index: i*3+j
    V index: j*2+k
    W index: k*3+i
    """
    U = np.zeros((9, 13), dtype=float)
    V = np.zeros((6, 13), dtype=float)
    W = np.zeros((6, 13), dtype=float)
    
    # Strassen 2x2x2 (7 terms) for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # M1 = (A00+A11)*(B00+B11)
    U[0,0] = 1; U[4,0] = 1  # A[0,0], A[1,1]
    V[0,0] = 1; V[3,0] = 1  # B[0,0], B[1,1]
    W[0,0] = 1; W[4,0] = 1  # C[0,0] (k=0,i=0), C[1,1] (k=1,i=1)
    
    # M2 = (A10+A11)*B00
    U[3,1] = 1; U[4,1] = 1  # A[1,0], A[1,1]
    V[0,1] = 1  # B[0,0]
    W[1,1] = 1  # C[1,0] (k=0,i=1)
    
    # M3 = A00*(B01-B11)
    U[0,2] = 1  # A[0,0]
    V[1,2] = 1; V[3,2] = -1  # B[0,1], -B[1,1]
    W[3,2] = 1  # C[0,1] (k=1,i=0)
    
    # M4 = A11*(B10-B00)
    U[4,3] = 1  # A[1,1]
    V[2,3] = 1; V[0,3] = -1  # B[1,0], -B[0,0]
    W[0,3] = -1; W[1,3] = 1  # -C[0,0], C[1,0]
    
    # M5 = (A00+A01)*B11
    U[0,4] = 1; U[1,4] = 1  # A[0,0], A[0,1]
    V[3,4] = 1  # B[1,1]
    W[0,4] = 1; W[3,4] = 1  # C[0,0], C[0,1]
    
    # M6 = (A10-A00)*(B00+B01)
    U[3,5] = 1; U[0,5] = -1  # A[1,0], -A[0,0]
    V[0,5] = 1; V[1,5] = 1  # B[0,0], B[0,1]
    W[4,5] = 1  # C[1,1] (k=1,i=1)
    
    # M7 = (A01-A11)*(B10+B11)
    U[1,6] = 1; U[4,6] = -1  # A[0,1], -A[1,1]
    V[2,6] = 1; V[3,6] = 1  # B[1,0], B[1,1]
    W[0,6] = -1; W[4,6] = -1  # -C[0,0], -C[1,1]
    
    # Cross terms (6 terms)
    # t1 = (A02+A12)*B20 -> C[0,0], C[1,0]
    U[2,7] = 1; U[5,7] = 1  # A[0,2], A[1,2]
    V[4,7] = 1  # B[2,0]
    W[0,7] = 1; W[1,7] = 1  # C[0,0], C[1,0]
    
    # t2 = (A02+A12)*B21 -> C[0,1], C[1,1]
    U[2,8] = 1; U[5,8] = 1  # A[0,2], A[1,2]
    V[5,8] = 1  # B[2,1]
    W[3,8] = 1; W[4,8] = 1  # C[0,1], C[1,1]
    
    # t3 = A20*(B00+B01) -> C[2,0], C[2,1]
    U[6,9] = 1  # A[2,0]
    V[0,9] = 1; V[1,9] = 1  # B[0,0], B[0,1]
    W[2,9] = 1; W[5,9] = 1  # C[2,0] (k=0,i=2), C[2,1] (k=1,i=2)
    
    # t4 = A21*(B10+B11) -> C[2,0], C[2,1]
    U[7,10] = 1  # A[2,1]
    V[2,10] = 1; V[3,10] = 1  # B[1,0], B[1,1]
    W[2,10] = 1; W[5,10] = 1  # C[2,0], C[2,1]
    
    # t5 = A22*B20 -> C[2,0]
    U[8,11] = 1  # A[2,2]
    V[4,11] = 1  # B[2,0]
    W[2,11] = 1  # C[2,0]
    
    # t6 = A22*B21 -> C[2,1]
    U[8,12] = 1  # A[2,2]
    V[5,12] = 1  # B[2,1]
    W[5,12] = 1  # C[2,1]
    
    return U, V, W

def verify_3x3x2(U, V, W):
    """Verify the decomposition."""
    n, m, p = 3, 3, 2
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if valid, 0 otherwise
    # Actually the tensor condition is:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give
    # the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
    # But the full tensor condition is more complex.
    
    # Better: verify by random test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Wait, that's not right. Let me think again.
    
    # The decomposition says: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[(i,j),(j',k),(i',k')] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # 
    # The multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_j sum_r T[(i,j),(j,k),(i,k)] * A[i,j] * B[j,k]
    # = sum_r (sum_{i,j} U[i*m+j,r]*A[i,j]) * (sum_{j,k} V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # 
    # No, that's not how it works. The correct formula is:
    # C[i,k] = sum_r W[k*n+i, r] * prod... 
    # 
    # Actually in the standard formulation:
    # C = sum_r W_r ⊙ (U_r^T A) ⊙ (V_r^T B)
    # where U_r is a vector of length n*m, V_r of length m*p, W_r of length n*p
    # and the operations are:
    # M1_r = U_r reshaped to (n,m) dot A ... no.
    
    # Let me use the direct tensor verification:
    # The tensor T has T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k'*n+i' where i=i', j=j', k=k'
    # i.e., T[a,b,c] = 1 iff a%m == b//p and a//m == c%n and b%p == c//n
    
    # For the decomposition: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Let's verify this directly
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    if not np.allclose(T, T_decomp):
        diff = np.abs(T - T_decomp)
        print("MISMATCH! Max diff:", diff.max())
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print("At index:", idx, "T=", T[idx], "T_decomp=", T_decomp[idx])
        return False
    return True

def construct():
    n, m, p = 3, 3, 6
    
    # Build 3x3x2 block
    U_block, V_block, W_block = build_3x3x2_block()
    
    # Verify
    if not verify_3x3x2(U_block, V_block, W_block):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Replicate 3 times for 3x3x6
    # The 6 columns of B are split into 3 groups of 2
    # Block g handles B columns 2g and 2g+1
    # U is the same for all blocks (A is 3x3)
    # V for block g: V_block with indices shifted
    # W for block g: W_block with indices shifted
    
    rank = 3 * 13  # 39
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for g in range(3):
        # U: same for all blocks
        U[:, g*13:(g+1)*13] = U_block
        
        # V: block g handles B columns 2g, 2g+1
        # V_block has indices j*2+k for j in {0,1,2}, k in {0,1}
        # For block g, B index j*p + (2g+k) = j*6 + 2g + k
        # V_block index: j*2 + k
        for j in range(m):
            for k in range(2):
                V[j * p + 2*g + k, g*13:(g+1)*13] = V_block[j*2+k, :]
        
        # W: block g handles C columns 2g, 2g+1
        # W_block has indices k*3+i for k in {0,1}, i in {0,1,2}
        # For block g, C index (2g+k)*n + i
        for k in range(2):
            for i in range(n):
                W[(2*g + k) * n + i, g*13:(g+1)*13] = W_block[k*n+i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
