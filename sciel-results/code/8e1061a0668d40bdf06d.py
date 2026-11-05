import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Strassen 2x2x2, rank 7. U(4,7),V(4,7),W(4,7). W[j*2+i]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1=(A00+A11)(B00+B11) -> C00+C11
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A10+A11)B00 -> C10-C11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3=A00(B01-B11) -> C01+C11
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4=A11(B10-B00) -> C10+C00
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5=(A00+A01)B11 -> C00-C10
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6=(A10-A00)(B00+B01) -> C11
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(A01-A11)(B10+B11) -> C00
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _strassen_4x4x4():
    """Strassen 4x4x4 via 2-level, rank 49. U(16,49),V(16,49),W(16,49)."""
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    big = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for bi, (A_b, B_b, C_b) in enumerate(big):
        for si in range(7):
            r = bi*7 + si
            for (bi_i, bi_k, ca) in A_b:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2*bi_i + i2; k = 2*bi_k + k2
                        U[i*4+k, r] += ca * U2[i2*2+k2, si]
            for (bk_i, bk_j, cb) in B_b:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2*bk_i + k2; j = 2*bk_j + j2
                        V[k*4+j, r] += cb * V2[k2*2+j2, si]
            for (cj_i, cj_j, cc) in C_b:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2*cj_i + j2; i = 2*cj_j + i2
                        W[j*4+i, r] += cc * W2[j2*2+i2, si]
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Validate decomposition using random test."""
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    rank = U_np.shape[1]
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_col = U_np[:, r].reshape(n, m)
        v_col = V_np[:, r].reshape(m, p)
        w_col = W_np[:, r].reshape(p, n)
        C_computed += u_col @ v_col * w_col.T  # Wait, this isn't right
    
    # Actually, the tensor decomposition is:
    # T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # So C[i,j] = sum_k A[i,k] * T[i,k,j] * B[k,j]... no
    # 
    # The matrix multiplication tensor: T[i,k,j] such that
    # (A@B)[i,j] = sum_k A[i,k] * B[k,j]
    # The rank-r decomposition: T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # Then (A@B)[i,j] = sum_k sum_r A[i,k] * U[i,k,r] * V[k,j,r] * W[j,i,r] * B[k,j]
    # This doesn't simplify nicely unless we interpret it as:
    # (A@B)[i,j] = sum_r (sum_k A[i,k]*U[i,k,r]) * (sum_k B[k,j]*V[k,j,r]) * W[j,i,r]
    # Wait, that's not right either because U and V share the k index.
    
    # Let me re-derive. The standard form is:
    # M[i,j] = sum_k A[i,k] B[k,j]
    # Rank-r: M[i,j] = sum_r (sum_k A[i,k] U_r[i,k]) (sum_k B[k,j] V_r[k,j]) W_r[i,j]
    # But that's not standard. The standard is:
    # M[i,j] = sum_r (sum_k A[i,k] U_r[i,k]) (sum_k B[k,j] V_r[k,j]) 
    # No, that would be rank-1 in a different sense.
    
    # Actually the correct interpretation for tensor rank:
    # The tensor T has T[i,k,j] = 1 if this is the (i,k,j) entry of the multiplication tensor
    # Rank-r decomposition: T[i,k,j] = sum_r U_r[i,k] V_r[k,j] W_r[i,j]
    # Then M[i,j] = sum_k A[i,k] B[k,j] = sum_k sum_r A[i,k] B[k,j] U_r[i,k] V_r[k,j] W_r[i,j]
    # = sum_r W_r[i,j] (sum_k A[i,k] U_r[i,k]) (sum_k B[k,j] V_r[k,j])
    # Wait, that's not right because U_r[i,k] and V_r[k,j] both depend on k.
    
    # Let me just directly compute:
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            val = 0.0
            for k in range(m):
                for r in range(rank):
                    val += A[i,k] * U_np[i*m+k, r] * V_np[k*p+j, r] * W_np[j*n+i, r] * B[k,j]
            C_computed[i,j] = val
    
    # Hmm, that's O(n*p*m*rank) which is fine for small sizes
    # But wait, that's not the right formula either. Let me think again.
    
    # The tensor T[i,k,j] represents the coefficient such that
    # (A@B)[i,j] = sum_{k} A[i,k] * B[k,j] * T[i,k,j]
    # where T[i,k,j] = 1 for all valid i,k,j (since it's just standard multiplication)
    
    # A rank-r decomposition of T is: T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # (Note: W is indexed by (j,i) per the problem statement: W[k*n+i,r] means W is p*n x rank,
    #  indexed by (k,i) where k is the column index of B/C, i is the row index of A/C)
    
    # Wait, the problem says: W[k*n+i,r] where k ranges over p (columns of B) and i over n (rows of A)
    # So W has shape (p*n, rank), and W[k*n+i, r] is the coefficient for output C[i,k]
    
    # So the decomposition is: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # where i in [0,n), k in [0,m), j in [0,p)
    # And (A@B)[i,j] = sum_k A[i,k] * B[k,j] * T[i,k,j]
    # = sum_k sum_r A[i,k] * B[k,j] * U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r]
    
    # This is what we need to verify. Let me rewrite the validation:
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                for r in range(rank):
                    C_computed[i,j] += A[i,k] * B[k,j] * U_np[i*m+k, r] * V_np[k*p+j, r] * W_np[j*n+i, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build 4x4x4 Strassen decomposition
    U4, V4, W4 = _strassen_4x4x4()
    
    # Build 4x4x9 decomposition: split B into B1(4x4), B2(4x4), B3(4x1)
    total_rank = 49 + 49 + 16
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Block 1: columns 0-3 of B
    U[:, 0:49] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+j, 0:49] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[j*4+i, 0:49] = W4[j*4+i, :]
    
    # Block 2: columns 4-7 of B
    U[:, 49:98] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), 49:98] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, 49:98] = W4[j*4+i, :]
    
    # Block 3: column 8 of B (direct, rank 16)
    for i in range(4):
        for k in range(4):
            r = 98 + i*4 + k
            U[i*4+k, r] = 1
            V[k*9+8, r] = 1
            W[8*4+i, r] = 1
    
    # Validate
    if not _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_fb[i*m+k, col] = 1
                    V_fb[k*p+j, col] = 1
                    W_fb[j*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
