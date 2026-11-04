import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Strassen 2x2x2, rank 7. U(4,7),V(4,7),W(4,7). W indexed by k*2+i."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1=(A00+A11)(B00+B11) -> C00+C11
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A10+A11)B00 -> C10-C11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3=A00(B01-B11) -> C01+C11
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[4,2]=1; W[3,2]=1
    # M4=A11(B10-B00) -> C10+C00
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[2,3]=1; W[0,3]=1
    # M5=(A00+A01)B11 -> C00-C10
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[4,4]=1
    # M6=(A10-A00)(B00+B01) -> C11
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(A01-A11)(B10+B11) -> C00
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _strassen_4x4x4():
    """Two-level Strassen for 4x4x4, rank 49. U(16,49),V(16,49),W(16,49). W[k*4+i]."""
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # 7 Strassen patterns at the 4x4 level
    patterns = [
        # (A_blocks, B_blocks, C_blocks) where each is list of (block_idx, coeff)
        # Block indices: 00,01,10,11 for 2x2 blocks
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((1,0),1),((0,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((1,0),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]
    
    for pi, (A_b, B_b, C_b) in enumerate(patterns):
        for si in range(7):
            r = pi * 7 + si
            # Fill U
            for (bi, bk), ca in A_b:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2*bi + i2
                        k = 2*bk + k2
                        U[i*4+k, r] += ca * U2[i2*2+k2, si]
            # Fill V
            for (bk, bj), cb in B_b:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2*bk + k2
                        j = 2*bj + j2
                        V[k*4+j, r] += cb * V2[k2*2+j2, si]
            # Fill W
            for (cj, ci), cc in C_b:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2*cj + j2
                        i = 2*ci + i2
                        W[k*4+i, r] += cc * W2[j2*2+i2, si]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Reconstruct C from decomposition
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # In decomposition: C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]... 
    # Actually the convention is: W[k*n+i, r] where k is the B-row/C-col index, i is the A-row/C-row index
    # So C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j]) * ... 
    # No, the standard tensor decomposition is:
    # C[i,j] = sum_{k,r} A[i,k] * B[k,j] * T[i,j,k,r]
    # Where T is the rank-r decomposition
    # U[i*m+k, r] corresponds to A factor, V[k*p+j, r] to B factor, W[j*n+i, r] to C factor
    # So: C[i,j] = sum_r (sum_k U[i*m+k,r]*A[i,k]) * (sum_k V[k*p+j,r]*B[k,j]) ... no
    
    # Correct interpretation: The multiplication tensor M has entries M[i,j,k] = 1 if we're computing C[i,j] from A[i,k]*B[k,j]
    # Rank-r decomposition: M[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[i,j,r]
    # In our indexing: U is (n*m, rank) with U[i*m+k, r], V is (m*p, rank) with V[k*p+j, r], W is (p*n, rank) with W[j*n+i, r]
    # So C[i,j] = sum_r (sum_k A[i,k]*U[i*m+k,r]) * (sum_k B[k,j]*V[k*p+j,r]) ... no, that's not right either.
    
    # The correct formula is:
    # C[i,j] = sum_{k,r} A[i,k] * B[k,j] * (coefficient)
    # In tensor form: the multiplication tensor T[i,j,k] = 1 for the (i,k,j) entry
    # Rank-r: T[i,j,k] = sum_r u[i,k,r] * v[k,j,r] * w[i,j,r]
    # So C[i,j] = sum_{k,r} A[i,k] * B[k,j] * w[i,j,r] * ... no
    
    # Let me just verify by computing: for each (i,j), C[i,j] should equal sum over r of the product
    # Actually, the standard way: given U(n*m, r), V(m*p, r), W(p*n, r)
    # C[i,j] = sum_r (U[i*m + :, r] @ A[i, :]) * (V[:, j] @ B[:, j]) ... no
    
    # The correct verification: 
    # The tensor M[i,j,k] = 1 means C[i,j] = sum_k A[i,k] * B[k,j]
    # A rank-r decomposition gives: M[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # This must hold for all i,j,k.
    # Then C[i,j] = sum_k A[i,k]*B[k,j] = sum_k sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r]*A[i,k]*B[k,j]
    # This doesn't directly give C. The decomposition is of the tensor, not a factorization of the product.
    
    # To verify: check that for all i,j,k: sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] == 1
    for i in range(n):
        for j in range(p):
            for k in range(m):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _build_4x4x9_strassen():
    """Build 4x4 x 4x9 decomposition using Strassen rank-49 for two 4x4x4 blocks + rank-16 for last column."""
    U4, V4, W4 = _strassen_4x4x4()
    rank = 49 + 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Block 1: B columns 0-3 (4x4 x 4x4 -> 4x4)
    U[:, 0:49] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+j, 0:49] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[j*4+i, 0:49] = W4[j*4+i, :]
    
    # Block 2: B columns 4-7 (4x4 x 4x4 -> 4x4)
    U[:, 49:98] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), 49:98] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, 49:98] = W4[j*4+i, :]
    
    # Block 3: B column 8 (4x4 x 4x1 -> 4x1), rank 16
    for i in range(4):
        for k in range(4):
            r = 98 + i*4 + k
            U[i*4+k, r] = 1.0
            V[k*9+8, r] = 1.0
            W[8*4+i, r] = 1.0
    
    return U, V, W, rank

def _schoolbook_fallback():
    """Schoolbook decomposition, rank n*m*p = 144."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W, rank = _build_4x4x9_strassen()
        
        # Verify the decomposition
        if not _verify_decomposition(U, V, W, n, m, p):
            return _schoolbook_fallback()
        
        # Convert to lists for the required interface
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    except Exception as e:
        return _schoolbook_fallback()
