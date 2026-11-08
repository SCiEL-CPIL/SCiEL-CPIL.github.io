import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    Ur = np.array(U).reshape(n, m, rank)
    Vr = np.array(V).reshape(m, p, rank)
    Wr = np.array(W).reshape(p, n, rank)
    
    # Check main tensor entries T[i,j,k] = 1
    T_main = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T_main, 1.0, atol=1e-9):
        return False
        
    # Check cross terms are 0
    # We need sum_r U[i,j,r] V[j',k,r] W[k',i,r] = 0 if j!=j' or k!=k'
    # This is hard to check directly with einsum without expanding dimensions.
    # However, if the main tensor is correct and the rank is minimal for the structure, 
    # and we constructed it via valid block methods, it is likely correct.
    # For a rigorous check, we can test random A, B and compare C.
    
    # Random test
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Reconstruct C from U, V, W
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) W[k,i,r]? 
    # No, the formula is:
    # C[i,k] = sum_r (U_r^T A)_{?} ...
    # Actually, the decomposition implies:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r] is WRONG.
    # The correct interpretation of tensor decomposition for bilinear map:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) is not dimensionally correct for W.
    
    # Let's stick to the tensor contraction check.
    # The condition is that the trilinear form matches.
    # <T, A, B, C*> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C*[i,k]
    # With T[i,j,k] = 1, this is sum_{i,j,k} A[i,j] B[j,k] C*[i,k]
    # Our decomposition gives sum_r (sum_{i,j} U[i,j,r] A[i,j]) (sum_{j,k} V[j,k,r] B[j,k]) (sum_{k,i} W[k,i,r] C*[k,i])
    # If we set C* to the identity of C (i.e., C*[i,k] = C[i,k]), 
    # then the decomposition computes:
    # Val = sum_r (sum_{i,j} U[i,j,r] A[i,j]) (sum_{j,k} V[j,k,r] B[j,k]) (sum_{k,i} W[k,i,r] C[i,k])
    # This is a scalar. It is not C.
    
    # The standard check is:
    # Does sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k?
    # And does sum_r U[i,j,r] V[j',k,r] W[k',i,r] == 0 for j!=j' or k!=k'?
    
    # We can check the second condition by constructing the full tensor of coefficients.
    # Coef[i, j, j', k, k', i'] = sum_r U[i,j,r] V[j',k,r] W[k',i',r]
    # We want Coef[i, j, j, k, k, i] = 1 and 0 otherwise.
    
    # This is expensive for large dimensions, but n=2, m=3, p=9 is small.
    # Size: 2*3*3*9*9*2 = 2916 entries. Rank ~36.
    # 2916 * 36 operations. Very fast.
    
    U_mat = np.array(U).reshape(n, m, rank)
    V_mat = np.array(V).reshape(m, p, rank)
    W_mat = np.array(W).reshape(p, n, rank)
    
    # Construct Coef tensor
    # Coef[i, j, jp, k, kp, ip] = sum_r U[i,j,r] V[jp,k,r] W[kp,ip,r]
    # Using einsum: 'imr,jpr,kqr->imjpkq' ? No.
    # U[i,j,r], V[jp,k,r], W[kp,ip,r]
    # Result indices: i, j, jp, k, kp, ip
    Coef = np.einsum('imr,jpr,kqr->imjpkq', U_mat, V_mat, W_mat)
    
    # Check diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(Coef[i, j, j, k, k, i] - 1.0) > 1e-9:
                    return False
    # Check off-diagonal
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if (j == jp and k == kp and i == ip):
                                continue
                            if abs(Coef[i, j, jp, k, kp, ip]) > 1e-9:
                                return False
    return True

def _block_2x3x1():
    """Rank 3 decomposition for 2x3x1."""
    n, m, p = 2, 3, 1
    rank = 3
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for r in range(m): # r corresponds to j
        for i in range(n):
            U[i*m + r, r] = 1
        V[r*p + 0, r] = 1
        for i in range(n):
            W[0*n + i, r] = 1
    return U, V, W

def _block_2x2x2():
    """Rank 7 Strassen decomposition for 2x2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen multipliers
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i,j,r], V[j,k,r], W[k,i,r]
    
    # r=0 (M1)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=1
    
    # r=1 (M2)
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # r=2 (M3)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # r=3 (M4)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # r=4 (M5)
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # r=5 (M6)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # r=6 (M7)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[1,6]=-1; W[3,6]=1
    
    return U, V, W

def _block_2x2x1():
    """Rank 2 decomposition for 2x2x1."""
    n, m, p = 2, 2, 1
    rank = 2
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for r in range(m):
        for i in range(n):
            U[i*m + r, r] = 1
        V[r*p + 0, r] = 1
        for i in range(n):
            W[0*n + i, r] = 1
    return U, V, W

def _block_2x1x3():
    """Rank 3 decomposition for 2x1x3."""
    n, m, p = 2, 1, 3
    rank = 3
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for r in range(p):
        for i in range(n):
            U[i*m + 0, r] = 1
        V[0*p + r, r] = 1
        for i in range(n):
            W[r*n + i, r] = 1
    return U, V, W

def _block_2x3x3():
    """Combine 2x2x3 and 2x1x3 to get 2x3x3. Rank 9+3=12."""
    # 2x2x3 = 2x2x2 (rank 7) + 2x2x1 (rank 2)
    U_222, V_222, W_222 = _block_2x2x2()
    U_221, V_221, W_221 = _block_2x2x1()
    
    U_223 = np.hstack([U_222, U_221])
    V_223 = np.hstack([V_222, V_221])
    W_223 = np.hstack([W_222, W_221])
    
    # 2x1x3 (rank 3)
    U_213, V_213, W_213 = _block_2x1x3()
    
    # Combine 2x2x3 and 2x1x3
    # U is 2x3. 2x2 part is cols 0,1. 2x1 part is col 2.
    # We need to map indices.
    # U_223 is 4x9. We need 6x12.
    # This block construction is getting complex to merge manually.
    # I will just return the schoolbook for 2x3x3 if this is too hard, 
    # but I can easily do 9 blocks of 2x3x1 (rank 27).
    pass

def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    
    # Use 9 blocks of 2x3x1, rank 3 * 9 = 27
    rank = 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for k_block in range(p):
        U_blk, V_blk, W_blk = _block_2x3x1()
        start = k_block * 3
        U[:, start:start+3] = U_blk
        # V is 3x1 for each block. In full V (3x9), this is column k_block.
        V[:, k_block] = V_blk.flatten()
        # W is 1x2 for each block. In full W (9x2), this is row k_block.
        W[k_block, :] = W_blk.flatten()
        
    sol = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if _verify(sol["U"], sol["V"], sol["W"], n, m, p):
        return sol
        
    # Fallback
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
