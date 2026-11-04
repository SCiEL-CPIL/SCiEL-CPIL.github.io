"""Matrix multiplication 2x4x8 via optimized tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Build the multiplication tensor T[i,j,k] = delta_{i,k} for the matmul
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 if this term contributes, else 0
    # Actually, the tensor for matmul is: T[i,j,k] where A is (n,m), B is (m,p), C is (n,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    
    # The tensor is T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # We want to write T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # Let's try to find a low-rank decomposition using the structure.
    # For 2x4x8, let's try rank 32 by sharing factors.
    
    # Strategy: For each (i-block, j-block) pair, we have 4 k-blocks.
    # The A-side factors are the same for all 4 k-blocks of a given (i,j) block.
    # So we can precompute 7 A-side combinations per (i,j) block.
    # For each k-block, we need 7 B-side combinations.
    # But the W factors will be different for each k-block.
    
    # Actually, let's just try to construct a rank-32 decomposition directly.
    # Split j into {0,1} and {2,3}, k into four blocks of 2.
    # For each (bj, bk) pair, use 4 terms instead of 7 by exploiting structure.
    
    # Let me try a different approach: use the known result that 2x4x8 can be done
    # in rank 32 by using 4 terms per 2x2x2 block with shared structure.
    
    # Actually, let's just verify if rank 56 is correct and try to reduce.
    # The theoretical lower bound for 2x4x8 is at least 2*4*8 / something...
    # Let's just try to find the minimum rank numerically.
    
    # Build tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try SVD to estimate rank
    T_reshaped = T.reshape(n*m, m*p * p*n)
    U_svd, s, Vt_svd = np.linalg.svd(T_reshaped, full_matrices=False)
    # The numerical rank
    tol = 1e-10
    numerical_rank = np.sum(s > tol)
    
    # Let's try to find a decomposition with small integer entries.
    # Start with rank 32 and try to solve.
    
    best_rank = 56
    best_decomp = None
    
    # First, let's just return the known rank-56 Strassen decomposition
    # and verify it works.
    
    def _strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen (rank 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        U[0,0] += 1; U[3,0] += 1
        V[0,0] += 1; V[3,0] += 1
        W[0,0] += 1; W[3,0] += 1
        
        # M2 = (c+d)e
        U[2,1] += 1; U[3,1] += 1
        V[0,1] += 1
        W[2,1] += 1; W[3,1] -= 1
        
        # M3 = a(f-h)
        U[0,2] += 1
        V[1,2] += 1; V[3,2] -= 1
        W[1,2] += 1; W[3,2] += 1
        
        # M4 = d(g-e)
        U[3,3] += 1
        V[2,3] += 1; V[0,3] -= 1
        W[0,3] += 1; W[2,3] += 1
        
        # M5 = (a+b)h
        U[0,4] += 1; U[1,4] += 1
        V[3,4] += 1
        W[0,4] -= 1; W[1,4] += 1
        
        # M6 = (c-a)(e+f)
        U[2,5] += 1; U[0,5] -= 1
        V[0,5] += 1; V[1,5] += 1
        W[3,5] += 1
        
        # M7 = (b-d)(g+h)
        U[1,6] += 1; U[3,6] -= 1
        V[2,6] += 1; V[3,6] += 1
        W[0,6] += 1
        
        return U, V, W
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # Build rank-56 decomposition
    U = np.zeros((n*m, 56))
    V = np.zeros((m*p, 56))
    W = np.zeros((p*n, 56))
    
    col = 0
    for bi in range(1):  # 1 i-block (size 2)
        for bj in range(2):  # 2 j-blocks (size 2 each)
            for bk in range(4):  # 4 k-blocks (size 2 each)
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                for r in range(7):
                    # Map local indices to global
                    for li in range(2):
                        gi = i0 + li
                        for lj in range(2):
                            gj = j0 + lj
                            U[gi*m + gj, col] += U_str[li*2+lj, r]
                    
                    for lj in range(2):
                        gj = j0 + lj
                        for lk in range(2):
                            gk = k0 + lk
                            V[gj*p + gk, col] += V_str[lj*2+lk, r]
                    
                    for lk in range(2):
                        gk = k0 + lk
                        for li in range(2):
                            gi = i0 + li
                            W[gk*n + gi, col] += W_str[lk*2+li, r]
                    
                    col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for k in range(p):
                expected = 0
                for j in range(m):
                    # T[i,j,k] = 1
                    contrib = 0
                    for r in range(U_mat.shape[1]):
                        contrib += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    expected += contrib
                if abs(expected - 1) > 1e-9:
                    return False
        return True
    
    if not verify(U, V, W):
        # Fallback to simple outer product (rank n*m*p = 64)
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        for idx in range(n*m*p):
            i = idx // (m*p)
            j = (idx // p) % m
            k = idx % p
            U[i*m+j, idx] = 1
            V[j*p+k, idx] = 1
            W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
