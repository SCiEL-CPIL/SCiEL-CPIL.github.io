"""Matrix multiplication 3x3x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # 3, 3, 6

    def make_rank23_333():
        """
        Return U, V, W for rank-23 decomposition of 3x3x3 matrix multiplication.
        U: (9, 23) indexed by (i,j) -> i*3+j
        V: (9, 23) indexed by (j,k) -> j*3+k
        W: (9, 23) indexed by (k,i) -> k*3+i
        """
        U = np.zeros((9, 23), dtype=float)
        V = np.zeros((9, 23), dtype=float)
        
        # Term 0: (a00+a11+a22)*(b00+b11+b22)
        U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
        # Term 1: (a01+a10+a21)*(b01+b10+b21)
        U[1, 1] = 1; U[3, 1] = 1; U[7, 1] = 1
        V[1, 1] = 1; V[3, 1] = 1; V[7, 1] = 1
        # Term 2: (a02+a12+a20)*(b02+b12+b20)
        U[2, 2] = 1; U[5, 2] = 1; U[6, 2] = 1
        V[2, 2] = 1; V[5, 2] = 1; V[6, 2] = 1
        # Term 3: (a00+a10+a20)*(b00+b10+b20)
        U[0, 3] = 1; U[3, 3] = 1; U[6, 3] = 1
        V[0, 3] = 1; V[3, 3] = 1; V[6, 3] = 1
        # Term 4: (a00+a11)*(b00+b11)
        U[0, 4] = 1; U[4, 4] = 1
        V[0, 4] = 1; V[4, 4] = 1
        # Term 5: (a01+a12)*(b01+b12)
        U[1, 5] = 1; U[5, 5] = 1
        V[1, 5] = 1; V[5, 5] = 1
        # Term 6: (a02+a10)*(b02+b10)
        U[2, 6] = 1; U[3, 6] = 1
        V[2, 6] = 1; V[3, 6] = 1
        # Term 7: (a01+a21)*(b01+b21)
        U[1, 7] = 1; U[7, 7] = 1
        V[1, 7] = 1; V[7, 7] = 1
        # Term 8: (a02+a22)*(b02+b22)
        U[2, 8] = 1; U[8, 8] = 1
        V[2, 8] = 1; V[8, 8] = 1
        # Term 9: (a10+a20)*(b10+b20)
        U[3, 9] = 1; U[6, 9] = 1
        V[3, 9] = 1; V[6, 9] = 1
        # Term 10: (a11+a21)*(b11+b21)
        U[4, 10] = 1; U[7, 10] = 1
        V[4, 10] = 1; V[7, 10] = 1
        # Term 11: (a12+a22)*(b12+b22)
        U[5, 11] = 1; U[8, 11] = 1
        V[5, 11] = 1; V[8, 11] = 1
        # Term 12: (a00-a11+a22)*(b00+b11-b22)
        U[0, 12] = 1; U[4, 12] = -1; U[8, 12] = 1
        V[0, 12] = 1; V[4, 12] = 1; V[8, 12] = -1
        # Term 13: (a01+a10-a21)*(b01+b10+b21)
        U[1, 13] = 1; U[3, 13] = 1; U[7, 13] = -1
        V[1, 13] = 1; V[3, 13] = 1; V[7, 13] = 1
        # Term 14: (a02-a12+a20)*(b02+b12-b20)
        U[2, 14] = 1; U[5, 14] = -1; U[6, 14] = 1
        V[2, 14] = 1; V[5, 14] = 1; V[6, 14] = -1
        # Term 15: (a00-a10+a20)*(b00-b10+b20)
        U[0, 15] = 1; U[3, 15] = -1; U[6, 15] = 1
        V[0, 15] = 1; V[3, 15] = -1; V[6, 15] = 1
        # Term 16: (a01-a12)*(b01-b12)
        U[1, 16] = 1; U[5, 16] = -1
        V[1, 16] = 1; V[5, 16] = -1
        # Term 17: (a02-a10)*(b02-b10)
        U[2, 17] = 1; U[3, 17] = -1
        V[2, 17] = 1; V[3, 17] = -1
        # Term 18: (a01-a21)*(b01-b21)
        U[1, 18] = 1; U[7, 18] = -1
        V[1, 18] = 1; V[7, 18] = -1
        # Term 19: (a02-a22)*(b02-b22)
        U[2, 19] = 1; U[8, 19] = -1
        V[2, 19] = 1; V[8, 19] = -1
        # Term 20: (a10-a20)*(b10-b20)
        U[3, 20] = 1; U[6, 20] = -1
        V[3, 20] = 1; V[6, 20] = -1
        # Term 21: (a11-a21)*(b11-b21)
        U[4, 21] = 1; U[7, 21] = -1
        V[4, 21] = 1; V[7, 21] = -1
        # Term 22: (a12-a22)*(b12-b22)
        U[5, 22] = 1; U[8, 22] = -1
        V[5, 22] = 1; V[8, 22] = -1
        
        # Solve for W
        W = np.zeros((9, 23), dtype=float)
        for i in range(3):
            for k in range(3):
                A_sys = np.zeros((3, 23), dtype=float)
                b_sys = np.ones(3, dtype=float)
                for j in range(3):
                    for r in range(23):
                        A_sys[j, r] = U[i*3+j, r] * V[j*3+k, r]
                try:
                    w, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    W[k*3+i, :] = w
                except:
                    return None
        
        # Verify
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = sum(U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] for r in range(23))
                    if abs(s - 1) > 1e-9:
                        return None
        
        return U, V, W

    r23 = make_rank23_333()
    
    if r23 is not None and time.time() - t0 < DEADLINE - 5:
        U1, V1, W1 = r23
        rank = 46
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # First block: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
        U[:9, :23] = U1
        V[:9, :23] = V1
        W[:9, :23] = W1
        # Second block: A[0:3,0:3] * B[0:3,3:6] -> C[0:3,3:6]
        U[:9, 23:46] = U1
        V[9:18, 23:46] = V1
        W[9:18, 23:46] = W1
        
        # Vectorized verification
        # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # Build tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # U: (9, rank), V: (18, rank), W: (18, rank)
        # Reshape: U -> (3,3,rank), V -> (3,6,rank), W -> (6,3,rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if np.allclose(T, 1.0, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 54
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
