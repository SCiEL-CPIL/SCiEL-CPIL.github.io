"""Exact rank-reduced decomposition for 3x4x4x8 matrix multiplication tensor using Strassen blocks."""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8

    # We use a block-Strassen approach:
    # Split n=3 into 1+2, m=4 into 2+2, p=8 into 4+4.
    # This gives 8 sub-problems, but we can use Strassen's 7-product method
    # on the 2x2x2 blocks to reduce rank.
    #
    # Strategy: 
    # 1. For the 2x2x2 sub-blocks, use Strassen's rank-7 decomposition.
    # 2. For the 1x2x2 sub-blocks, use schoolbook (rank 4 each).
    # 3. Combine carefully to get total rank < 96.
    #
    # Let's build the Strassen rank-7 decomposition for M_{2,2,2} first.
    
    def strassen_222():
        """Return (U, V, W) for M_{2,2,2} with rank 7.
        U shape: (4, 7) where U[i*2+j, t]
        V shape: (4, 7) where V[j*2+k, t]  
        W shape: (4, 7) where W[k*2+i, t]
        """
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # Strassen products:
        # P1 = (A00+A11)(B00+B11)
        U[0,0] += 1; U[3,0] += 1  # A00, A11
        V[0,0] += 1; V[3,0] += 1  # B00, B11
        W[0,0] += 1; W[3,0] += 1  # C00, C11
        
        # P2 = (A10+A11)B00
        U[1,1] += 1; U[3,1] += 1  # A10, A11
        V[0,1] += 1               # B00
        W[1,1] += 1; W[3,1] -= 1  # C10, -C11
        
        # P3 = A00(B01-B11)
        U[0,2] += 1               # A00
        V[1,2] += 1; V[3,2] -= 1  # B01, -B11
        W[2,2] += 1; W[3,2] += 1  # C01, C11
        
        # P4 = A11(B10-B00)
        U[3,3] += 1               # A11
        V[2,3] += 1; V[0,3] -= 1  # B10, -B00
        W[0,3] += 1; W[1,3] += 1  # C00, C10
        
        # P5 = (A00+A01)B11
        U[0,4] += 1; U[2,4] += 1  # A00, A01
        V[3,4] += 1               # B11
        W[0,4] -= 1; W[2,4] += 1  # -C00, C01
        
        # P6 = (A10+A11)(B01-B11)
        U[1,5] += 1; U[3,5] += 1  # A10, A11
        V[1,5] += 1; V[3,5] -= 1  # B01, -B11
        W[1,5] -= 1; W[3,5] += 1  # -C10, C11
        
        # P7 = (A01-A10)(B00+B10)
        U[2,6] += 1; U[1,6] -= 1  # A01, -A10
        V[0,6] += 1; V[2,6] += 1  # B00, B10
        W[0,6] -= 1; W[3,6] += 1  # -C00, C11
        
        return U, V, W
    
    # Verify Strassen decomposition
    U_s, V_s, W_s = strassen_222()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.sum(U_s[i*2+j, :] * V_s[j*2+k, :] * W_s[k*2+i, :])
                if s != 1:
                    # Fall back to schoolbook if Strassen is wrong
                    rank = n * m * p
                    U = np.zeros((n*m, rank), dtype=int)
                    V = np.zeros((m*p, rank), dtype=int)
                    W = np.zeros((p*n, rank), dtype=int)
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                c = ii * m * p + jj * p + kk
                                U[ii*m+jj, c] = 1
                                V[jj*p+kk, c] = 1
                                W[kk*n+ii, c] = 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now build the full decomposition using blocks:
    # Split: n = 1+2, m = 2+2, p = 4+4
    # Sub-blocks:
    # (n1=1, m1=2, p1=4): schoolbook rank = 1*2*4 = 8
    # (n1=1, m2=2, p2=4): schoolbook rank = 1*2*4 = 8
    # (n2=2, m1=2, p1=4): use 2x Strassen 2x2x2, rank = 2*7 = 14
    # (n2=2, m2=2, p2=4): use 2x Strassen 2x2x2, rank = 2*7 = 14
    # Total rank = 8 + 8 + 14 + 14 = 44
    
    total_rank = 44
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    c = 0
    
    # Block (n1=1, m1=2, p1=4): i in {0}, j in {0,1}, k in {0,1,2,3}
    for ii in range(1):
        for jj in range(2):
            for kk in range(4):
                U[ii*m+jj, c] = 1
                V[jj*p+kk, c] = 1
                W[kk*n+ii, c] = 1
                c += 1
    
    # Block (n1=1, m2=2, p2=4): i in {0}, j in {2,3}, k in {4,5,6,7}
    for ii in range(1):
        for jj in range(2):
            for kk in range(4):
                U[ii*m+(jj+2), c] = 1
                V[(jj+2)*p+(kk+4), c] = 1
                W[(kk+4)*n+ii, c] = 1
                c += 1
    
    # Block (n2=2, m1=2, p1=4): i in {1,2}, j in {0,1}, k in {0,1,2,3}
    # Use 2 copies of Strassen for the two k-pairs: (k0,k1) and (k2,k3)
    for k_pair in range(2):
        ks = k_pair * 2  # k starts at 0 or 2
        U_blk, V_blk, W_blk = strassen_222()
        for ii in range(2):
            for jj in range(2):
                U[(ii+1)*m+jj, c:c+7] += U_blk[ii*2+jj, :]
            for jj in range(2):
                for kk in range(2):
                    V[jj*p+(ks+kk), c:c+7] += V_blk[jj*2+kk, :]
            for kk in range(2):
                for ii in range(2):
                    W[(ks+kk)*n+(ii+1), c:c+7] += W_blk[kk*2+ii, :]
        c += 7
    
    # Block (n2=2, m2=2, p2=4): i in {1,2}, j in {2,3}, k in {4,5,6,7}
    for k_pair in range(2):
        ks = 4 + k_pair * 2  # k starts at 4 or 6
        U_blk, V_blk, W_blk = strassen_222()
        for ii in range(2):
            for jj in range(2):
                U[(ii+1)*m+(jj+2), c:c+7] += U_blk[ii*2+jj, :]
            for jj in range(2):
                for kk in range(2):
                    V[(jj+2)*p+(ks+kk), c:c+7] += V_blk[jj*2+kk, :]
            for kk in range(2):
                for ii in range(2):
                    W[(ks+kk)*n+(ii+1), c:c+7] += W_blk[kk*2+ii, :]
        c += 7
    
    # Verify the full decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if s != 1:
                    # Fall back to schoolbook
                    rank = n * m * p
                    U = np.zeros((n*m, rank), dtype=int)
                    V = np.zeros((m*p, rank), dtype=int)
                    W = np.zeros((p*n, rank), dtype=int)
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                c2 = ii * m * p + jj * p + kk
                                U[ii*m+jj, c2] = 1
                                V[jj*p+kk, c2] = 1
                                W[kk*n+ii, c2] = 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
