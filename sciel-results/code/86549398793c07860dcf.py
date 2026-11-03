"""Program to find a rank-minimal exact tensor decomposition for 2x4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    def verify(U, V, W, rank):
        """Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition, each of shape (4, 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    def build_strassen_28():
        """Build the rank-28 Strassen-based decomposition with correct index mapping."""
        U_full = np.zeros((8, 28))
        V_full = np.zeros((16, 28))
        W_full = np.zeros((8, 28))
        
        # Block 1: A1 * B11 -> C11
        # A1: i in {0,1}, j in {0,1} -> U rows 0,1,4,5
        # B11: j in {0,1}, k in {0,1} -> V rows 0,1,4,5
        # C11: k in {0,1}, i in {0,1} -> W rows 0,1,2,3
        U1, V1, W1 = strassen_2x2x2()
        
        # Map U1 (rows 0,1,2,3 for j=0,0,1,1) to U_full (rows 0,1,4,5)
        U_full[0, 0:7] = U1[0, :] # i=0, j=0
        U_full[1, 0:7] = U1[1, :] # i=0, j=1
        U_full[4, 0:7] = U1[2, :] # i=1, j=0
        U_full[5, 0:7] = U1[3, :] # i=1, j=1
        
        # Map V1 (rows 0,1,2,3 for j=0,0,1,1) to V_full (rows 0,1,4,5)
        V_full[0, 0:7] = V1[0, :] # j=0, k=0
        V_full[1, 0:7] = V1[1, :] # j=0, k=1
        V_full[4, 0:7] = V1[2, :] # j=1, k=0
        V_full[5, 0:7] = V1[3, :] # j=1, k=1
        
        # Map W1 (rows 0,1,2,3 for k=0,0,1,1) to W_full (rows 0,1,2,3)
        W_full[0, 0:7] = W1[0, :] # k=0, i=0
        W_full[1, 0:7] = W1[1, :] # k=0, i=1
        W_full[2, 0:7] = W1[2, :] # k=1, i=0
        W_full[3, 0:7] = W1[3, :] # k=1, i=1
        
        # Block 2: A2 * B21 -> C11 (adds to C11)
        # A2: i in {0,1}, j in {2,3} -> U rows 2,3,6,7
        # B21: j in {2,3}, k in {0,1} -> V rows 8,9,12,13
        # C11: k in {0,1}, i in {0,1} -> W rows 0,1,2,3
        U2, V2, W2 = strassen_2x2x2()
        
        U_full[2, 7:14] = U2[0, :] # i=0, j=2
        U_full[3, 7:14] = U2[1, :] # i=0, j=3
        U_full[6, 7:14] = U2[2, :] # i=1, j=2
        U_full[7, 7:14] = U2[3, :] # i=1, j=3
        
        V_full[8, 7:14] = V2[0, :] # j=2, k=0
        V_full[9, 7:14] = V2[1, :] # j=2, k=1
        V_full[12, 7:14] = V2[2, :] # j=3, k=0
        V_full[13, 7:14] = V2[3, :] # j=3, k=1
        
        W_full[0, 7:14] = W2[0, :] # k=0, i=0
        W_full[1, 7:14] = W2[1, :] # k=0, i=1
        W_full[2, 7:14] = W2[2, :] # k=1, i=0
        W_full[3, 7:14] = W2[3, :] # k=1, i=1
        
        # Block 3: A1 * B12 -> C12
        # A1: i in {0,1}, j in {0,1} -> U rows 0,1,4,5
        # B12: j in {0,1}, k in {2,3} -> V rows 2,3,6,7
        # C12: k in {2,3}, i in {0,1} -> W rows 4,5,6,7
        U3, V3, W3 = strassen_2x2x2()
        
        U_full[0, 14:21] = U3[0, :]
        U_full[1, 14:21] = U3[1, :]
        U_full[4, 14:21] = U3[2, :]
        U_full[5, 14:21] = U3[3, :]
        
        V_full[2, 14:21] = V3[0, :] # j=0, k=2
        V_full[3, 14:21] = V3[1, :] # j=0, k=3
        V_full[6, 14:21] = V3[2, :] # j=1, k=2
        V_full[7, 14:21] = V3[3, :] # j=1, k=3
        
        W_full[4, 14:21] = W3[0, :] # k=2, i=0
        W_full[5, 14:21] = W3[1, :] # k=2, i=1
        W_full[6, 14:21] = W3[2, :] # k=3, i=0
        W_full[7, 14:21] = W3[3, :] # k=3, i=1
        
        # Block 4: A2 * B22 -> C12 (adds to C12)
        # A2: i in {0,1}, j in {2,3} -> U rows 2,3,6,7
        # B22: j in {2,3}, k in {2,3} -> V rows 10,11,14,15
        # C12: k in {2,3}, i in {0,1} -> W rows 4,5,6,7
        U4, V4, W4 = strassen_2x2x2()
        
        U_full[2, 21:28] = U4[0, :]
        U_full[3, 21:28] = U4[1, :]
        U_full[6, 21:28] = U4[2, :]
        U_full[7, 21:28] = U4[3, :]
        
        V_full[10, 21:28] = V4[0, :] # j=2, k=2
        V_full[11, 21:28] = V4[1, :] # j=2, k=3
        V_full[14, 21:28] = V4[2, :] # j=3, k=2
        V_full[15, 21:28] = V4[3, :] # j=3, k=3
        
        W_full[4, 21:28] = W4[0, :]
        W_full[5, 21:28] = W4[1, :]
        W_full[6, 21:28] = W4[2, :]
        W_full[7, 21:28] = W4[3, :]
        
        return U_full, V_full, W_full
    
    best_result = None
    
    # Try rank 28
    U28, V28, W28 = build_strassen_28()
    if verify(U28, V28, W28, 28):
        best_result = (U28, V28, W28, 28)
    else:
        # Schoolbook fallback
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        best_result = (U_fb, V_fb, W_fb, rank)
    
    U_best, V_best, W_best, rank_best = best_result
    
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
