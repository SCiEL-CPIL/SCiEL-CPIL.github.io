"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-28 decomposition for (2x4)*(4x4) matrix multiplication.
    
    Strategy: Partition A into two 2x2 blocks and B into four 2x2 blocks,
    then use Strassen (rank 7) for each of the 4 required 2x2*2x2 products.
    
    A (2x4) = [A0 | A1] where A0, A1 are 2x2
    B (4x4) = [[B00, B01], [B10, B11]] where each is 2x2
    
    C = A * B = [A0*B00 + A1*B10 | A0*B01 + A1*B11]
    
    We compute 4 products using Strassen: 4 * 7 = 28 scalar multiplications.
    
    Index conventions:
    - U index for A[i,j]: i*m + j = i*4 + j
    - V index for B[j,k]: j*p + k = j*4 + k  
    - W index for C[i,k]: k*n + i = k*2 + i (official trace convention)
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen for 2x2 * 2x2:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z[0,0] = M1+M4-M5+M7
    # Z[0,1] = M3+M5
    # Z[1,0] = M2+M4
    # Z[1,1] = M1-M2+M3+M6
    #
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    
    def add_strassen(prod_id, x_indices, y_indices, z_indices):
        """
        x_indices: [[x00, x01], [x10, x11]] - flat indices into U
        y_indices: [[y00, y01], [y10, y11]] - flat indices into V
        z_indices: [[z00, z01], [z10, z11]] - flat indices into W
        """
        base = prod_id * 7
        x00, x01 = int(x_indices[0,0]), int(x_indices[0,1])
        x10, x11 = int(x_indices[1,0]), int(x_indices[1,1])
        y00, y01 = int(y_indices[0,0]), int(y_indices[0,1])
        y10, y11 = int(y_indices[1,0]), int(y_indices[1,1])
        z00, z01 = int(z_indices[0,0]), int(z_indices[0,1])
        z10, z11 = int(z_indices[1,0]), int(z_indices[1,1])
        
        # M1 = (x00+x11)*(y00+y11)
        U[x00, base+0] += 1; U[x11, base+0] += 1
        V[y00, base+0] += 1; V[y11, base+0] += 1
        W[z00, base+0] += 1; W[z11, base+0] += 1
        
        # M2 = (x10+x11)*y00
        U[x10, base+1] += 1; U[x11, base+1] += 1
        V[y00, base+1] += 1
        W[z10, base+1] += 1
        
        # M3 = x00*(y01-y11)
        U[x00, base+2] += 1
        V[y01, base+2] += 1; V[y11, base+2] -= 1
        W[z01, base+2] += 1
        
        # M4 = x11*(y10-y00)
        U[x11, base+3] += 1
        V[y10, base+3] += 1; V[y00, base+3] -= 1
        W[z00, base+3] += 1; W[z10, base+3] += 1
        
        # M5 = (x00+x01)*y11
        U[x00, base+4] += 1; U[x01, base+4] += 1
        V[y11, base+4] += 1
        W[z00, base+4] -= 1; W[z01, base+4] += 1
        
        # M6 = (x10-x00)*(y00+y01)
        U[x10, base+5] += 1; U[x00, base+5] -= 1
        V[y00, base+5] += 1; V[y01, base+5] += 1
        W[z11, base+5] += 1
        
        # M7 = (x01-x11)*(y10+y11)
        U[x01, base+6] += 1; U[x11, base+6] -= 1
        V[y10, base+6] += 1; V[y11, base+6] += 1
        W[z00, base+6] += 1
    
    # Flat index mappings:
    # A[i,j] -> i*4+j:
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    #   A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # B[j,k] -> j*4+k:
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    #   B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    #   B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    #   B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    # C[i,k] -> k*2+i:
    #   C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
    #   C[0,2]=4, C[1,2]=5, C[0,3]=6, C[1,3]=7
    
    # Block A0 = A[:, 0:2]: [[A[0,0],A[0,1]],[A[1,0],A[1,1]]] = [[0,1],[4,5]]
    # Block A1 = A[:, 2:4]: [[A[0,2],A[0,3]],[A[1,2],A[1,3]]] = [[2,3],[6,7]]
    # Block B00 = B[0:2, 0:2]: [[B[0,0],B[0,1]],[B[1,0],B[1,1]]] = [[0,1],[4,5]]
    # Block B01 = B[0:2, 2:4]: [[B[0,2],B[0,3]],[B[1,2],B[1,3]]] = [[2,3],[6,7]]
    # Block B10 = B[2:4, 0:2]: [[B[2,0],B[2,1]],[B[3,0],B[3,1]]] = [[8,9],[12,13]]
    # Block B11 = B[2:4, 2:4]: [[B[2,2],B[2,3]],[B[3,2],B[3,3]]] = [[10,11],[14,15]]
    
    # C[:, 0:2] = [[C[0,0],C[0,1]],[C[1,0],C[1,1]]]
    # W indices: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    # So z_indices for C[:,0:2] = [[0,2],[1,3]]
    
    # C[:, 2:4] = [[C[0,2],C[0,3]],[C[1,2],C[1,3]]]
    # W indices: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    # So z_indices for C[:,2:4] = [[4,6],[5,7]]
    
    # Product 1: A0 * B00 -> contributes to C[:, 0:2]
    add_strassen(0,
                 np.array([[0,1],[4,5]]),    # A0
                 np.array([[0,1],[4,5]]),    # B00
                 np.array([[0,2],[1,3]]))    # C[:,0:2]
    
    # Product 2: A1 * B10 -> contributes to C[:, 0:2]
    add_strassen(1,
                 np.array([[2,3],[6,7]]),    # A1
                 np.array([[8,9],[12,13]]),  # B10
                 np.array([[0,2],[1,3]]))    # C[:,0:2]
    
    # Product 3: A0 * B01 -> contributes to C[:, 2:4]
    add_strassen(2,
                 np.array([[0,1],[4,5]]),    # A0
                 np.array([[2,3],[6,7]]),    # B01
                 np.array([[4,6],[5,7]]))    # C[:,2:4]
    
    # Product 4: A1 * B11 -> contributes to C[:, 2:4]
    add_strassen(3,
                 np.array([[2,3],[6,7]]),    # A1
                 np.array([[10,11],[14,15]]),# B11
                 np.array([[4,6],[5,7]]))    # C[:,2:4]
    
    # Verify the decomposition numerically
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{j,k}
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0 if j == k else 0.0
                max_error = max(max_error, abs(val - expected))
    
    if max_error > 1e-10:
        # Fallback to schoolbook rank-32 if Strassen construction has a bug
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
