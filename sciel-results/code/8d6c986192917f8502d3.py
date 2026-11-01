"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen_block22():
    # Strassen 2x2x2: 7 terms.
    # Returns U (4,7), V (4,7), W (4,7) for (i,j) -> i*2+j, (j,k)->j*2+k, (k,i)->k*2+i
    # Standard Strassen formulas for C = A*B where A is 2x2, B is 2x2.
    # C[0,0] = M5 - M4 + M2 - M3
    # C[0,1] = M1 + M5
    # C[1,0] = M4 + M6
    # C[1,1] = M5 - M1 + M3 + M7
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # U[i,j] corresponds to A[i,j], V[j,k] to B[j,k], W[k,i] to C[k,i] in output
    # Output convention: W[k*n+i] with n=2 here, so W[k*2+i]
    
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1: (A00+A11)*(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    # contributes to C[0,0]: -M1, C[0,1]: +M1, C[1,0]: 0, C[1,1]: -M1
    W[0*2+0, 0] = -1  # C[0,0]
    W[0*2+1, 0] = 1   # C[0,1]
    W[1*2+0, 0] = 0   # C[1,0]
    W[1*2+1, 0] = -1  # C[1,1]
    
    # M2: (A10+A11)*B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    # C[0,0]: +M2, C[0,1]: 0, C[1,0]: 0, C[1,1]: 0
    W[0*2+0, 1] = 1
    W[0*2+1, 1] = 0
    W[1*2+0, 1] = 0
    W[1*2+1, 1] = 0
    
    # M3: A00*(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    # C[0,0]: -M3, C[0,1]: 0, C[1,0]: 0, C[1,1]: +M3
    W[0*2+0, 2] = -1
    W[0*2+1, 2] = 0
    W[1*2+0, 2] = 0
    W[1*2+1, 2] = 1
    
    # M4: A11*(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    # C[0,0]: -M4, C[0,1]: 0, C[1,0]: +M4, C[1,1]: 0
    W[0*2+0, 3] = -1
    W[0*2+1, 3] = 0
    W[1*2+0, 3] = 1
    W[1*2+1, 3] = 0
    
    # M5: (A00+A01)*B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    # C[0,0]: +M5, C[0,1]: +M5, C[1,0]: 0, C[1,1]: +M5
    W[0*2+0, 4] = 1
    W[0*2+1, 4] = 1
    W[1*2+0, 4] = 0
    W[1*2+1, 4] = 1
    
    # M6: (A10-A00)*(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    # C[0,0]: 0, C[0,1]: 0, C[1,0]: +M6, C[1,1]: 0
    W[0*2+0, 5] = 0
    W[0*2+1, 5] = 0
    W[1*2+0, 5] = 1
    W[1*2+1, 5] = 0
    
    # M7: (A01-A11)*(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    # C[0,0]: 0, C[0,1]: 0, C[1,0]: 0, C[1,1]: +M7
    W[0*2+0, 6] = 0
    W[0*2+1, 6] = 0
    W[1*2+0, 6] = 0
    W[1*2+1, 6] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Should equal 1 if i,j,k all in range, else 0. But actually for matmul:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = 1 for the multiplication pattern.
    # The decomposition gives: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 
    # 1 when this corresponds to a valid multiplication term, i.e., always 1 for the
    # standard basis. Actually the tensor for matmul is T[i,j,k]=1 for all valid i,j,k.
    # Wait - the tensor for matrix multiplication (n x m) * (m x p) is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) triple contributes with weight 1.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # This should equal 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U_arr[i*m+j], V_arr[j*p+k]), W_arr[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    deadline = time.time() + 140.0  # safety margin
    
    # Strategy: Split the 2x4x6 problem into blocks.
    # n=2 is already 2x2 blocks (1 block of size 2 in dim i)
    # m=4 can be split into 2 blocks of size 2
    # p=6 can be split into 3 blocks of size 2
    
    # For each block (i_block, j_block, k_block) of size (2,2,2), we can use Strassen (rank 7).
    # Total blocks: (2/2) * (4/2) * (6/2) = 1 * 2 * 3 = 6 blocks
    # Total rank: 6 * 7 = 42
    
    # Let's construct this block-diagonal Strassen decomposition.
    # i ranges 0..1 (n=2), j ranges 0..3 (m=4), k ranges 0..5 (p=6)
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for ib in range(n // 2):  # 0
        for jb in range(m // 2):  # 0, 1
            for kb in range(p // 2):  # 0, 1, 2
                Ub, Vb, Wb = _strassen_block22()
                U_blocks.append(Ub)
                V_blocks.append(Vb)
                W_blocks.append(Wb)
    
    # Now assemble. For block (ib, jb, kb):
    # U rows correspond to i in [ib*2, ib*2+1], j in [jb*2, jb*2+1]
    #   so U global row = i*m + j where i = ib*2+di, j = jb*2+dj
    # V rows correspond to j in [jb*2, jb*2+1], k in [kb*2, kb*2+1]
    #   so V global row = j*p + k where j = jb*2+dj, k = kb*2+dk
    # W rows correspond to k in [kb*2, kb*2+1], i in [ib*2, ib*2+1]
    #   so W global row = k*n + i where k = kb*2+dk, i = ib*2+di
    
    total_rank = sum(Ub.shape[1] for Ub in U_blocks)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for ib in range(n // 2):
        for jb in range(m // 2):
            for kb in range(p // 2):
                block_idx = ib * (m//2) * (p//2) + jb * (p//2) + kb
                Ub, Vb, Wb = U_blocks[block_idx], V_blocks[block_idx], W_blocks[block_idx]
                r = Ub.shape[1]
                
                for di in range(2):
                    for dj in range(2):
                        i = ib * 2 + di
                        j = jb * 2 + dj
                        U[i * m + j, col_offset:col_offset+r] = Ub[di*2+dj, :]
                
                for dj in range(2):
                    for dk in range(2):
                        j = jb * 2 + dj
                        k = kb * 2 + dk
                        V[j * p + k, col_offset:col_offset+r] = Vb[dj*2+dk, :]
                
                for dk in range(2):
                    for di in range(2):
                        k = kb * 2 + dk
                        i = ib * 2 + di
                        W[k * n + i, col_offset:col_offset+r] = Wb[dk*2+di, :]
                
                col_offset += r
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
