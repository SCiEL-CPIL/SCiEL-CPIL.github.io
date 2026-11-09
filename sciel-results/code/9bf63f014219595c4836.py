"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135.0
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Schoolbook rank
    rank_school = n * m * p  # 56
    
    # Schoolbook solution
    def schoolbook(n, m, p):
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
        return U, V, W
    
    best_rank = rank_school
    best_U, best_V, best_W = schoolbook(n, m, p)
    
    # Fast verification using numpy with exact integer arithmetic
    def verify_fast(U, V, W, n, m, p):
        """Check if U, V, W form a valid decomposition using numpy."""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        
        # The condition is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
        # We can compute this as a tensor contraction.
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is not a standard matmul.
        
        # Efficient way:
        # For each r, we have a rank-1 tensor u_r v_r w_r.
        # We need to check if the sum of these rank-1 tensors equals the multiplication tensor.
        
        # Let's just iterate, it's fast enough for r=50 and 56 entries.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(r):
                        s += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
                    if s != 1:
                        return False
        return True

    # Strategy: Block-wise Strassen.
    # Split A (2x4) into A_left (2x2) and A_right (2x2).
    # Split B (4x7) into B_top (2x7) and B_bottom (2x7).
    # C = A_left B_top + A_right B_bottom.
    # Rank of 2x2 * 2x7 multiplication:
    # Split 2x7 into three 2x2 blocks and one 2x1 block.
    # Rank = 3 * rank(2x2*2x2) + rank(2x2*2x1)
    # rank(2x2*2x2) = 7 (Strassen)
    # rank(2x2*2x1) = 4 (Schoolbook, as 2x2*1x2 is rank 4)
    # Total rank per block = 3 * 7 + 4 = 25.
    # Two blocks -> Total rank = 50.
    
    rank_50 = 50
    U = [[0] * rank_50 for _ in range(n * m)]
    V = [[0] * rank_50 for _ in range(m * p)]
    W = [[0] * rank_50 for _ in range(p * n)]
    
    # Helper to add a Strassen block for 2x2 * 2x2
    # Indices:
    # U: i (0,1), j (0,1) -> global row i*m + j
    # V: j (0,1), k (0,1) -> global row j*p + k
    # W: k (0,1), i (0,1) -> global row k*n + i
    
    def add_strassen_2x2_2x2(offset, i_base, j_base, k_base):
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        U[i_base + 0*m + 0][offset+0] += 1
        U[i_base + 1*m + 1][offset+0] += 1
        V[j_base + 0*p + 0][offset+0] += 1
        V[j_base + 1*p + 1][offset+0] += 1
        W[k_base + 0*n + 0][offset+0] += 1
        W[k_base + 1*n + 1][offset+0] += 1
        
        # M2 = (A21+A22)B11 -> C21, C22
        U[i_base + 1*m + 0][offset+1] += 1
        U[i_base + 1*m + 1][offset+1] += 1
        V[j_base + 0*p + 0][offset+1] += 1
        W[k_base + 0*n + 1][offset+1] += 1
        W[k_base + 1*n + 1][offset+1] -= 1
        
        # M3 = A11(B21-B22) -> C12, C22
        U[i_base + 0*m + 0][offset+2] += 1
        V[j_base + 1*p + 0][offset+2] += 1
        V[j_base + 1*p + 1][offset+2] -= 1
        W[k_base + 1*n + 0][offset+2] += 1
        W[k_base + 1*n + 1][offset+2] += 1
        
        # M4 = A22(B11-B12) -> C11, C21
        U[i_base + 1*m + 1][offset+3] += 1
        V[j_base + 0*p + 0][offset+3] += 1
        V[j_base + 0*p + 1][offset+3] -= 1
        W[k_base + 0*n + 0][offset+3] += 1
        W[k_base + 0*n + 1][offset+3] += 1
        
        # M5 = (A11+A12)B22 -> C11, C12
        U[i_base + 0*m + 0][offset+4] += 1
        U[i_base + 0*m + 1][offset+4] += 1
        V[j_base + 1*p + 1][offset+4] += 1
        W[k_base + 0*n + 0][offset+4] -= 1
        W[k_base + 1*n + 0][offset+4] += 1
        
        # M6 = (A21-A11)(B11+B12) -> C22
        U[i_base + 1*m + 0][offset+5] += 1
        U[i_base + 0*m + 0][offset+5] -= 1
        V[j_base + 0*p + 0][offset+5] += 1
        V[j_base + 0*p + 1][offset+5] += 1
        W[k_base + 1*n + 1][offset+5] += 1
        
        # M7 = (A12-A22)(B21+B22) -> C11
        U[i_base + 0*m + 1][offset+6] += 1
        U[i_base + 1*m + 1][offset+6] -= 1
        V[j_base + 1*p + 0][offset+6] += 1
        V[j_base + 1*p + 1][offset+6] += 1
        W[k_base + 0*n + 0][offset+6] += 1

    def add_schoolbook_2x2_2x1(offset, i_base, j_base, k_base):
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        
        # Term 1: A11*B1 -> C1
        U[i_base + 0*m + 0][offset+0] += 1
        V[j_base + 0*p + 0][offset+0] += 1
        W[k_base + 0*n + 0][offset+0] += 1
        
        # Term 2: A12*B2 -> C1
        U[i_base + 0*m + 1][offset+1] += 1
        V[j_base + 1*p + 0][offset+1] += 1
        W[k_base + 0*n + 0][offset+1] += 1
        
        # Term 3: A21*B1 -> C2
        U[i_base + 1*m + 0][offset+2] += 1
        V[j_base + 0*p + 0][offset+2] += 1
        W[k_base + 0*n + 1][offset+2] += 1
        
        # Term 4: A22*B2 -> C2
        U[i_base + 1*m + 1][offset+3] += 1
        V[j_base + 1*p + 0][offset+3] += 1
        W[k_base + 0*n + 1][offset+3] += 1

    # Construct the rank 50 solution
    # Block 1: A_left (i=0..1, j=0..1) * B_top (j=0..1, k=0..6)
    # i_base=0, j_base=0
    # k groups: 0-1, 2-3, 4-5, 6
    add_strassen_2x2_2x2(0, 0, 0, 0)
    add_strassen_2x2_2x2(7, 0, 0, 2)
    add_strassen_2x2_2x2(14, 0, 0, 4)
    add_schoolbook_2x2_2x1(21, 0, 0, 6)
    
    # Block 2: A_right (i=0..1, j=2..3) * B_bottom (j=2..3, k=0..6)
    # i_base=0, j_base=2
    add_strassen_2x2_2x2(25, 0, 2, 0)
    add_strassen_2x2_2x2(32, 0, 2, 2)
    add_strassen_2x2_2x2(39, 0, 2, 4)
    add_schoolbook_2x2_2x1(46, 0, 2, 6)
    
    # Verify
    if verify_fast(U, V, W, n, m, p):
        best_rank = 50
        best_U, best_V, best_W = U, V, W
    else:
        # Fallback
        best_rank = rank_school
        best_U, best_V, best_W = schoolbook(n, m, p)

    # Convert to the required format
    r = best_rank
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
