"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import itertools

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Indices: A[i,j] i in {0,1}, j in {0,1}; B[j,k] j in {0,1}, k in {0,1}
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # U row index: i*2+j (0..3)
    # V row index: j*2+k (0..3)
    # W row index: k*2+i (0..3)
    
    # Strassen products M1..M7
    # Each M is a product of a linear combo of A entries and a linear combo of B entries.
    # U gives coefficients for A, V gives coefficients for B, W gives how M contributes to C.
    
    U = [[0]*7 for _ in range(4)]  # 4 rows (i*2+j), 7 cols
    V = [[0]*7 for _ in range(4)]  # 4 rows (j*2+k), 7 cols
    W = [[0]*7 for _ in range(4)]  # 4 rows (k*2+i), 7 cols
    
    # M1 = (A00+A11)(B00+B11)
    # A00 is U[0,0]=1, A11 is U[3,0]=1
    # B00 is V[0,0]=1, B11 is V[3,0]=1
    # M1 contributes to C00 (+1), C11 (+1)
    # C00 = W[0*2+0, 0] = W[0,0] += 1
    # C11 = W[1*2+1, 0] = W[3,0] += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # A10 is U[2,1]=1, A11 is U[3,1]=1
    # B00 is V[0,1]=1
    # M2 contributes to C10 (+1), C11 (-1)
    # C10 = W[0*2+1, 1] = W[1,1] += 1
    # C11 = W[1*2+1, 1] = W[3,1] -= 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    # A00 is U[0,2]=1
    # B01 is V[1,2]=1, B11 is V[3,2]=-1
    # M3 contributes to C01 (+1), C11 (+1)
    # C01 = W[1*2+0, 2] = W[2,2] += 1
    # C11 = W[1*2+1, 2] = W[3,2] += 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    # A11 is U[3,3]=1
    # B10 is V[2,3]=1, B00 is V[0,3]=-1
    # M4 contributes to C00 (+1), C10 (+1)
    # C00 = W[0*2+0, 3] = W[0,3] += 1
    # C10 = W[0*2+1, 3] = W[1,3] += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    # A00 is U[0,4]=1, A01 is U[1,4]=1
    # B11 is V[3,4]=1
    # M5 contributes to C00 (-1), C01 (+1)
    # C00 = W[0*2+0, 4] = W[0,4] -= 1
    # C01 = W[1*2+0, 4] = W[2,4] += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # A10 is U[2,5]=1, A00 is U[0,5]=-1
    # B00 is V[0,5]=1, B01 is V[1,5]=1
    # M6 contributes to C11 (+1)
    # C11 = W[1*2+1, 5] = W[3,5] += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01 is U[1,6]=1, A11 is U[3,6]=-1
    # B10 is V[2,6]=1, B11 is V[3,6]=1
    # M7 contributes to C00 (+1), C10 (-1)
    # C00 = W[0*2+0, 6] = W[0,6] += 1
    # C10 = W[0*2+1, 6] = W[1,6] -= 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # A is n x m = 2x4, B is m x p = 4x8, C is n x p = 2x8
    # 
    # Partition: 
    # A = [A_i0 | A_i1] where A_ij is 2x2, i in {0,1} (row block), j in {0,1} (col block)
    # B = [B_j0; B_j1] where B_jk is 2x4... 
    # 
    # Better: split m=4 into 2 groups of 2, and p=8 into 4 groups of 2.
    # A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B[j, k] where j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    # C[i, k] = sum_j A[i,j] * B[j,k]
    #
    # Group j into pairs: j = 2*jj + jj_rem, jj in {0,1}, jj_rem in {0,1}
    # Group k into pairs: k = 2*kk + kk_rem, kk in {0,1,2,3}, kk_rem in {0,1}
    #
    # For each (i, jj, kk): we have a 2x2 * 2x2 subproblem
    # A_sub[i, jj_rem] * B_sub[jj_rem, kk_rem] summed over jj_rem
    # This is a 2x2 * 2x2 multiplication.
    #
    # Total subproblems: 2 (i) * 2 (jj) * 4 (kk) = 16? No...
    # 
    # Wait: i is fixed (2 choices), jj is the j-group (2 choices), kk is the k-group (4 choices)
    # For each (i, jj, kk), we multiply a 2x2 matrix (A rows i, cols 2*jj..2*jj+1)
    # by a 2x2 matrix (B rows 2*jj..2*jj+1, cols 2*kk..2*kk+1)
    # Result contributes to C[i, 2*kk..2*kk+1]
    #
    # So 2 * 2 * 4 = 16 subproblems, each rank 7 = 112? That's worse.
    #
    # Let me reconsider. The finding said 8 subproblems of 2x2*2x2.
    # 2 j-groups x 4 k-groups = 8, but what about i?
    #
    # Actually for fixed i, the subproblem is:
    # C[i, :] = A[i, :] * B[:, :]
    # A[i, :] is 1x4, B is 4x8. This is a dot product per output element.
    #
    # Let me think differently. The 2x4 * 4x8 can be seen as:
    # Split A into 2 blocks of 2x2: A = [A0 | A1] (split columns)
    # Split B into 2 blocks of 4x4: B = [B0; B1] (split rows)
    # C = A0*B0 + A1*B1
    # Each A_i is 2x2, B_i is 4x4.
    # For A_i * B_i: split B_i into 2 blocks of 2x2: B_i = [B_i0 | B_i1]
    # A_i * B_i0 and A_i * B_i1 are each 2x2 * 2x2.
    # So total: 2 (i) * 2 (B_i split) * 2 (A_i has 2 rows... no)
    #
    # For fixed i: A_i (2x2) * B_i (4x4) = A_i * [B_i0 | B_i1] = [A_i*B_i0 | A_i*B_i1]
    # Each A_i*B_ij is 2x2*2x2, rank 7.
    # So for each i: 2 subproblems * rank 7 = 14
    # Total: 2 * 14 = 28? That seems too low.
    #
    # Wait, I think the issue is that these subproblems share structure. Let me just use the simple approach:
    # 8 independent 2x2*2x2 subproblems as stated in the finding.
    #
    # The 8 subproblems come from: 2 (row blocks of A) x 2 (col groups of A / row groups of B) x ... 
    # Hmm, let me just do: for each of the 8 combinations of (j_group, k_group) where j_group in {0,1}, k_group in {0,1,2,3},
    # and for each row i in {0,1}, we have a 2x2*2x2. That's 16.
    #
    # Actually re-reading: "2 j-groups x 4 k-groups" = 8, and each is a 2x2*2x2.
    # The 2x2*2x2 must be over the (i, j_rem) and (j_rem, k_rem) indices with i summed implicitly.
    # 
    # I think the correct decomposition is:
    # For each jj in {0,1} (j-group) and kk in {0,1,2,3} (k-group):
    #   The subproblem is: sum over jj_rem in {0,1} of A[i, 2*jj+jj_rem] * B[2*jj+jj_rem, 2*kk+kk_rem] for i in {0,1}, kk_rem in {0,1}
    #   This is a 2x2 * 2x2 multiplication where the "2" for A is the i dimension... no.
    #
    # Let me just implement it as 8 independent 2x2*2x2 Strassen decompositions and verify.
    # The 8 subproblems: (jj, kk) for jj in {0,1}, kk in {0,1,2,3}
    # For each (jj, kk): multiply 2x2 (indexed by i and jj_rem) by 2x2 (indexed by jj_rem and kk_rem)
    # Result is 2x2 (indexed by i and kk_rem)
    
    U_str, V_str, W_str = _strassen_2x2()
    rank = 8 * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    for jj in range(2):  # j-group
        for kk in range(4):  # k-group
            for c in range(7):  # Strassen coefficient
                col = (jj * 4 + kk) * 7 + c
                # U: rows are i*m + j = i*4 + j, where j = 2*jj + jj_rem
                # A[i, j] -> U[i*4 + j, col]
                for i in range(2):
                    for jj_rem in range(2):
                        j = 2 * jj + jj_rem
                        u_idx = i * 4 + j
                        u_val = U_str[i * 2 + jj_rem][c]
                        if u_val != 0:
                            U[u_idx][col] = u_val
                # V: rows are j*p + k = j*8 + k, where j = 2*jj + jj_rem, k = 2*kk + kk_rem
                for jj_rem in range(2):
                    for kk_rem in range(2):
                        j = 2 * jj + jj_rem
                        k = 2 * kk + kk_rem
                        v_idx = j * 8 + k
                        v_val = V_str[jj_rem * 2 + kk_rem][c]
                        if v_val != 0:
                            V[v_idx][col] = v_val
                # W: rows are k*n + i = k*2 + i, where k = 2*kk + kk_rem
                for kk_rem in range(2):
                    for i in range(2):
                        k = 2 * kk + kk_rem
                        w_idx = k * 2 + i
                        w_val = W_str[kk_rem * 2 + i][c]
                        if w_val != 0:
                            W[w_idx][col] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
