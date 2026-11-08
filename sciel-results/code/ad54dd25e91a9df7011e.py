"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140
    n, m, p = 2, 4, 6
    
    # Strassen rank-7 decomposition for (2x2)*(2x2)
    # 7 products: U (4x7), V (4x7), W (4x7)
    # U indexed by (i,j) -> row i*2+j for A[i,j]
    # V indexed by (j,k) -> row j*2+k for B[j,k]
    # W indexed by (k,i) -> row k*2+i for C[k,i] (trace convention W[k*n+i])
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = A00*(B01-B11)
    U_s[0*2+0, 0] = 1.0
    V_s[0*2+1, 0] = 1.0
    V_s[1*2+1, 0] = -1.0
    W_s[0*2+0, 0] = 1.0  # C00 += M1
    W_s[1*2+0, 0] = 1.0  # C01 += M1
    
    # M2 = (A10-A00)*B00
    U_s[1*2+0, 1] = 1.0
    U_s[0*2+0, 1] = -1.0
    V_s[0*2+0, 1] = 1.0
    W_s[0*2+0, 1] = -1.0  # C00 -= M2
    W_s[1*2+1, 1] = 1.0   # C11 += M2... wait let me recheck
    
    # Let me redo this carefully.
    # Strassen:
    # M1 = A00*(B01-B11)
    # M2 = (A10-A00)*B00
    # M3 = (A00+A11)*B11
    # M4 = (A11-A01)*(B00+B10)
    # M5 = (A01+A11)*(B10-B00)
    # M6 = (A00-A10)*(B01+B11)
    # M7 = (A10+A01)*(B11-B10)
    #
    # C00 = M1+M3-M5+M7
    # C01 = M1+M4
    # C10 = M2+M5
    # C11 = M3+M4-M2+M6
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = A00*(B01-B11)
    U_s[0, 0] = 1.0       # A[0,0]
    V_s[1, 0] = 1.0       # B[0,1]
    V_s[3, 0] = -1.0      # B[1,1]
    W_s[0, 0] = 1.0       # C[0,0] += M1
    W_s[2, 0] = 1.0       # C[0,1] += M1  (k=0,i=1 -> row 0*2+1=1? No, k*n+i = 0*2+1=1)
    
    # W indexed by k*n+i where n=2, so row = k*2+i
    # C00: k=0,i=0 -> row 0
    # C01: k=0,i=1 -> row 1
    # C10: k=1,i=0 -> row 2
    # C11: k=1,i=1 -> row 3
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = A00*(B01-B11)
    U_s[0, 0] = 1.0
    V_s[1, 0] = 1.0
    V_s[3, 0] = -1.0
    W_s[0, 0] = 1.0   # C00
    W_s[1, 0] = 1.0   # C01
    
    # M2 = (A10-A00)*B00
    U_s[2, 1] = 1.0    # A[1,0]
    U_s[0, 1] = -1.0   # A[0,0]
    V_s[0, 1] = 1.0    # B[0,0]
    W_s[2, 1] = 1.0    # C10 += M2
    W_s[3, 1] = -1.0   # C11 -= M2
    
    # M3 = (A00+A11)*B11
    U_s[0, 2] = 1.0    # A[0,0]
    U_s[3, 2] = 1.0    # A[1,1]
    V_s[3, 2] = 1.0    # B[1,1]
    W_s[0, 2] = 1.0    # C00 += M3
    W_s[3, 2] = 1.0    # C11 += M3
    
    # M4 = (A11-A01)*(B00+B10)
    U_s[3, 3] = 1.0    # A[1,1]
    U_s[1, 3] = -1.0   # A[0,1]
    V_s[0, 3] = 1.0    # B[0,0]
    V_s[2, 3] = 1.0    # B[1,0]
    W_s[1, 3] = 1.0    # C01 += M4
    W_s[3, 3] = 1.0    # C11 += M4
    
    # M5 = (A01+A11)*(B10-B00)
    U_s[1, 4] = 1.0    # A[0,1]
    U_s[3, 4] = 1.0    # A[1,1]
    V_s[2, 4] = 1.0    # B[1,0]
    V_s[0, 4] = -1.0   # B[0,0]
    W_s[0, 4] = -1.0   # C00 -= M5
    W_s[2, 4] = 1.0    # C10 += M5
    
    # M6 = (A00-A10)*(B01+B11)
    U_s[0, 5] = 1.0    # A[0,0]
    U_s[2, 5] = -1.0   # A[1,0]
    V_s[1, 5] = 1.0    # B[0,1]
    V_s[3, 5] = 1.0    # B[1,1]
    W_s[3, 5] = 1.0    # C11 += M6
    
    # M7 = (A10+A01)*(B11-B10)
    U_s[2, 6] = 1.0    # A[1,0]
    U_s[1, 6] = 1.0    # A[0,1]
    V_s[3, 6] = 1.0    # B[1,1]
    V_s[2, 6] = -1.0   # B[1,0]
    W_s[0, 6] = 1.0    # C00 += M7
    
    # Verify Strassen decomposition for 2x2x2
    # Build tensor T[i,j,k] = sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c]
    T_check = np.zeros((2,2,2))
    for c in range(7):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T_check[i,j,k] += U_s[i*2+j, c] * V_s[j*2+k, c] * W_s[k*2+i, c]
    
    # T[i,j,k] should be 1 for all i,j,k (it's the multiplication tensor)
    # Actually no - the tensor for matrix mult is T[i,j,k]=1 if the (i,j) input to A and (j,k) input to B produce output at (i,k)
    # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) since A[i,j]*B[j,k] contributes to C[i,k]
    # Wait, that's not right either. Let me think again.
    
    # The matrix multiplication tensor T is defined by T[i,j,k] = 1 where the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]
    # So T[i,j,k] = 1 for all i,j,k in the 2x2x2 case.
    
    # Let's verify: sum_c U[i*2+j,c] V[j*2+k,c] W[k*2+i,c] should give 1 for all i,j,k
    # But that's not what we want. We want the decomposition to represent the multiplication map.
    
    # Actually, the tensor decomposition of matrix multiplication means:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k]) W[k,i,c]... 
    
    # No wait. The standard tensor decomposition for matmul (n,m,p) is:
    # T in R^{nm} x R^{mp} x R^{pn} where T[i*m+j, j*p+k, k*n+i] = 1
    # And we want T = sum_c U_c ⊗ V_c ⊗ W_c where U_c in R^{nm}, V_c in R^{mp}, W_c in R^{pn}
    
    # For 2x2x2: T[i*2+j, j*2+k, k*2+i] = 1 for all i,j,k in {0,1}
    # Decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where (U_c ⊗ V_c ⊗ W_c)[i*2+j, j*2+k, k*2+i] = U_c[i*2+j] * V_c[j*2+k] * W_c[k*2+i]
    
    # So we need: sum_c U_c[i*2+j] * V_c[j*2+k] * W_c[k*2+i] = 1 for all i,j,k
    
    # Let me verify my Strassen factors:
    T_check2 = np.zeros((4,4,4))
    for c in range(7):
        for a in range(4):
            for b in range(4):
                for d in range(4):
                    # a = i*2+j, b = j*2+k, d = k*2+i
                    # Need consistency: a's j must equal b's j, b's k must equal d's k, d's i must equal a's i
                    i_a, j_a = a // 2, a % 2
                    j_b, k_b = b // 2, b % 2
                    k_d, i_d = d // 2, d % 2
                    if j_a == j_b and k_b == k_d and i_d == i_a:
                        T_check2[a,b,d] += U_s[a,c] * V_s[b,c] * W_s[d,c]
    
    # T_check2[a,b,d] should be 1 when indices are consistent, 0 otherwise
    # But actually the tensor is only nonzero at consistent indices, and equals 1 there
    all_ones = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i*2+j
                b = j*2+k
                d = k*2+i
                if abs(T_check2[a,b,d] - 1.0) > 1e-10:
                    all_ones = False
                    print(f"Mismatch at i={i},j={j},k={k}: got {T_check2[a,b,d]}")
    
    if all_ones:
        print("Strassen 2x2x2 verified!")
    
    # Now build the full 2x4x6 decomposition
    # n=2, m=4, p=6
    # A is 2x4, B is 4x6, C is 2x6
    # Split: A = [A1 A2] (each 2x2), B = [B1; B2] (each 2x6)
    # C = A1*B1 + A2*B2
    # Further split B1 = [B1a B1b B1c] (each 2x2), B2 = [B2a B2b B2c] (each 2x2)
    # C = A1*B1a + A1*B1b + A1*B1c + A2*B2a + A2*B2b + A2*B2c
    # Each term is (2x2)*(2x2) using Strassen rank 7
    # Total rank = 6*7 = 42
    
    total_rank = 42
    U_full = np.zeros((n*m, total_rank))  # 8 x 42
    V_full = np.zeros((m*p, total_rank))  # 24 x 42
    W_full = np.zeros((p*n, total_rank))  # 12 x 42
    
    # 6 subproblems: (a_block, b_block) where a_block in {0,1}, b_block in {0,1,2}
    # a_block 0: A columns 0-1, a_block 1: A columns 2-3
    # b_block 0: B rows 0-1, cols 0-1; b_block 1: B rows 0-1, cols 2-3; b_block 2: B rows 0-1, cols 4-5
    # Wait, B is 4x6. Split into top 2x6 and bottom 2x6.
    # Top: rows 0-1, bottom: rows 2-3
    # Then split each 2x6 into three 2x2 blocks by columns
    
    for a_blk in range(2):  # 0 or 1
        for b_blk in range(3):  # 0, 1, or 2
            # This subproblem: A[a_blk*2:(a_blk+1)*2, :] (2x2) * B[top/bottom, b_blk*2:(b_blk+1)*2] (2x2)
            # But wait, we need A1*B1 where A1 is 2x2 (first 2 cols of A) and B1 is 2x6 (first 2 rows of B)
            # Then B1 is split into 3 blocks of 2x2 by columns
            
            # A sub-block: rows 0-1, cols a_blk*2 to a_blk*2+1
            # B sub-block: rows 0-1 (if a_blk=0) or 2-3 (if a_blk=1)? 
            # No! A1*B1: A1 is first 2 columns of A (2x2), B1 is first 2 rows of B (2x6)
            # A2*B2: A2 is last 2 columns of A (2x2), B2 is last 2 rows of B (2x6)
            
            # So for a_blk=0: A cols 0-1, B rows 0-1
            # For a_blk=1: A cols 2-3, B rows 2-3
            
            b_row_offset = a_blk * 2  # B row offset: 0 for a_blk=0, 2 for a_blk=1
            a_col_offset = a_blk * 2  # A col offset: 0 for a_blk=0, 2 for a_blk=1
            b_col_offset = b_blk * 2  # B col offset within the 2x6 block: 0, 2, or 4
            
            # Strassen product index offset
            s_off = (a_blk * 3 + b_blk) * 7
            
            for c in range(7):
                col = s_off + c
                # U: A[i, a_col_offset+j] for i in 0,1; j in 0,1
                for i in range(2):
                    for j in range(2):
                        U_full[i*4 + a_col_offset + j, col] = U_s[i*2+j, c]
                
                # V: B[b_row_offset+j, b_col_offset+k] for j in 0,1; k in 0,1
                for j in range(2):
                    for k in range(2):
                        V_full[(b_row_offset + j)*6 + b_col_offset + k, col] = V_s[j*2+k, c]
                
                # W: C[i, b_col_offset+k] -> indexed as k*n+i where k is the column index in C
                # C is 2x6, so C[i, col_c] where col_c = b_col_offset + k
                # W index: k*n+i where k is the output column index
                # But wait, the W factor is indexed by (output_col, i) -> row = output_col*n + i
                # In the 2x2 subproblem, output is 2x2 with indices (i_out, k_out) where i_out in {0,1}, k_out in {0,1}
                # The global output column is b_col_offset + k_out
                # So W row = (b_col_offset + k_out)*n + i_out = (b_col_offset + k_out)*2 + i_out
                for i_out in range(2):
                    for k_out in range(2):
                        W_full[(b_col_offset + k_out)*n + i_out, col] = W_s[k_out*n + i_out, c]
    
    # Verify the full decomposition
    # Build the full tensor T[i*4+j, j*6+k, k*2+i] = 1
    T_target = np.zeros((8, 24, 12))
    for i in range(2):
        for j in range(4):
            for k in range(6):
                T_target[i*4+j, j*6+k, k*2+i] = 1.0
    
    # Compute from factors
    T_comp = np.zeros((8, 24, 12))
    for col in range(total_rank):
        u_col = U_full[:, col]
        v_col = V_full[:, col]
        w_col = W_full[:, col]
        # Outer product
        for a in range(8):
            if u_col[a] != 0:
                for b in range(24):
                    if v_col[b] != 0:
                        val = u_col[a] * v_col[b]
                        for d in range(12):
                            if w_col[d] != 0:
                                T_comp[a, b, d] += val * w_col[d]
    
    # Check
    max_err = np.max(np.abs(T_comp - T_target))
    print(f"Max error in full decomposition: {max_err}")
    
    if max_err < 1e-10:
        # Convert to lists
        U_out = [[float(x) for x in row] for row in U_full]
        V_out = [[float(x) for x in row] for row in V_full]
        W_out = [[float(x) for x in row] for row in W_full]
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
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
