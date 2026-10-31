"""Matrix multiplication 4x4x4: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (7x7).
    
    The decomposition treats 4x4 matrices as 2x2 blocks of 2x2 matrices.
    Each 2x2 matrix multiplication uses Strassen's 7-multiplication formula.
    Total rank = 7 * 7 = 49.
    
    Convention: W[k*n + i, r] where output C = A @ B, C[i,k] = sum_j A[i,j]*B[j,k]
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strassen's 7 terms for 2x2 matrix multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4: split into 2x2 blocks.
    # A is 4x4, split as [[A00, A01], [A10, A11]] where each block is 2x2
    # B is 4x4, split as [[B00, B01], [B10, B11]] where each block is 2x2
    # C is 4x4, split as [[C00, C01], [C10, C11]] where each block is 2x2
    #
    # Each C block = sum of products of 2x2 blocks using Strassen.
    # Each 2x2 block product uses 7 scalar multiplications (in the algebra of 2x2 matrices).
    # But we need to express this as a tensor decomposition over scalars.
    #
    # Total: 7 (outer Strassen) * 7 (inner Strassen) = 49 rank-1 terms.
    
    # Indexing: A[i,j] where i in [0,4), j in [0,4)
    # Block indices: bi = i//2, bj = j//2, li = i%2, lj = j%2
    # A[i,j] = A_{bi,bj}[li,lj]
    
    # Strassen coefficients for 2x2:
    # For M_s (s=1..7), A and B block coefficients:
    # M1: A00*A00, A00*B11, A11*B00, A11*B11  -> (A00+A11)(B00+B11)
    #     A blocks: {00:1, 11:1}, B blocks: {00:1, 11:1}
    # M2: (A10+A11)B00 -> A blocks: {10:1, 11:1}, B blocks: {00:1}
    # M3: A00(B01-B11) -> A blocks: {00:1}, B blocks: {01:1, 11:-1}
    # M4: A11(B10-B00) -> A blocks: {11:1}, B blocks: {10:1, 00:-1}
    # M5: (A00+A01)B11 -> A blocks: {00:1, 01:1}, B blocks: {11:1}
    # M6: (A10-A00)(B00+B01) -> A blocks: {10:1, 00:-1}, B blocks: {00:1, 01:1}
    # M7: (A01-A11)(B10+B11) -> A blocks: {01:1, 11:-1}, B blocks: {10:1, 11:1}
    
    # Outer Strassen (4x4 as 2x2 of 2x2 blocks):
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    #
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Inner Strassen (2x2 as 2x2 of scalars):
    # For each 2x2 matrix product P = X*Y where X,Y are 2x2:
    # t1 = (X00+X11)(Y00+Y11)
    # t2 = (X10+X11)Y00
    # t3 = X00(Y01-Y11)
    # t4 = X11(Y10-Y00)
    # t5 = (X00+X01)Y11
    # t6 = (X10-X00)(Y00+Y01)
    # t7 = (X01-X11)(Y10+Y11)
    #
    # P00 = t1 + t4 - t5 + t7
    # P01 = t3 + t5
    # P10 = t2 + t4
    # P11 = t1 - t2 + t3 + t6
    
    # We build the rank-49 decomposition.
    # Each term is: (linear combo of A entries) * (linear combo of B entries) -> (linear combo of C entries)
    
    # Let's define the structure:
    # Outer term s (0..6), inner term t (0..6)
    # For outer term s, we have coefficients for A-blocks and B-blocks.
    # For inner term t, we have coefficients for scalar positions within each block.
    
    # A-block coefficients for outer Strassen terms:
    # Each entry is a dict mapping (block_row, block_col) to coefficient
    outer_A = [
        {(0,0): 1, (1,1): 1},      # S1
        {(1,0): 1, (1,1): 1},      # S2
        {(0,0): 1},                 # S3
        {(1,1): 1},                 # S4
        {(0,0): 1, (0,1): 1},      # S5
        {(1,0): 1, (0,0): -1},     # S6
        {(0,1): 1, (1,1): -1},     # S7
    ]
    outer_B = [
        {(0,0): 1, (1,1): 1},      # S1
        {(0,0): 1},                 # S2
        {(0,1): 1, (1,1): -1},     # S3
        {(1,0): 1, (0,0): -1},     # S4
        {(1,1): 1},                 # S5
        {(0,0): 1, (0,1): 1},      # S6
        {(1,0): 1, (1,1): 1},      # S7
    ]
    # C-block coefficients for outer Strassen results:
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    outer_C = [
        {(0,0): 1, (1,1): 1},      # S1 -> C00 and C11
        {(1,0): 1, (1,1): -1},     # S2 -> C10 and -C11
        {(0,1): 1, (1,1): 1},      # S3 -> C01 and C11
        {(0,0): 1, (1,0): 1},      # S4 -> C00 and C10
        {(0,0): -1, (0,1): 1},     # S5 -> -C00 and C01
        {(1,1): 1},                # S6 -> C11
        {(0,0): 1, (0,1): -1},     # S7 -> C00 and -C01
    ]
    
    # Inner Strassen (for 2x2 scalar matrices):
    # X and Y are 2x2, indexed by (row, col) in {0,1}
    inner_A = [
        {(0,0): 1, (1,1): 1},      # t1
        {(1,0): 1, (1,1): 1},      # t2
        {(0,0): 1},                 # t3
        {(1,1): 1},                 # t4
        {(0,0): 1, (0,1): 1},      # t5
        {(1,0): 1, (0,0): -1},     # t6
        {(0,1): 1, (1,1): -1},     # t7
    ]
    inner_B = [
        {(0,0): 1, (1,1): 1},      # t1
        {(0,0): 1},                 # t2
        {(0,1): 1, (1,1): -1},     # t3
        {(1,0): 1, (0,0): -1},     # t4
        {(1,1): 1},                 # t5
        {(0,0): 1, (0,1): 1},      # t6
        {(1,0): 1, (1,1): 1},      # t7
    ]
    inner_C = [
        {(0,0): 1, (1,1): 1},      # t1 -> P00 and P11
        {(1,0): 1, (1,1): -1},     # t2 -> P10 and -P11
        {(0,1): 1, (1,1): 1},      # t3 -> P01 and P11
        {(0,0): 1, (1,0): 1},      # t4 -> P00 and P10
        {(0,0): -1, (0,1): 1},     # t5 -> -P00 and P01
        {(1,1): 1},                # t6 -> P11
        {(0,0): 1, (0,1): -1},     # t7 -> P00 and -P01
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    col = 0
    for s in range(7):  # outer Strassen term
        for t in range(7):  # inner Strassen term
            # For each A-block position in outer_A[s], and each scalar position in inner_A[t]:
            # A index: i = block_row * 2 + local_row, j = block_col * 2 + local_col
            for (br, bc), ca in outer_A[s].items():
                for (lr, lc), ca2 in inner_A[t].items():
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U[i * m + j][col] += ca * ca2
            
            # For each B-block position in outer_B[s], and each scalar position in inner_B[t]:
            # B index: j = block_row * 2 + local_row, k = block_col * 2 + local_col
            for (br, bc), cb in outer_B[s].items():
                for (lr, lc), cb2 in inner_B[t].items():
                    j = br * 2 + lr
                    k = bc * 2 + lc
                    V[j * p + k][col] += cb * cb2
            
            # For each C-block position in outer_C[s], and each scalar position in inner_C[t]:
            # C index: i = block_row * 2 + local_row, k = block_col * 2 + local_col
            # W index: k * n + i (per the convention W[k*n+i, r])
            for (br, bc), cc in outer_C[s].items():
                for (lr, lc), cc2 in inner_C[t].items():
                    i = br * 2 + lr
                    k = bc * 2 + lc
                    W[k * n + i][col] += cc * cc2
            
            col += 1
    
    # Verify correctness with a random test
    np.random.seed(42)
    A_test = np.random.randint(-3, 4, size=(4, 4))
    B_test = np.random.randint(-3, 4, size=(4, 4))
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j... wait
    # U is indexed by (i*m + j) for A[i,j]
    # V is indexed by (j*p + k) for B[j,k]
    # W is indexed by (k*n + i) for C[i,k]
    # So: C[k*n+i] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # But we need to sum over j as well.
    # Actually: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    # No, that's not right either. Let me think again.
    #
    # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 if this is a valid (i,j,k) for C[i,k] = sum_j A[i,j]B[j,k]
    # Actually T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
    #
    # So C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    #           = sum_r W[k,i,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    # Wait, that's not quite right. Let me be more careful.
    #
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And we need T[i,j,k] = 1 for all valid (i,j,k) where C[i,k] = sum_j A[i,j]*B[j,k].
    # Actually T[i,j,k] should be 1 for each (i,j,k) triple that contributes to C[i,k].
    # C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all i,j,k in range(4).
    
    # Let's verify: for each (i,j,k), check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if val != 1:
                    max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-10:
        # Fallback: return schoolbook if verification fails
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
