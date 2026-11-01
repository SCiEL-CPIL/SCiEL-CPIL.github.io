"""Matrix multiplication 2x4x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We use the block Strassen approach:
    # A is 2x4, B is 4x6
    # Split A into two 2x2 blocks: A = [A_left, A_right]
    # Split B into three 4x2 blocks: B = [B_top; B_mid; B_bot]
    # C = A_left * B_top + A_right * B_mid + A_right * B_bot... 
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # A has rows 0,1 and cols 0,1,2,3
    # B has rows 0,1,2,3 and cols 0,1,2,3,4,5
    # C has rows 0,1 and cols 0,1,2,3,4,5
    #
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    #
    # Group by pairs of B columns:
    # For cols (0,1): C[:,0:2] = A[:,0:2] @ B[0:2,0:2] + A[:,2:4] @ B[2:4,0:2]
    # Wait, let me think differently.
    #
    # Split the 4 columns of A / 4 rows of B into two groups of 2:
    # Group 0: rows/cols 0,1
    # Group 1: rows/cols 2,3
    #
    # C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
    # C[:, 4:6] = A[:, 0:2] @ B[0:2, 4:6] + A[:, 2:4] @ B[2:4, 4:6]
    #
    # Each term is a 2x2 @ 2x2 = 2x2 multiplication, which needs 7 Strassen terms.
    # Total: 6 terms * 7 = 42 rank.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2x2 terms:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B00)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For a 2x2 block product P = X @ Y where X is 2x2, Y is 2x2:
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    #
    # U entries (for X): 
    #   M1: x00 + x11
    #   M2: x10 + x11
    #   M3: x00
    #   M4: x11
    #   M5: x00 + x01
    #   M6: x10 - x00
    #   M7: x01 - x11
    #
    # V entries (for Y):
    #   M1: y00 + y11
    #   M2: y00
    #   M3: y10 - y00
    #   M4: y01 - y10
    #   M5: y11
    #   M6: y00 + y10
    #   M7: y10 + y11
    #
    # W entries (for P):
    #   M1: p00 + p11
    #   M2: p10 + p11
    #   M3: p01 + p11
    #   M4: p10 + p00
    #   M5: p01 + p00
    #   M6: p11 - p01
    #   M7: p00 - p11
    # Wait, let me re-derive.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So for W (output C):
    # M1 contributes: +1 to C00, +1 to C11
    # M2 contributes: +1 to C10, -1 to C11
    # M3 contributes: +1 to C01, +1 to C11
    # M4 contributes: +1 to C00, +1 to C10
    # M5 contributes: -1 to C00, +1 to C01
    # M6 contributes: +1 to C11
    # M7 contributes: +1 to C00, -1 to C11
    
    strassen_U = [
        [(0, 1), (3, 1)],   # M1: x00 + x11
        [(2, 1), (3, 1)],   # M2: x10 + x11
        [(0, 1)],           # M3: x00
        [(3, 1)],           # M4: x11
        [(0, 1), (1, 1)],   # M5: x00 + x01
        [(2, 1), (0, -1)],  # M6: x10 - x00
        [(1, 1), (3, -1)],  # M7: x01 - x11
    ]
    
    strassen_V = [
        [(0, 1), (3, 1)],   # M1: y00 + y11
        [(0, 1)],           # M2: y00
        [(2, 1), (0, -1)],  # M3: y10 - y00
        [(1, 1), (2, -1)],  # M4: y01 - y10
        [(3, 1)],           # M5: y11
        [(0, 1), (2, 1)],   # M6: y00 + y10
        [(2, 1), (3, 1)],   # M7: y10 + y11
    ]
    
    strassen_W = [
        [(0, 1), (3, 1)],   # M1: C00 + C11
        [(2, 1), (3, -1)],  # M2: C10 - C11
        [(1, 1), (3, 1)],   # M3: C01 + C11
        [(0, 1), (2, 1)],   # M4: C00 + C10
        [(0, -1), (1, 1)],  # M5: -C00 + C01
        [(3, 1)],           # M6: C11
        [(0, 1), (3, -1)],  # M7: C00 - C11
    ]
    
    col = 0
    # 6 block products: (row_group, col_group) where row_group in {0,1} for A cols, col_group in {0,1,2} for B col pairs
    # Actually: for each of the 3 B column-pairs, we have 2 A row-groups
    # Block (ag, bg): A[ag*2:(ag+1)*2, :] @ B[:, bg*2:(bg+1)*2]
    # Wait no. Let me re-index.
    
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x6: B[j, k] where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # C is 2x6: C[i, k] where i in {0,1}, k in {0,1,2,3,4,5}
    
    # Split j into two groups: jg0 = {0,1}, jg1 = {2,3}
    # Split k into three groups: kg0 = {0,1}, kg1 = {2,3}, kg2 = {4,5}
    
    # For each (jg, kg) pair, we compute A[:, jg] @ B[jg, kg] which is 2x2 @ 2x2 = 2x2
    # That's 2 * 3 = 6 block products, each with 7 Strassen terms = 42 total.
    
    for jg in range(2):  # j group: 0 or 1
        for kg in range(3):  # k group: 0, 1, or 2
            for s in range(7):  # Strassen term
                # U: A[i, j] where i in {0,1}, j in jg*2, jg*2+1
                # local index in A block: (i, j_local) where j_local in {0,1}
                # U local idx: i*2 + j_local
                for local_u, coeff in strassen_U[s]:
                    i = local_u // 2
                    j_local = local_u % 2
                    j_global = jg * 2 + j_local
                    global_u = i * m + j_global
                    U[global_u][col] += coeff
                
                # V: B[j, k] where j in jg*2, jg*2+1, k in kg*2, kg*2+1
                # local index in B block: (j_local, k_local)
                # V local idx: j_local*2 + k_local
                for local_v, coeff in strassen_V[s]:
                    j_local = local_v // 2
                    k_local = local_v % 2
                    j_global = jg * 2 + j_local
                    k_global = kg * 2 + k_local
                    global_v = j_global * p + k_global
                    V[global_v][col] += coeff
                
                # W: C[i, k] where i in {0,1}, k in kg*2, kg*2+1
                # local index in C block: (i, k_local)
                # W local idx: i*2 + k_local
                # But W is indexed as W[k*n+i, r] per the trace convention
                # So W index = k_global * n + i
                for local_w, coeff in strassen_W[s]:
                    i = local_w // 2
                    k_local = local_w % 2
                    k_global = kg * 2 + k_local
                    global_w = k_global * n + i
                    W[global_w][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
