"""Matrix multiplication 4x5x6 via rank-101 block decomposition."""
import time
import random

DIMENSIONS = [4, 5, 6]

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_DEADLINE = 140.0

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W as lists of lists (4x7, 4x7, 4x7) for 2x2*2x2.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: p1 = (a+d)(e+h)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[2][1] += 1
    
    # c=1: p2 = (c+d)e
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[0][1] += 1; W[2][1] -= 1
    
    # c=2: p3 = a(f-h)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][0] += 1; W[2][1] += 1
    
    # c=3: p4 = d(g-e)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][0] += 1; W[0][1] += 1
    
    # c=4: p5 = (a+b)h
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][0] -= 1; W[1][0] += 1
    
    # c=5: p6 = (c-a)(e+f)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[2][1] += 1
    
    # c=6: p7 = (b-d)(g+h)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][0] += 1
    
    return U, V, W

def _build_4x4x4():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Strassen on 2x2 blocks of 2x2 matrices.
    Local indices:
    U: 16 rows (4x4), V: 16 rows (4x4), W: 16 rows (4x4)
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    su, sv, sw = _strassen_2x2()
    
    # Strassen formulas for 2x2 block matrices:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map the local 2x2x2 Strassen terms to the 4x4x4 structure.
    # The 4x4x4 problem is composed of 16 2x2x2 sub-problems? No.
    # It is composed of 7 "Strassen products" P1..P7, each of which is a 2x2x2 matrix multiplication.
    # Each Pi is a 2x2 matrix.
    # So we invoke _strassen_2x2() 7 times? No, the structure is fixed.
    
    # Let's define the linear combinations for A and B blocks for each Pi.
    # A blocks: A00 (top-left 2x2), A01 (top-right 2x2), A10 (bot-left 2x2), A11 (bot-right 2x2)
    # B blocks: B00, B01, B10, B11
    
    # For each Pi, we have coefficients for A_blocks and B_blocks.
    # Then we multiply the resulting 2x2 matrices using the 2x2x2 Strassen decomposition.
    
    # Actually, it's easier to think of the 4x4x4 tensor as a sum of 49 rank-1 terms.
    # Each term corresponds to a specific combination of indices.
    
    # Let's iterate over the 7 Strassen products for the outer 2x2 block structure.
    # For each product index m in 0..6:
    #   We need to compute Pi = A_lin * B_lin, where A_lin and B_lin are 2x2 matrices
    #   formed by linear combinations of the 2x2 blocks of A and B.
    #   Then Pi is a 2x2 matrix.
    #   The contribution of Pi to the result C is given by the outer Strassen formulas.
    #   But we need the tensor decomposition.
    
    # Alternative: Just hardcode the 49 terms?
    # Or construct it programmatically.
    
    # Let's use the property that rank(4,4,4) = 49.
    # We can construct it by taking the Kronecker product of the 2x2x2 Strassen decomposition with itself?
    # No, Strassen is not a Kronecker product in that simple way.
    
    # Let's build it by iterating over the 7 outer products and 7 inner products.
    
    # Outer products m=0..6:
    # A_coeffs: list of 4 (coeffs for A00, A01, A10, A11)
    # B_coeffs: list of 4 (coeffs for B00, B01, B10, B11)
    # C_coeffs: list of 4 (coeffs for C00, C01, C10, C11)
    
    outer_products = [
        # m, A_coeffs, B_coeffs, C_coeffs
        (0, [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1: (A00+A11)(B00+B11) -> C00, C11
        (1, [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # P2: (A10+A11)B00 -> C10, C11
        (2, [1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # P3: A00(B01-B11) -> C01, C11
        (3, [0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # P4: A11(B10-B00) -> C00, C10
        (4, [1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # P5: (A00+A01)B11 -> C00, C01
        (5, [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6: (A10-A00)(B00+B01) -> C11
        (6, [0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # P7: (A01-A11)(B10+B11) -> C00
    ]
    
    # Wait, the C_coeffs above are for the *block* C.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So for each outer product m, we have a specific Pi.
    # Pi is a 2x2 matrix.
    # We need to decompose the multiplication A_lin * B_lin = Pi using the 2x2x2 Strassen decomposition.
    # Then we combine the results according to the C_coeffs.
    
    # This is getting complicated to map to global indices directly.
    # Let's just generate the local 4x4x4 U, V, W.
    
    # Local U: 16 rows. Row index = i*4 + j.
    # Local V: 16 rows. Row index = j*4 + k.
    # Local W: 16 rows. Row index = k*4 + i.
    
    # We will fill U, V, W with the 49 terms.
    
    # For each outer product m (0..6):
    #   We have A_lin = sum_a A_coeffs[a] * A_block_a
    #   B_lin = sum_b B_coeffs[b] * B_block_b
    #   We need to compute A_lin * B_lin.
    #   This is a 2x2x2 multiplication.
    #   We use the 7 terms of _strassen_2x2().
    #   For each inner term c (0..6):
    #     We have local U, V, W vectors for the 2x2x2 problem.
    #     These vectors are defined on the 2x2 blocks.
    #     We need to expand them to the 4x4 structure.
    
    # This is doable but tedious.
    # Given the time, I will rely on the fact that the rank is 49 and construct a valid decomposition
    # by combining the 7 outer Strassen steps, each using the 7 inner Strassen steps.
    
    # I will implement a helper to expand a 2x2x2 decomposition into a 4x4x4 context.
    
    pass # Placeholder, will implement in full below

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will build the rank 101 decomposition.
    # Total rank 101.
    rank = 101
    U = [[0] * rank for _ in range(n * m)] # 20 x 101
    V = [[0] * rank for _ in range(m * p)] # 30 x 101
    W = [[0] * rank for _ in range(p * n)] # 24 x 101
    
    col_offset = 0
    
    # Helper to add a sub-decomposition
    def add_sub(U_sub, V_sub, W_sub, u_map, v_map, w_map):
        nonlocal col_offset
        sub_rank = len(U_sub[0])
        for c in range(sub_rank):
            for u_row, val in u_map:
                if val != 0:
                    U[u_row][col_offset + c] += val
            for v_row, val in v_map:
                if val != 0:
                    V[v_row][col_offset + c] += val
            for w_row, val in w_map:
                if val != 0:
                    W[w_row][col_offset + c] += val
        col_offset += sub_rank
        return sub_rank
    
    # 1. A1 B11 (4x4x4) -> Rank 49
    # A1: A[i, 0:4], B11: B[0:4, 0:4], C11: C[i, 0:4]
    # Local indices:
    # u_local(i, j) -> global U[i*5 + j]
    # v_local(j, k) -> global V[j*6 + k]
    # w_local(k, i) -> global W[k*4 + i]
    
    # We need the 4x4x4 decomposition.
    # I will construct it on the fly.
    U_444, V_444, W_444 = _build_4x4x4()
    
    u_map_444 = []
    for i in range(4):
        for j in range(4):
            u_map_444.append((i*5 + j, 1)) # Coeff 1, will be multiplied by U_444 values
    # This mapping is not quite right. U_444 has values.
    # I should pass the matrices and maps.
    
    # Let's refactor add_sub to take matrices and index maps.
    
    # Actually, simpler: just iterate.
    
    # Sub 1: 4x4x4
    su, sv, sw = _build_4x4x4()
    for c in range(49):
        for i in range(4):
            for j in range(4):
                val = su[i*4+j][c]
                if val != 0:
                    U[i*5+j][col_offset+c] = val
        for j in range(4):
            for k in range(4):
                val = sv[j*4+k][c]
                if val != 0:
                    V[j*6+k][col_offset+c] = val
        for k in range(4):
            for i in range(4):
                val = sw[k*4+i][c]
                if val != 0:
                    W[k*4+i][col_offset+c] = val
    col_offset += 49
    
    # Sub 2: A1 B12 (4x4x2) -> Rank 28
    # A1: A[i, 0:4], B12: B[0:4, 4:6], C12: C[i, 4:6]
    # Split A1 into A_top (2x4) and A_bot (2x4).
    # A_top * B12 is 2x2. Rank 7.
    # A_bot * B12 is 2x2. Rank 7.
    # Wait, B12 is 4x2.
    # A_top (2x4) * B12 (4x2) -> 2x2.
    # This is a 2x4x2 problem.
    # Split A_top into A_top0 (2x2) and A_top1 (2x2).
    # B12 into B120 (2x2) and B121 (2x2).
    # C_top = A_top0 B120 + A_top1 B121.
    # Each is 2x2x2, rank 7.
    # So C_top rank 14.
    # Similarly C_bot rank 14.
    # Total 28.
    
    su2, sv2, sw2 = _strassen_2x2()
    
    # A_top0: A[0:2, 0:2], B120: B[0:2, 4:6], C_top0: C[0:2, 4:6]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                val = su2[i*2+j][c]
                if val != 0:
                    U[i*5+j][col_offset+c] = val
        for j in range(2):
            for k in range(2):
                val = sv2[j*2+k][c]
                if val != 0:
                    V[j*6+(4+k)][col_offset+c] = val
        for k in range(2):
            for i in range(2):
                val = sw2[k*2+i][c]
                if val != 0:
                    W[(4+k)*4+i][col_offset+c] = val
    col_offset += 7
    
    # A_top1: A[0:2, 2:4], B121: B[2:4, 4:6], C_top1: C[0:2, 4:6]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                val = su2[i*2+j][c]
                if val != 0:
                    U[i*5+(2+j)][col_offset+c] = val
        for j in range(2):
            for k in range(2):
                val = sv2[j*2+k][c]
                if val != 0:
                    V[(2+j)*6+(4+k)][col_offset+c] = val
        for k in range(2):
            for i in range(2):
                val = sw2[k*2+i][c]
                if val != 0:
                    W[(4+k)*4+i][col_offset+c] = val
    col_offset += 7
    
    # A_bot0: A[2:4, 0:2], B120: B[0:2, 4:6], C_bot0: C[2:4, 4:6]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                val = su2[i*2+j][c]
                if val != 0:
                    U[(2+i)*5+j][col_offset+c] = val
        for j in range(2):
            for k in range(2):
                val = sv2[j*2+k][c]
                if val != 0:
                    V[j*6+(4+k)][col_offset+c] = val
        for k in range(2):
            for i in range(2):
                val = sw2[k*2+i][c]
                if val != 0:
                    W[(4+k)*4+(2+i)][col_offset+c] = val
    col_offset += 7
    
    # A_bot1: A[2:4, 2:4], B121: B[2:4, 4:6], C_bot1: C[2:4, 4:6]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                val = su2[i*2+j][c]
                if val != 0:
                    U[(2+i)*5+(2+j)][col_offset+c] = val
        for j in range(2):
            for k in range(2):
                val = sv2[j*2+k][c]
                if val != 0:
                    V[(2+j)*6+(4+k)][col_offset+c] = val
        for k in range(2):
            for i in range(2):
                val = sw2[k*2+i][c]
                if val != 0:
                    W[(4+k)*4+(2+i)][col_offset+c] = val
    col_offset += 7
    
    # Sub 3: A2 B21 (4x1x4) -> Rank 16
    # A2: A[i, 4], B21: B[4, 0:4], C21: C[i, 0:4]
    # C[i, k] = A[i, 4] B[4, k]
    for i in range(4):
        for k in range(4):
            U[i*5+4][col_offset] = 1
            V[4*6+k][col_offset] = 1
            W[k*4+i][col_offset] = 1
            col_offset += 1
    
    # Sub 4: A2 B22 (4x1x2) -> Rank 8
    # A2: A[i, 4], B22: B[4, 4:6], C22: C[i, 4:6]
    for i in range(4):
        for k in range(2):
            U[i*5+4][col_offset] = 1
            V[4*6+(4+k)][col_offset] = 1
            W[(4+k)*4+i][col_offset] = 1
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}

# Implement _build_4x4x4 correctly
def _build_4x4x4():
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    su, sv, sw = _strassen_2x2()
    
    outer_products = [
        # m, A_coeffs (A00, A01, A10, A11), B_coeffs (B00, B01, B10, B11), C_coeffs (C00, C01, C10, C11)
        (0, [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        (1, [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        (2, [1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        (3, [0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        (4, [1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        (5, [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        (6, [0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    for m, a_c, b_c, c_c in outer_products:
        # For this outer product, we compute Pi = A_lin * B_lin
        # A_lin is 2x2, B_lin is 2x2.
        # We use the 7 terms of Strassen for 2x2x2.
        for c_idx in range(7):
            # U for A_lin:
            # A_lin[0,0] = a_c[0]*A00[0,0] + a_c[1]*A01[0,0] + ...
            # Actually, A_lin is formed from the blocks.
            # The 2x2x2 decomposition su, sv, sw acts on the entries of A_lin, B_lin.
            # So we need to map su[i*2+j] to the global U.
            
            # su[i*2+j] is the coefficient for A_lin[i,j].
            # A_lin[i,j] = sum_{a} a_c[a] * A_block_a[i,j]
            # So U_global[global_idx(i,j)] += su[i*2+j] * a_c[a]
            
            # Global index for A_block_a[i,j]:
            # Block 0 (A00): i, j
            # Block 1 (A01): i, 2+j
            # Block 2 (A10): 2+i, j
            # Block 3 (A11): 2+i, 2+j
            
            for i in range(2):
                for j in range(2):
                    coeff = su[i*2+j][c_idx]
                    if coeff == 0: continue
                    for a_idx in range(4):
                        if a_c[a_idx] != 0:
                            val = coeff * a_c[a_idx]
                            if a_idx == 0:
                                u_row = i*4 + j
                            elif a_idx == 1:
                                u_row = i*4 + (2+j)
                            elif a_idx == 2:
                                u_row = (2+i)*4 + j
                            else:
                                u_row = (2+i)*4 + (2+j)
                            U[u_row][col] += val
                            
            # V for B_lin:
            for j in range(2):
                for k in range(2):
                    coeff = sv[j*2+k][c_idx]
                    if coeff == 0: continue
                    for b_idx in range(4):
                        if b_c[b_idx] != 0:
                            val = coeff * b_c[b_idx]
                            if b_idx == 0:
                                v_row = j*4 + k
                            elif b_idx == 1:
                                v_row = j*4 + (2+k)
                            elif b_idx == 2:
                                v_row = (2+j)*4 + k
                            else:
                                v_row = (2+j)*4 + (2+k)
                            V[v_row][col] += val
                            
            # W for C:
            # C_block_c receives contribution from Pi.
            # C_block_c[i,j] += c_c[c_idx] * Pi[i,j]
            # Pi[i,j] is computed by the 2x2x2 decomposition.
            # sw[k*2+i] is the coefficient for Pi[k,i]? No, W[k*2+i] in 2x2x2 is for C[k,i].
            # So C[i,j] gets contribution from sw[j*2+i] * c_c[c_idx].
            
            for i in range(2):
                for k in range(2): # k is col index of C, i is row index
                    coeff = sw[k*2+i][c_idx]
                    if coeff == 0: continue
                    for c_idx_block in range(4):
                        if c_c[c_idx_block] != 0:
                            val = coeff * c_c[c_idx_block]
                            if c_idx_block == 0:
                                w_row = k*4 + i
                            elif c_idx_block == 1:
                                w_row = k*4 + (2+i)
                            elif c_idx_block == 2:
                                w_row = (2+k)*4 + i
                            else:
                                w_row = (2+k)*4 + (2+i)
                            W[w_row][col] += val
                            
            col += 1
            
    return U, V, W

