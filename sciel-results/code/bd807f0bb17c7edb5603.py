import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _strassen_2x2():
    """Rank-7 decomposition of 2x2x2 matrix multiplication.
    
    Returns U (4x7), V (4x7), W (4x7) where:
      U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    r = 7
    U = [[0]*r for _ in range(4)]
    V = [[0]*r for _ in range(4)]
    W = [[0]*r for _ in range(4)]
    
    # p1 = (a+b)*(e+f)
    U[0][0]=1; U[1][0]=1
    V[0][0]=1; V[1][0]=1
    W[0][0]=1; W[3][0]=1  # C00 += p1, C11 += p1
    
    # p2 = (c+d)*e
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1  # C10 += p2
    
    # p3 = a*(f-h)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=-1  # C01 += p3, C11 -= p3
    
    # p4 = d*(g-e)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1  # C00 += p4, C10 += p4
    
    # p5 = (a+d)*(e+h)
    U[0][4]=1; U[3][4]=1
    V[0][4]=1; V[3][4]=1
    W[0][4]=-1; W[1][4]=1  # C00 -= p5, C01 += p5
    
    # p6 = (b-c)*(e+g)
    U[1][5]=1; U[2][5]=-1
    V[0][5]=1; V[2][5]=1
    W[3][5]=1  # C11 += p6
    
    # p7 = (a-c)*(f+h)
    U[0][6]=1; U[2][6]=-1
    V[1][6]=1; V[3][6]=1
    W[0][6]=1; W[3][6]=-1  # C00 += p7, C11 -= p7
    
    return U, V, W

def _strassen_4x4():
    """Rank-49 decomposition of 4x4x4 matrix multiplication.
    
    Uses Strassen on 2x2 blocks, composing two layers.
    
    Returns U (16x49), V (16x49), W (16x49) where:
      U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    # First, get the 2x2x2 Strassen
    U2, V2, W2 = _strassen_2x2()
    
    # 4x4x4 via block Strassen:
    # A = [[A00, A01],[A10, A11]] each 2x2
    # B = [[B00, B01],[B10, B11]] each 2x2
    # C = [[C00, C01],[C10, C11]] each 2x2
    #
    # P1 = (A00+A11)*(B00+B11)  -> 2x2 product, rank 7
    # P2 = (A10+A11)*B00        -> 2x2 product, rank 7
    # P3 = A00*(B10-B11)        -> 2x2 product, rank 7
    # P4 = A11*(B01-B10)        -> 2x2 product, rank 7
    # P5 = (A00+A01)*B11        -> 2x2 product, rank 7
    # P6 = (A10-A00)*(B00+B10)  -> 2x2 product, rank 7
    # P7 = (A01-A11)*(B01+B11)  -> 2x2 product, rank 7
    #
    # C00 = P1 - P4 + P5 + P7
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P3 + P6 + P7
    
    # Each Pk is a 2x2x2 multiplication, decomposed with Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    r = 49
    U = [[0]*r for _ in range(16)]
    V = [[0]*r for _ in range(16)]
    W = [[0]*r for _ in range(16)]
    
    # Block indices: A00 = rows 0-1, cols 0-1; A01 = rows 0-1, cols 2-3
    #                A10 = rows 2-3, cols 0-1; A11 = rows 2-3, cols 2-3
    # B00 = rows 0-1, cols 0-1; B01 = rows 0-1, cols 2-3
    # B10 = rows 2-3, cols 0-1; B11 = rows 2-3, cols 2-3
    # C00 = rows 0-1, cols 0-1; C01 = rows 0-1, cols 2-3
    # C10 = rows 2-3, cols 0-1; C11 = rows 2-3, cols 2-3
    
    # For each Pk, we need to:
    # 1. Form the A-factor (2x2) as a linear combination of blocks
    # 2. Form the B-factor (2x2) as a linear combination of blocks
    # 3. Apply Strassen 2x2x2 to get 7 rank-1 terms
    # 4. Place the result into the appropriate C-block with appropriate sign
    
    # Define block offsets
    # A-block (bi, bj) where bi=0 or 1 (row block), bj=0 or 1 (col block)
    # A-index: (bi*2 + i) * 4 + (bj*2 + j) for i,j in 0..1
    # B-block (bi, bj): B-index: (bi*2 + j) * 4 + (bj*2 + k) for j,k in 0..1
    # C-block (ci, cj): C-index: (cj*2 + k) * 4 + (ci*2 + i) for i,k in 0..1
    
    # For each of the 7 Strassen terms in the block-level decomposition:
    # We define A-linear-combination and B-linear-combination
    
    # P1: A = A00 + A11, B = B00 + B11
    # P2: A = A10 + A11, B = B00
    # P3: A = A00, B = B10 - B11
    # P4: A = A11, B = B01 - B10
    # P5: A = A00 + A01, B = B11
    # P6: A = A10 - A00, B = B00 + B10
    # P7: A = A01 - A11, B = B01 + B11
    
    # C-output blocks:
    # C00 gets: +P1, -P4, +P5, +P7
    # C01 gets: +P3, +P4
    # C10 gets: +P2, +P5
    # C11 gets: +P1, -P3, +P6, +P7
    
    # For each Pk (k=1..7), we have a Strassen 2x2x2 decomposition.
    # The 2x2 A-matrix for Pk is a linear combination of A-blocks.
    # The 2x2 B-matrix for Pk is a linear combination of B-blocks.
    # The 2x2 C-matrix for Pk goes to one or more C-blocks with signs.
    
    # Let's define for each Pk:
    #   a_comb: dict from (bi,bj) to coefficient for A-block
    #   b_comb: dict from (bi,bj) to coefficient for B-block
    #   c_targets: list of (ci, cj, sign) for C-blocks
    
    P_defs = [
        # (a_comb, b_comb, c_targets)
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, [(0,0,1), (1,1,1)]),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, [(1,0,1)]),
        # P3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, [(0,1,1)]),
        # P4
        ({(1,1):1}, {(0,1):1, (1,0):-1}, [(0,0,-1), (0,1,1)]),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, [(0,0,1), (1,0,1)]),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, [(1,1,1)]),
        # P7
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, [(0,0,1), (1,1,-1)]),
    ]
    
    for pk in range(7):
        a_comb, b_comb, c_targets = P_defs[pk]
        
        # For each of the 7 Strassen 2x2x2 terms:
        for s in range(7):
            col = pk * 7 + s
            
            # U: for each (i,j) in 2x2, U[i,j] * a_comb contribution
            # U2[i*2+j, s] is the coefficient for A[i,j] in Strassen term s
            for i in range(2):
                for j in range(2):
                    u2_val = U2[i*2+j][s]
                    if u2_val == 0:
                        continue
                    # This A[i,j] element gets contributions from all A-blocks
                    for (bi, bj), coeff in a_comb.items():
                        # A-index: (bi*2+i)*4 + (bj*2+j)
                        a_idx = (bi*2+i)*4 + (bj*2+j)
                        U[a_idx][col] += u2_val * coeff
            
            # V: for each (j,k) in 2x2, V[j,k] * b_comb contribution
            for j in range(2):
                for k in range(2):
                    v2_val = V2[j*2+k][s]
                    if v2_val == 0:
                        continue
                    for (bi, bj), coeff in b_comb.items():
                        # B-index: (bi*2+j)*4 + (bj*2+k)
                        b_idx = (bi*2+j)*4 + (bj*2+k)
                        V[b_idx][col] += v2_val * coeff
            
            # W: for each (k,i) in 2x2, W[k,i] * sign
            for k in range(2):
                for i in range(2):
                    w2_val = W2[k*2+i][s]
                    if w2_val == 0:
                        continue
                    for (ci, cj, sign) in c_targets:
                        # C-index: (cj*2+k)*4 + (ci*2+i)
                        c_idx = (cj*2+k)*4 + (ci*2+i)
                        W[c_idx][col] += w2_val * sign
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m by m x p.
    
    Returns U (n*m x n*m*p), V (m*p x n*m*p), W (p*n x n*m*p)
    """
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

def _combine(U_list, V_list, W_list, offsets):
    """Combine multiple decompositions.
    
    U_list: list of U matrices, each of shape (n*m, r_i)
    V_list: list of V matrices, each of shape (m*p, r_i)  
    W_list: list of W matrices, each of shape (p*n, r_i)
    offsets: list of (i_off, j_off, k_off) for each decomposition,
             indicating which sub-block of the full matrices they act on.
    
    Returns combined U, V, W.
    """
    n, m, p = DIMENSIONS
    total_rank = sum(U.shape[1] for U in U_list)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_off = 0
    for idx, (U_mat, V_mat, W_mat, (i_off, j_off, k_off)) in enumerate(zip(U_list, V_list, W_list, offsets)):
        r_i = U_mat.shape[1]
        
        for i in range(U_mat.shape[0]):
            for c in range(r_i):
                if U_mat[i][c] != 0:
                    U[i][col_off + c] = U_mat[i][c]
        
        for j in range(V_mat.shape[0]):
            for c in range(r_i):
                if V_mat[j][c] != 0:
                    V[j][col_off + c] = V_mat[j][c]
        
        for k in range(W_mat.shape[0]):
            for c in range(r_i):
                if W_mat[k][c] != 0:
                    W[k][col_off + c] = W_mat[k][c]
        
        col_off += r_i
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 110
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Block decomposition:
    # Split j at 4, k at 4:
    # Sub-problems (i_range, j_range, k_range):
    # 1. (0:4, 0:4, 0:4) -> 4x4x4, Strassen rank 49
    # 2. (0:4, 0:4, 4:6) -> 4x4x2, schoolbook rank 32
    # 3. (0:4, 4:5, 0:4) -> 4x1x4, schoolbook rank 16
    # 4. (0:4, 4:5, 4:6) -> 4x1x2, schoolbook rank 8
    # 5. (4:5, 0:4, 0:4) -> 1x4x4, schoolbook rank 16
    # 6. (4:5, 0:4, 4:6) -> 1x4x2, schoolbook rank 8
    # 7. (4:5, 4:5, 0:4) -> 1x1x4, schoolbook rank 4
    # 8. (4:5, 4:5, 4:6) -> 1x1x2, schoolbook rank 2
    
    # Total: 49 + 32 + 16 + 8 + 16 + 8 + 4 + 2 = 135
    # Hmm, that's worse than schoolbook 120.
    
    # Wait, I think the issue is that for rectangular blocks, schoolbook is optimal
    # or near-optimal, and the Strassen savings on 4x4x4 (64->49, saving 15) are
    # offset by the fact that we're not saving anywhere else.
    
    # Actually, schoolbook total = 4*5*6 = 120.
    # My block total = 135. That's worse!
    
    # The problem: when I split, the sub-problems don't overlap, so total rank
    # should equal sum of sub-ranks. For schoolbook, each sub-problem has rank
    # = product of its dimensions, and the sum equals 120.
    
    # 4*4*4 + 4*4*2 + 4*1*4 + 4*1*2 + 1*4*4 + 1*4*2 + 1*1*4 + 1*1*2
    # = 64 + 32 + 16 + 8 + 16 + 8 + 4 + 2 = 150
    # That's not 120! I'm double counting.
    
    # Oh wait, I see the issue. The full schoolbook has 120 terms, one for each
    # (i,j,k) triple. When I partition into blocks, each block has (Δi)(Δj)(Δk) terms,
    # and the sum should be 120.
    
    # 4*4*4 = 64
    # 4*4*2 = 32
    # 4*1*4 = 16
    # 4*1*2 = 8
    # 1*4*4 = 16
    # 1*4*2 = 8
    # 1*1*4 = 4
    # 1*1*2 = 2
    # Sum = 64+32+16+8+16+8+4+2 = 150 ≠ 120
    
    # That's wrong. Let me recount. The full space is i in 0..3 (4 values),
    # j in 0..4 (5 values), k in 0..5 (6 values). Total = 4*5*6 = 120.
    
    # My partition:
    # j in 0..3 (4 values) or j=4 (1 value)
    # k in 0..3 (4 values) or k in 4..5 (2 values)
    
    # Blocks:
    # (i:0-3, j:0-3, k:0-3): 4*4*4 = 64
    # (i:0-3, j:0-3, k:4-5): 4*4*2 = 32
    # (i:0-3, j:4, k:0-3): 4*1*4 = 16
    # (i:0-3, j:4, k:4-5): 4*1*2 = 8
    # (i:4, j:0-3, k:0-3): 1*4*4 = 16
    # (i:4, j:0-3, k:4-5): 1*4*2 = 8
    # (i:4, j:4, k:0-3): 1*1*4 = 4
    # (i:4, j:4, k:4-5): 1*1*2 = 2
    # Sum = 64+32+16+8+16+8+4+2 = 150
    
    # But 4*5*6 = 120. Where's the discrepancy?
    # Oh! i only goes 0..3, so there's no i=4. Let me redo.
    
    # i in 0..3 (4 values), j in 0..4 (5 values), k in 0..5 (6 values)
    # j: 0-3 (4) or 4 (1)
    # k: 0-3 (4) or 4-5 (2)
    
    # Blocks:
    # (i:0-3, j:0-3, k:0-3): 4*4*4 = 64
    # (i:0-3, j:0-3, k:4-5): 4*4*2 = 32
    # (i:0-3, j:4, k:0-3): 4*1*4 = 16
    # (i:0-3, j:4, k:4-5): 4*1*2 = 8
    # Sum = 64+32+16+8 = 120. Correct!
    
    # I don't need the i=4 blocks because i only goes to 3.
    
    # So the correct partition is:
    # 1. (i:0-3, j:0-3, k:0-3): 4x4x4, Strassen rank 49
    # 2. (i:0-3, j:0-3, k:4-5): 4x4x2, schoolbook rank 32
    # 3. (i:0-3, j:4, k:0-3): 4x1x4, schoolbook rank 16
    # 4. (i:0-3, j:4, k:4-5): 4x1x2, schoolbook rank 8
    # Total = 49 + 32 + 16 + 8 = 105
    
    # That's better than 120! Let me implement this.
    
    # Get Strassen 4x4x4
    U1, V1, W1 = _strassen_4x4()
    
    # Get schoolbook for other blocks
    U2, V2, W2 = _schoolbook_block(4, 4, 2)  # 4x4x2
    U3, V3, W3 = _schoolbook_block(4, 1, 4)  # 4x1x4
    U4, V4, W4 = _schoolbook_block(4, 1, 2)  # 4x1x2
    
    # Now I need to embed these into the full 4x5x6 space.
    # U is (n*m, rank) = (20, rank), indexed by i*m+j = i*5+j
    # V is (m*p, rank) = (30, rank), indexed by j*p+k = j*6+k
    # W is (p*n, rank) = (24, rank), indexed by k*n+i = k*4+i
    
    n, m, p = 4, 5, 6
    total_rank = 49 + 32 + 16 + 8
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Block 1: i in 0-3, j in 0-3, k in 0-3 (Strassen 4x4x4)
    # U1 is (16, 49), indexed by i*4+j for i,j in 0-3
    # V1 is (16, 49), indexed by j*4+k for j,k in 0-3
    # W1 is (16, 49), indexed by k*4+i for k,i in 0-3
    # Full U index: i*5+j
    # Full V index: j*6+k
    # Full W index: k*4+i
    
    for i in range(4):
        for j in range(4):
            u1_idx = i*4+j
            u_full_idx = i*5+j
            for c in range(49):
                U[u_full_idx][c] = U1[u1_idx][c]
    
    for j in range(4):
        for k in range(4):
            v1_idx = j*4+k
            v_full_idx = j*6+k
            for c in range(49):
                V[v_full_idx][c] = V1[v1_idx][c]
    
    for k in range(4):
        for i in range(4):
            w1_idx = k*4+i
            w_full_idx = k*4+i
            for c in range(49):
                W[w_full_idx][c] = W1[w1_idx][c]
    
    # Block 2: i in 0-3, j in 0-3, k in 4-5 (schoolbook 4x4x2)
    # U2 is (16, 32), indexed by i*4+j
    # V2 is (8, 32), indexed by j*2+k' where k' in 0-1 (k = 4+k')
    # W2 is (8, 32), indexed by k'*4+i
    
    col_off = 49
    for i in range(4):
        for j in range(4):
            u2_idx = i*4+j
            u_full_idx = i*5+j
            for c in range(32):
                U[u_full_idx][col_off+c] = U2[u2_idx][c]
    
    for j in range(4):
        for kp in range(2):  # k' in 0-1, k = 4+k'
            v2_idx = j*2+kp
            v_full_idx = j*6+(4+kp)
            for c in range(32):
                V[v_full_idx][col_off+c] = V2[v2_idx][c]
    
    for kp in range(2):
        for i in range(4):
            w2_idx = kp*4+i
            w_full_idx = (4+kp)*4+i
            for c in range(32):
                W[w_full_idx][col_off+c] = W2[w2_idx][c]
    
    # Block 3: i in 0-3, j=4, k in 0-3 (schoolbook 4x1x4)
    # U3 is (4, 16), indexed by i*1+0 = i
    # V3 is (4, 16), indexed by 0*4+k = k
    # W3 is (16, 16), indexed by k*4+i
    
    col_off = 49 + 32
    for i in range(4):
        u3_idx = i
        u_full_idx = i*5+4
        for c in range(16):
            U[u_full_idx][col_off+c] = U3[u3_idx][c]
    
    for k in range(4):
        v3_idx = k
        v_full_idx = 4*6+k
        for c in range(16):
            V[v_full_idx][col_off+c] = V3[v3_idx][c]
    
    for k in range(4):
        for i in range(4):
            w3_idx = k*4+i
            w_full_idx = k*4+i
            for c in range(16):
                W[w_full_idx][col_off+c] = W3[w3_idx][c]
    
    # Block 4: i in 0-3, j=4, k in 4-5 (schoolbook 4x1x2)
    # U4 is (4, 8), indexed by i
    # V4 is (2, 8), indexed by k'
    # W4 is (8, 8), indexed by k'*4+i
    
    col_off = 49 + 32 + 16
    for i in range(4):
        u4_idx = i
        u_full_idx = i*5+4
        for c in range(8):
            U[u_full_idx][col_off+c] = U4[u4_idx][c]
    
    for kp in range(2):
        v4_idx = kp
        v_full_idx = 4*6+(4+kp)
        for c in range(8):
            V[v_full_idx][col_off+c] = V4[v4_idx][c]
    
    for kp in range(2):
        for i in range(4):
            w4_idx = kp*4+i
            w_full_idx = (4+kp)*4+i
            for c in range(8):
                W[w_full_idx][col_off+c] = W4[w4_idx][c]
    
    # Verify correctness
    if time.time() - start_time < deadline - 10:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should give C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # should equal 1 if we're decomposing the multiplication tensor.
        
        # Actually, the decomposition satisfies:
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i,j,c]*A[i,j]) * (sum_k V[j,k,c]*B[j,k]) * W[k,i,c]
        # Wait, that's not quite right. Let me think more carefully.
        
        # The tensor T_{ijk} for matrix multiplication is defined as T_{ijk} = 1 for all i,j,k.
        # The decomposition is: T_{ijk} = sum_c U_{ijc} * V_{jkc} * W_{kic}
        
        # Let's verify this.
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(total_rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    expected = 1
                    err = abs(val - expected)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-10:
            # Fallback to schoolbook
            r = n*m*p
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
