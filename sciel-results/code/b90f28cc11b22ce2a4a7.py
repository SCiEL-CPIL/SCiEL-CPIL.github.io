"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # safety margin

    n, m, p = DIMENSIONS  # 4, 5, 5
    # We decompose M_{4,5,5} = M_{4,4,4} + M_{4,4,1} + M_{4,1,5}
    # M_{4,4,4}: i in 0..3, j in 0..3, k in 0..3
    # M_{4,4,1}: i in 0..3, j in 0..3, k=4
    # M_{4,1,5}: i in 0..3, j=4, k in 0..4

    # Each part has its own Strassen decomposition
    # Total rank = 49 + 16 + 20 = 85

    rank_total = 85
    U = [[0] * rank_total for _ in range(n * m)]  # 20 x 85
    V = [[0] * rank_total for _ in range(m * p)]  # 25 x 85
    W = [[0] * rank_total for _ in range(p * n)]  # 20 x 85

    # Strassen 2x2 decomposition:
    # For matrices A (2x2), B (2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # The 7 Strassen factors for 2x2:
    # Factor s has:
    #   U_s: 2x1 (coefficients for A)
    #   V_s: 2x1 (coefficients for B)
    #   W_s: 2x2 (coefficients for C, but really it's a 4x1 since C is 2x2)
    # Actually, let's think of it differently.
    # For 2x2 multiplication, the tensor has:
    #   U: shape (4, 7) - coefficients for A elements
    #   V: shape (4, 7) - coefficients for B elements  
    #   W: shape (4, 7) - coefficients for C elements
    # Where row index in U/V/W is the matrix element index.

    # Strassen 2x2 decomposition:
    # A elements indexed 0..3: A00=0, A01=1, A10=2, A11=3
    # B elements indexed 0..3: B00=0, B01=1, B10=2, B11=3
    # C elements indexed 0..3: C00=0, C01=1, C10=2, C11=3

    # Strassen factors (U_col, V_col, W_col) where col is the 7th factor:
    # s=0 (M1): A: [1,0,0,1], B: [1,0,0,1], C: [1,0,0,-1]  wait, C = M1*... 
    # Let me be more careful.
    # M1 = (A00+A11)(B00+B11)
    #   U contribution: A00 gets 1, A11 gets 1 => U_col[0]=1, U_col[3]=1
    #   V contribution: B00 gets 1, B11 gets 1 => V_col[0]=1, V_col[3]=1
    #   C00 += M1, so W_col[0] += 1
    #   C11 -= M1, so W_col[3] -= 1

    # Let me define the Strassen 2x2 decomposition properly.
    # For factor s, we have:
    #   u_s: vector in R^4 (coeffs for A)
    #   v_s: vector in R^4 (coeffs for B)
    #   w_s: vector in R^4 (coeffs for C)
    # The product is: sum_s (u_s . A) * (v_s . B) * w_s (outer product with C)

    # Strassen 2x2:
    # s=0: M1 = (A00+A11)(B00+B11)
    #   u = [1,0,0,1], v = [1,0,0,1]
    #   C00 += M1 => w[0]=1; C11 -= M1 => w[3]=-1
    #   w = [1, 0, 0, -1]

    # s=1: M2 = (A10+A11)B00
    #   u = [0,0,1,1], v = [1,0,0,0]
    #   C10 += M2 => w[2]=1; C11 -= M2 => w[3]=-1
    #   w = [0, 0, 1, -1]

    # s=2: M3 = A00(B01-B11)
    #   u = [1,0,0,0], v = [0,1,0,-1]
    #   C01 += M3 => w[1]=1; C11 += M3 => w[3]=1
    #   w = [0, 1, 0, 1]

    # s=3: M4 = A11(B10-B00)
    #   u = [0,0,0,1], v = [0,0,1,-1]
    #   C00 += M4 => w[0]=1; C10 += M4 => w[2]=1
    #   w = [1, 0, 1, 0]

    # s=4: M5 = (A00+A01)B11
    #   u = [1,1,0,0], v = [0,0,0,1]
    #   C00 -= M5 => w[0]=-1; C01 += M5 => w[1]=1
    #   w = [-1, 1, 0, 0]

    # s=5: M6 = (A10-A00)(B00+B01)
    #   u = [-1,0,1,0], v = [1,1,0,0]
    #   C11 += M6 => w[3]=1
    #   w = [0, 0, 0, 1]

    # s=6: M7 = (A01-A11)(B10+B11)
    #   u = [0,1,0,-1], v = [0,0,1,1]
    #   C00 += M7 => w[0]=1
    #   w = [1, 0, 0, 0]

    strassen_U_2x2 = [
        [1, 0, 0, 1],   # s=0
        [0, 0, 1, 1],   # s=1
        [1, 0, 0, 0],   # s=2
        [0, 0, 0, 1],   # s=3
        [1, 1, 0, 0],   # s=4
        [-1, 0, 1, 0],  # s=5
        [0, 1, 0, -1],  # s=6
    ]
    strassen_V_2x2 = [
        [1, 0, 0, 1],   # s=0
        [1, 0, 0, 0],   # s=1
        [0, 1, 0, -1],  # s=2
        [0, 0, 1, -1],  # s=3
        [0, 0, 0, 1],   # s=4
        [1, 1, 0, 0],   # s=5
        [0, 0, 1, 1],   # s=6
    ]
    strassen_W_2x2 = [
        [1, 0, 0, -1],  # s=0
        [0, 0, 1, -1],  # s=1
        [0, 1, 0, 1],   # s=2
        [1, 0, 1, 0],   # s=3
        [-1, 1, 0, 0],  # s=4
        [0, 0, 0, 1],   # s=5
        [1, 0, 0, 0],   # s=6
    ]

    # For 4x4 multiplication via Strassen:
    # We treat A, B as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2

    # Each "block multiplication" M_s = X_s * Y_s where X_s, Y_s are linear combos of A-blocks and B-blocks.
    # Each block multiplication itself is a 2x2 multiplication (rank 7).
    # So for each Strassen step s (7 steps), we have 7 sub-steps t (for the 2x2 block mult).
    # Total: 49 factors.

    # For the 4x4 case:
    # A element (i,j) where i,j in 0..3.
    # Block index: (i//2, j//2) = (bi, bj) where bi,bj in 0..1
    # Within-block: (i%2, j%2) = (wi, wj) where wi,wj in 0..1
    # So A[i,j] is in block A[bi,bj] at position [wi,wj].

    # For Strassen step s:
    # X_s is a linear combination of A-blocks.
    # Y_s is a linear combination of B-blocks.
    # The coefficients come from strassen_U_2x2[s] and strassen_V_2x2[s].
    # strassen_U_2x2[s] = [u00, u01, u10, u11] meaning:
    #   X_s = u00*A00 + u01*A01 + u10*A10 + u11*A11
    # Similarly for Y_s.

    # Then M_s = X_s * Y_s (2x2 block multiplication).
    # Using Strassen for the block multiplication:
    # For sub-step t:
    #   M_s = sum_t (x_t . X_s) * (y_t . Y_s) * w_t (where x_t, y_t, w_t from 2x2 Strassen)
    #   where x_t . X_s means: x_t[0]*X_s[0,0] + x_t[1]*X_s[0,1] + x_t[2]*X_s[1,0] + x_t[3]*X_s[1,1]
    #   and X_s[0,0] = u00*A[0,0] + u01*A[0,2] + u10*A[2,0] + u11*A[2,2] (since A00 block is A[0:2,0:2])

    # Let me define the 4x4 Strassen decomposition.
    # For factor (s,t) where s in 0..6 (outer Strassen step), t in 0..6 (inner Strassen step):
    # U_{s,t}[i*4+j] = sum over block positions (bi,bj) of:
    #   strassen_U_2x2[s][bi*2+bj] * strassen_U_2x2[t][wi*2+wj]
    #   where bi = i//2, bj = j//2, wi = i%2, wj = j%2
    # Wait, that's not quite right. Let me re-derive.

    # X_s[wi,wj] = sum_{bi,bj} strassen_U_2x2[s][bi*2+bj] * A[bi*2+wi, bj*2+wj]
    # So for a given A element A[i,j] where i=bi*2+wi, j=bj*2+wj:
    #   X_s[wi,wj] contains strassen_U_2x2[s][bi*2+bj] * A[i,j]
    #   So the coefficient of A[i,j] in X_s[wi,wj] is strassen_U_2x2[s][bi*2+bj]

    # Similarly, M_s = sum_t (x_t . X_s) * (y_t . Y_s) * w_t
    # where x_t . X_s = sum_{wi,wj} strassen_U_2x2[t][wi*2+wj] * X_s[wi,wj]
    #   = sum_{wi,wj} strassen_U_2x2[t][wi*2+wj] * sum_{bi,bj} strassen_U_2x2[s][bi*2+bj] * A[bi*2+wi, bj*2+wj]
    #   = sum_{bi,bj,wi,wj} strassen_U_2x2[t][wi*2+wj] * strassen_U_2x2[s][bi*2+bj] * A[bi*2+wi, bj*2+wj]

    # So the coefficient of A[i,j] in the inner product (x_t . X_s) is:
    #   strassen_U_2x2[t][(i%2)*2+(j%2)] * strassen_U_2x2[s][(i//2)*2+(j//2)]
    # where i=bi*2+wi, j=bj*2+wj.

    # Similarly for B.
    # And for C, the output C[ci,cj] gets:
    #   sum_{s,t} w_t[C_block_position] * (coefficient from A) * (coefficient from B)
    # where C_block_position corresponds to which C-block the element belongs to.

    # Let me just build it directly.
    # For 4x4, A has 16 elements indexed 0..15 (row-major: i*4+j).
    # For factor (s,t):
    #   U[i*4+j] = strassen_U_2x2[t][(i%2)*2+(j%2)] * strassen_U_2x2[s][(i//2)*2+(j//2)]
    #   V[i*4+j] = strassen_V_2x2[t][(i%2)*2+(j%2)] * strassen_V_2x2[s][(i//2)*2+(j//2)]
    #   W[k*4+l] = strassen_W_2x2[t][(k%2)*2+(l%2)] * strassen_W_2x2[s][(k//2)*2+(l//2)]

    # This gives a valid rank-49 decomposition of M_{4,4,4}.

    # Now for M_{4,4,1}: i in 0..3, j in 0..3, k=4.
    # This is a "slice" of the multiplication tensor.
    # We can decompose it using the same 4x4 Strassen for the first two indices,
    # but since k=4 (only one value), the V matrix only needs to handle j in 0..3, k=4.
    # Actually, M_{4,4,1} is: A (4x4) * B (4x1) = C (4x1).
    # This is just matrix-vector multiplication, which has rank 16 (schoolbook).
    # But we can also use a Strassen-like approach? Actually, for M_{n,m,1}, the rank is n*m.
    # So M_{4,4,1} has rank 16.

    # For M_{4,1,5}: i in 0..3, j=4, k in 0..4.
    # This is: A (4x1) * B (1x5) = C (4x5).
    # Rank is 4*5 = 20.

    # Let's build the full decomposition.

    def build_strassen_4x4(U, V, W, offset):
        """Build Strassen 4x4 decomposition, adding to U,V,W at column offset."""
        for s in range(7):
            for t in range(7):
                col = offset + s * 7 + t
                for i in range(4):
                    for j in range(4):
                        a_idx = i * 4 + j
                        bi, wi = i // 2, i % 2
                        bj, wj = j // 2, j % 2
                        U[a_idx][col] = strassen_U_2x2[t][wi * 2 + wj] * strassen_U_2x2[s][bi * 2 + bj]
                for i in range(4):
                    for j in range(4):
                        b_idx = i * 4 + j
                        bi, wi = i // 2, i % 2
                        bj, wj = j // 2, j % 2
                        V[b_idx][col] = strassen_V_2x2[t][wi * 2 + wj] * strassen_V_2x2[s][bi * 2 + bj]
                for i in range(4):
                    for j in range(4):
                        c_idx = i * 4 + j
                        bi, wi = i // 2, i % 2
                        bj, wj = j // 2, j % 2
                        W[c_idx][col] = strassen_W_2x2[t][wi * 2 + wj] * strassen_W_2x2[s][bi * 2 + bj]

    def build_schoolbook_slice(U, V, W, offset, n, m, p, i_range, j_range, k_val=None):
        """Build schoolbook decomposition for a slice."""
        col = offset
        for i in i_range:
            for j in j_range:
                if k_val is not None:
                    k_list = [k_val]
                else:
                    k_list = range(p)
                for k in k_list:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col - offset

    # Part 1: M_{4,4,4} - Strassen rank 49
    # i in 0..3, j in 0..3, k in 0..3
    # In the full 4x5x5 tensor:
    # U row: i*5+j where i in 0..3, j in 0..3
    # V row: j*5+k where j in 0..3, k in 0..3
    # W row: k*4+i where k in 0..3, i in 0..3

    # We need to map the 4x4 Strassen indices to the full tensor indices.
    # A[i,j] in the 4x4 subproblem corresponds to U row i*5+j in the full problem.
    # B[j,k] in the 4x4 subproblem corresponds to V row j*5+k in the full problem.
    # C[k,i] in the 4x4 subproblem corresponds to W row k*4+i in the full problem.

    # So for the 4x4 Strassen, we use:
    # U_full[i*5+j] = U_4x4[i*4+j]
    # V_full[j*5+k] = V_4x4[j*4+k]
    # W_full[k*4+i] = W_4x4[k*4+i]

    # Let's build the 4x4 Strassen into temporary arrays, then map.
    U_44 = [[0] * 49 for _ in range(16)]
    V_44 = [[0] * 49 for _ in range(16)]
    W_44 = [[0] * 49 for _ in range(16)]
    build_strassen_4x4(U_44, V_44, W_44, 0)

    # Map to full tensor
    for i in range(4):
        for j in range(4):
            for col in range(49):
                U[i * 5 + j][col] = U_44[i * 4 + j][col]
    for j in range(4):
        for k in range(4):
            for col in range(49):
                V[j * 5 + k][col] = V_44[j * 4 + k][col]
    for k in range(4):
        for i in range(4):
            for col in range(49):
                W[k * 4 + i][col] = W_44[k * 4 + i][col]

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Part 2: M_{4,4,1} - schoolbook rank 16
    # i in 0..3, j in 0..3, k=4
    # U row: i*5+j
    # V row: j*5+4
    # W row: 4*4+i = 16+i
    offset2 = 49
    col = offset2
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[16 + i][col] = 1
            col += 1

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Part 3: M_{4,1,5} - schoolbook rank 20
    # i in 0..3, j=4, k in 0..4
    # U row: i*5+4
    # V row: 4*5+k = 20+k
    # W row: k*4+i
    offset3 = 65
    col = offset3
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4][col] = 1
            V[20 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
