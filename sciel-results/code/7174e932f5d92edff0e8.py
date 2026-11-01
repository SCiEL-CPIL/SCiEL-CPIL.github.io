import time

n, m, p = 2, 3, 5
RANK = 28
START = time.time()
DEADLINE = 140.0

U = [[0] * RANK for _ in range(n * m)]
V = [[0] * RANK for _ in range(m * p)]
W = [[0] * RANK for _ in range(p * n)]
col = 0


def place(u, v, w):
    """Add a rank-1 column: increment col and fill U[u][col], V[v][col], W[w][col]."""
    global col
    c = col
    U[u][c] += 1
    V[v][c] += 1
    W[w][c] += 1
    col += 1


def place2(u, v, w1, w2):
    """Add a rank-1 column with W split into w1 (coef 1) and w2 (coef -1)."""
    global col
    c = col
    U[u][c] = 1
    V[v][c] = 1
    W[w1][c] = 1
    W[w2][c] = -1
    col += 1


def strassen_block(k0):
    """
    Strassen-7 for 2x2x2 block on (i=0,1; j=1,2; k=k0, k0+1).
    Local coords: i' = i, j' = j-1, k' = k - k0.
    Products:
      P1 = (A00+A11)(B00+B11) -> C00, C11
      P2 = (A10+A11)B00      -> C10, -C11
      P3 = A00(B01-B11)      -> -C01, C11
      P4 = A11(B10-B00)      -> C00, C10
      P5 = (A00+A01)B11      -> -C00, C01
      P6 = (A10-A00)(B00+B01)-> C11
      P7 = (A01-A11)(B10+B11)-> C00
    """
    global col
    base_u0 = 0 * m      # 0
    base_u1 = 1 * m      # 3
    j1 = 1
    j2 = 2
    k1 = k0
    k2 = k0 + 1
    # Row indices in U
    U00 = base_u0 + j1   # A[0,1]
    U01 = base_u0 + j2   # A[0,2]
    U10 = base_u1 + j1   # A[1,1]
    U11 = base_u1 + j2   # A[1,2]

    # V indices: V[j*p + k]
    V00 = j1 * p + k1
    V01 = j1 * p + k2
    V10 = j2 * p + k1
    V11 = j2 * p + k2

    # W indices: W[k*n + i]
    W00 = k1 * n + 0     # C[0, k1]
    W01 = k2 * n + 0     # C[0, k2]
    W10 = k1 * n + 1     # C[1, k1]
    W11 = k2 * n + 1     # C[1, k2]

    # P1: (A00+A11)(B00+B11)
    # But we must use the same column for both products... no, each is a separate column.
    # Wait, each P is a single rank-1 tensor. The coefficients in U, V, W encode the linear forms.
    # P1: U has entries at (0,j1) and (1,j2). V has entries at (j1,k1) and (j2,k2).
    # W has entries at (k1, 0) and (k2, 1).
    # However, the decomposition is not just placing a 1 at the U index. The U vector is a linear form in A.
    # For P1, the U linear form is A[0][1] + A[1][2]. But in our column representation,
    # each column r gives a rank-1 tensor u^{(r)} v^{(r)} w^{(r)}.
    # The coefficient for A[0][1] in column r is U[0*m+1][r].
    # So for P1, we need TWO columns? No.
    # A single column r represents the product (sum_i U_i,r A_i) * (sum_j V_j,r B_j) * (sum_k W_k,r C_k).
    # If U has non-zero entries at multiple indices, that means the "u" vector is a linear combination.
    # So one column can have U[0*m+1][r] = 1 and U[1*m+2][r] = 1.
    # This represents the product of the linear form (A01 + A12) with other linear forms.
    
    # So let's re-evaluate. One column per Strassen product.
    
    # P1: U: A01, A12. V: B(j1,k1), B(j2,k2). W: C(k1,0), C(k2,1)
    # Wait, if U has 2 entries and V has 2 entries, this is not a "rank-1 tensor" in the sense of the decomposition sum.
    # The decomposition is T = sum_{r} u_r \otimes v_r \otimes w_r where u_r, v_r, w_r are vectors.
    # If u_r = e_{01} + e_{12} (in the A basis), v_r = e_{j1k1} + e_{j2k2}, w_r = e_{k10} + e_{k21}.
    # Then the term is (A01+A12)(Bj1k1+Bj2k2)(Ck10+Ck21). This expands to 4 products, not 1.
    # This is incorrect. The decomposition T = sum u_r v_r w_r means T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # If U_{i,r} is non-zero for two i's, then for a fixed r, the term contributes to multiple (i,j,k).
    # But it's a SINGLE rank-1 tensor in the tensor space.
    # Yes! The tensor T is in A^* \otimes B^* \otimes C^*.
    # A rank-1 tensor is u \otimes v \otimes w where u \in A^*, v \in B^*, w \in C^*.
    # The matrix U has rows indexed by A-basis, columns by r. So U[:, r] is the vector u_r.
    # If U[:, r] = [0, 1, 0, ..., 0, 0, ..., 0, 1, 0, ...] (1 at positions corresponding to A01 and A12),
    # then u_r = A01^* + A12^*. This is a valid linear form.
    # So ONE column can indeed represent P1.
    
    # Let's implement it correctly.
    
    # Clear the previous attempts in this function? No, we call it fresh.
    # But we must ensure we don't re-add. We are building fresh in construct().
    
    # Actually, my initial approach in the previous code was adding 1s to U, V, W entries.
    # For P1:
    c = col
    
    # U: A01 (index 0*m+1=1), A12 (index 1*m+2=5)
    U[1][c] = 1
    U[5][c] = 1
    # V: B(j1,k1) (index 1*5+0=5), B(j2,k2) (index 2*5+1=11)
    V[5][c] = 1
    V[11][c] = 1
    # W: C(k1,0) (index 0*2+0=0), C(k2,1) (index 1*2+1=3)
    W[0][c] = 1
    W[3][c] = 1
    col += 1
    
    # P2: (A10+A11) B00
    # A10 = A[1][1] index 1*m+1=4, A11 = A[1][2] index 5
    # B00 = B[j1,k1] index 5
    # W: C10 (index k1*n+1 = 1), -C11 (index k2*n+1 = 3)
    c = col
    U[4][c] = 1
    U[5][c] = 1
    V[5][c] = 1
    W[1][c] = 1
    W[3][c] = -1
    col += 1
    
    # P3: A00 (B01-B11)
    # A00 = A[0][1] index 1
    # B01 = B[j1,k2] index 1*5+1=6, B11 = B[j2,k2] index 2*5+1=11
    # W: -C01 (index k2*n+0 = 2), C11 (index k2*n+1 = 3)
    c = col
    U[1][c] = 1
    V[6][c] = 1
    V[11][c] = -1
    W[2][c] = -1
    W[3][c] = 1
    col += 1
    
    # P4: A11 (B10-B00)
    # A11 = A[1][2] index 5
    # B10 = B[j2,k1] index 2*5+0=10, B00 = B[j1,k1] index 5
    # W: C00 (index k1*n+0 = 0), C10 (index k1*n+1 = 1)
    c = col
    U[5][c] = 1
    V[10][c] = 1
    V[5][c] = -1
    W[0][c] = 1
    W[1][c] = 1
    col += 1
    
    # P5: (A00+A01) B11
    # A00 = A[0][1] index 1, A01 = A[0][2] index 2
    # B11 = B[j2,k2] index 11
    # W: -C00 (index 0), C01 (index 2)
    c = col
    U[1][c] = 1
    U[2][c] = 1
    V[11][c] = 1
    W[0][c] = -1
    W[2][c] = 1
    col += 1
    
    # P6: (A10-A00)(B00+B01)
    # A10 = A[1][1] index 4, A00 = A[0][1] index 1
    # B00 = B[j1,k1] index 5, B01 = B[j1,k2] index 6
    # W: C11 (index 3)
    c = col
    U[4][c] = 1
    U[1][c] = -1
    V[5][c] = 1
    V[6][c] = 1
    W[3][c] = 1
    col += 1
    
    # P7: (A01-A11)(B10+B11)
    # A01 = A[0][2] index 2, A11 = A[1][2] index 5
    # B10 = B[j2,k1] index 10, B11 = B[j2,k2] index 11
    # W: C00 (index 0)
    c = col
    U[2][c] = 1
    U[5][c] = -1
    V[10][c] = 1
    V[11][c] = 1
    W[0][c] = 1
    col += 1


def construct():
    global U, V, W, col
    # Block 1: j=0. C_k = A_{i0} B_{0k}.
    for i in range(n):
        for k in range(p):
            c = col
            U[i * m + 0][c] = 1
            V[0 * p + k][c] = 1
            W[k * n + i][c] = 1
            col += 1
    
    # Block 2a: j=1,2; k=0,1 (Strassen)
    strassen_block(0)
    
    # Block 2b: j=1,2; k=2 (Schoolbook 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            c = col
            U[i * m + j][c] = 1
            V[j * p + 2][c] = 1
            W[2 * n + i][c] = 1
            col += 1
    
    # Block 2c: j=1,2; k=3,4 (Strassen)
    strassen_block(3)
    
    if col != RANK:
        return None
    
    return {"U": U, "V": V, "W": W}
