"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_block(n, m, p):
    """
    Returns U, V, W for multiplying an n x m matrix A by an m x p matrix B
    to get an n x p matrix C, with W indexed as W[k*n + i, r].
    """
    if n == 1 and m == 1 and p == 1:
        rank = 1
        U = [[1.0]]
        V = [[1.0]]
        W = [[1.0]]
        return U, V, W

    if n == 1 and m == 2 and p == 1:
        rank = 2
        U = [[1.0, 0.0],
             [0.0, 1.0]]
        V = [[1.0, 0.0],
             [0.0, 1.0]]
        W = [[1.0, 0.0],
             [0.0, 1.0]]
        return U, V, W

    if n == 2 and m == 1 and p == 1:
        rank = 1
        U = [[1.0],
             [1.0]]
        V = [[1.0]]
        W = [[1.0]]
        return U, V, W

    if n == 1 and m == 1 and p == 2:
        rank = 1
        U = [[1.0]]
        V = [[1.0]]
        W = [[1.0],
             [1.0]]
        return U, V, W

    if n == 2 and m == 2 and p == 2:
        rank = 7
        # Strassen decomposition
        # U maps A (row-major) to 7 intermediates
        # V maps B (row-major) to 7 intermediates
        # W maps intermediates to C (column-major per contract: W[k*n+i, r])
        
        U = [[0.0]*7 for _ in range(4)]
        V = [[0.0]*7 for _ in range(4)]
        W = [[0.0]*7 for _ in range(4)]

        # Indices for U: A is 2x2. A[i*m+j].
        # Strassen formulas:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # U construction:
        # U[0] (A00): M1=1, M3=1, M5=1, M6=-1 -> [1, 0, 1, 0, 1, -1, 0]
        # U[1] (A01): M5=1, M7=1 -> [0, 0, 0, 0, 1, 0, 1]
        # U[2] (A10): M2=1, M6=1 -> [0, 1, 0, 0, 0, 1, 0]
        # U[3] (A11): M1=1, M2=1, M4=1, M7=-1 -> [1, 1, 0, 1, 0, 0, -1]
        
        U[0] = [1, 0, 1, 0, 1, -1, 0]
        U[1] = [0, 0, 0, 0, 1, 0, 1]
        U[2] = [0, 1, 0, 0, 0, 1, 0]
        U[3] = [1, 1, 0, 1, 0, 0, -1]

        # V construction:
        # V[0] (B00): M1=1, M2=1, M4=-1, M6=1 -> [1, 1, 0, -1, 0, 1, 0]
        # V[1] (B01): M3=-1, M6=1, M7=1 -> [0, 0, -1, 0, 0, 1, 1]
        # V[2] (B10): M4=1, M7=1 -> [0, 0, 0, 1, 0, 0, 1]
        # V[3] (B11): M1=1, M3=1, M5=1, M7=1 -> [1, 0, 1, 0, 1, 0, 1]
        
        V[0] = [1, 1, 0, -1, 0, 1, 0]
        V[1] = [0, 0, -1, 0, 0, 1, 1]
        V[2] = [0, 0, 0, 1, 0, 0, 1]
        V[3] = [1, 0, 1, 0, 1, 0, 1]

        # W construction:
        # C is 2x2. C[i, k]. Index k*n + i.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # W row for C00 (k=0, i=0 -> idx 0): [1, 0, 0, 1, -1, 0, -1]
        # W row for C01 (k=1, i=0 -> idx 1): [0, 0, 1, 0, 1, 0, 0]
        # W row for C10 (k=0, i=1 -> idx 2): [0, 1, 0, 1, 0, 0, 0]
        # W row for C11 (k=1, i=1 -> idx 3): [1, -1, 1, 0, 0, 0, 1]
        
        W[0] = [1, 0, 0, 1, -1, 0, -1]
        W[1] = [0, 0, 1, 0, 1, 0, 0]
        W[2] = [0, 1, 0, 1, 0, 0, 0]
        W[3] = [1, -1, 1, 0, 0, 0, 1]

        return U, V, W

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _combine_blocks(blocks):
    """
    Combine a list of (U, V, W) blocks into a single decomposition.
    Blocks are independent (direct sum).
    """
    if not blocks:
        return None, None, None
    
    first_U, first_V, first_W = blocks[0]
    
    if len(blocks) == 1:
        return first_U, first_V, first_W

    # Determine dimensions from the first block
    rows_U = len(first_U)
    rows_V = len(first_V)
    rows_W = len(first_W)
    
    rank_total = 0
    for U_b, V_b, W_b in blocks:
        rank_total += len(U_b[0])

    U = [[0.0] * rank_total for _ in range(rows_U)]
    V = [[0.0] * rank_total for _ in range(rows_V)]
    W = [[0.0] * rank_total for _ in range(rows_W)]

    offset = 0
    for U_b, V_b, W_b in blocks:
        r = len(U_b[0])
        for i in range(rows_U):
            for k in range(r):
                U[i][offset + k] = U_b[i][k]
        for i in range(rows_V):
            for k in range(r):
                V[i][offset + k] = V_b[i][k]
        for i in range(rows_W):
            for k in range(r):
                W[i][offset + k] = W_b[i][k]
        offset += r

    return U, V, W

def _decompose_2x3x3():
    """
    Decompose 2x3x3 into blocks:
    A (2x3) * B (3x3) = C (2x3)
    
    Split B into B_left (3x2) and B_right (3x1).
    C = A * B_left + A * B_right
    
    Block 1: A (2x3) * B_left (3x2) -> C_left (2x2)
    Block 2: A (2x3) * B_right (3x1) -> C_right (2x1)
    
    However, A is shared. We can't just sum ranks if A is shared in a non-trivial way?
    Actually, the tensor is a direct sum if the supports are disjoint.
    Here, the input A is the same, but the output columns are distinct.
    The multiplication tensor T_{i,j,k} = 1 if j is the shared index.
    T = T1 + T2 where T1 has support on k in {0,1} and T2 on k=2.
    Since the supports on k are disjoint, the tensor rank is sum of ranks.
    
    So:
    Block 1: 2 x 3 x 2
    Block 2: 2 x 3 x 1
    
    Now decompose 2 x 3 x 2:
    Split B (3x2) into B_top (2x2) and B_bottom (1x2).
    A (2x3) split into A_left (2x2) and A_right (2x1).
    
    C1 = A * B = A_left * B_top + A_right * B_bottom ? No.
    A is 2x3, B is 3x2.
    A = [A_left (2x2), A_right (2x1)]
    B = [B_top (2x2); B_bottom (1x2)]
    
    C1 = A_left * B_top + A_right * B_bottom.
    These terms involve disjoint parts of A and B?
    A_left uses A[0:2, 0:2], A_right uses A[0:2, 2:3]. Disjoint.
    B_top uses B[0:2, 0:2], B_bottom uses B[2:3, 0:2]. Disjoint.
    So yes, direct sum.
    
    Sub-blocks for 2x3x2:
    1. 2 x 2 x 2 (Rank 7)
    2. 2 x 1 x 2 (Rank 2) -> A_right (2x1) * B_bottom (1x2) -> C (2x2)
       Wait, A_right is 2x1, B_bottom is 1x2. Product is 2x2.
       Rank of 2x1x2?
       C_{i,k} = A_{i,0} * B_{0,k}. Outer product of vector A and vector B^T.
       Rank 1?
       Let's check.
       C = u * v^T. Tensor is u_i v_k.
       This is rank 1 tensor?
       T_{i, 0, k} = u_i v_k.
       Yes, rank 1.
       Wait, my previous thought said Rank 2 for 2x1x2. Let me re-verify.
       C_{ik} = A_i B_k.
       This is an outer product. Rank 1.
       So Sub-block 2 is Rank 1.
       
    So 2x3x2 Rank = 7 + 1 = 8.
    
    Now 2 x 3 x 1:
    A (2x3) * B (3x1) -> C (2x1).
    C_i = sum_j A_{ij} B_j. Dot product.
    Split B into B_top (2x1) and B_bottom (1x1).
    A into A_left (2x2) and A_right (2x1).
    
    C = A_left * B_top + A_right * B_bottom.
    Sub-block 3: 2 x 2 x 1.
    C_i = sum_{j=0}^1 A_{ij} B_j.
    This is a 2x2 matrix multiplication by a vector?
    Rank of 2x2x1?
    It's two dot products of length 2.
    C0 = A00 B0 + A01 B1
    C1 = A10 B0 + A11 B1
    Can we do it in rank 2?
    T_{i,j,k} with k=0.
    T_{0,0,0}=1, T_{0,1,0}=1, T_{1,0,0}=1, T_{1,1,0}=1.
    Rank 2?
    Let's try to represent.
    U (2x2), V (2x1), W (1x2).
    W is 1x2. Let W[0] = [1, 1] (since k=0 only, W index is k*n+i = 0*2+i = i).
    Wait, W shape is (p*n, rank) = (1*2, rank) = (2, rank).
    U shape (n*m, rank) = (4, rank).
    V shape (m*p, rank) = (2, rank).
    
    C_i = sum_r U_{i, r} V_{r} W_{i} ? No.
    C_{i,k} = sum_{j,r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    Here p=1, so j*p+k = j. k=0, so W index is i.
    C_i = sum_r W_i * (sum_j U_{i*m+j, r} V_{j, r}).
    
    Let P_{i,r} = sum_j U_{i*m+j, r} V_{j, r}.
    Then C_i = W_i * P_{i,r} ... sum over r.
    C_i = sum_r W_i P_{i,r}.
    
    We want C_0 = A_00 B_0 + A_01 B_1
    C_1 = A_10 B_0 + A_11 B_1
    
    If rank 1: C_0 = W_0 P_0, C_1 = W_1 P_1.
    P_0 is linear in A, B.
    P_0 = U_00 V_0 + U_01 V_1.
    P_1 = U_10 V_0 + U_11 V_1.
    
    We need:
    W_0 (U_00 V_0 + U_01 V_1) = A_00 B_0 + A_01 B_1
    W_1 (U_10 V_0 + U_11 V_1) = A_10 B_0 + A_11 B_1
    
    This requires cross terms to vanish or combine.
    It seems Rank 2 is needed.
    Let's construct Rank 2 for 2x2x1.
    
    Rank 2:
    r=0: A_00 B_0. U[0]=1, V[0]=1, W[0]=1. (Contributes to C_0)
    r=1: A_01 B_1. U[1]=1, V[1]=1, W[0]=1. (Contributes to C_0)
    r=2: A_10 B_0. U[2]=1, V[0]=1, W[1]=1. (Contributes to C_1)
    r=3: A_11 B_1. U[3]=1, V[1]=1, W[1]=1. (Contributes to C_1)
    
    This is rank 4 (schoolbook). Can we do better?
    
    Try Rank 3?
    Or Rank 2?
    
    Let's look at the tensor T.
    T_{i,j,k}. k=0.
    T_{0,0,0}=1, T_{0,1,0}=1
    T_{1,0,0}=1, T_{1,1,0}=1
    
    Matrix slices:
    Slice k=0: [[1, 1], [1, 1]].
    This matrix has rank 1.
    So the tensor rank is at least 1.
    But it's not an outer product of vectors in i, j, k because k is fixed.
    Actually, if k is fixed, the tensor is T_{i,j} * delta_{k,0}.
    If T_{i,j} = u_i v_j, then rank 1.
    Here T_{i,j} = [[1,1],[1,1]] = [1,1]^T [1,1].
    So T_{i,j,k} = 1*1 * 1.
    Wait.
    C_i = A_{i0} B_0 + A_{i1} B_1.
    This is NOT an outer product of A and B.
    A is 2x2, B is 2x1.
    C is 2x1.
    C = A B.
    This is standard matrix mult.
    Rank of 2x2x1?
    
    Let's check Rank 2.
    We need:
    C_0 = sum_r U_{0,r} V_{0,r} W_0 + U_{1,r} V_{1,r} W_0 ...
    
    Let's use the known rank of 2x2x1.
    It is 3?
    Actually, for 2x2x1, the rank is 3.
    Reference: "On the rank of matrix multiplication".
    Rank(2x2x1) = 3.
    
    Let's find the decomposition for 2x2x1.
    
    We want to compute:
    c0 = a00 b0 + a01 b1
    c1 = a10 b0 + a11 b1
    
    Using 3 terms:
    1. (a00+a10)(b0) -> contributes a00 b0, a10 b0
    2. (a01+a11)(b1) -> contributes a01 b1, a11 b1
    3. (a00+a01-a10-a11)(b0-b1) ... complicated.
    
    Standard Strassen-like for 2x2x1:
    m1 = (a00+a10) b0
    m2 = (a01+a11) b1
    m3 = (a00-a10) (b0-b1) ? No.
    
    Let's try:
    c0 = 1/2 [ (a00+a10)b0 + (a01+a11)b1 + (a00+a01-a10-a11)(b0+b1) + ... ]
    
    Actually, simpler:
    c0 = a00 b0 + a01 b1
    c1 = a10 b0 + a11 b1
    
    Let u1 = a00 + a10, v1 = b0.
    Let u2 = a01 + a11, v2 = b1.
    Let u3 = a00 - a10, v3 = b0 - b1?
    
    Sum:
    u1 v1 + u2 v2 + u3 v3
    = (a00+a10)b0 + (a01+a11)b1 + (a00-a10)(b0-b1)
    = a00 b0 + a10 b0 + a01 b1 + a11 b1 + a00 b0 - a00 b1 - a10 b0 + a10 b1
    = 2 a00 b0 + a01 b1 + a11 b1 - a00 b1 + a10 b1
    
    This doesn't look like c0 + c1.
    
    Let's just use Rank 3 decomposition if it exists, or Rank 4.
    Given the small size, Rank 4 (schoolbook) is safe.
    But we want minimal.
    
    Actually, Rank(2x2x1) is 3.
    Decomposition:
    U, V, W.
    
    Let's try to construct it.
    
    If we can't find it quickly, we use schoolbook for the small blocks.
    
    So:
    Block 2x3x2 = 2x2x2 (Rank 7) + 2x1x2 (Rank 1) = 8.
    Block 2x3x1 = 2x2x1 (Rank 3 or 4) + 2x1x1 (Rank 1).
    
    If Rank(2x2x1)=3, Total = 8 + 3 + 1 = 12.
    If Rank(2x2x1)=4, Total = 8 + 4 + 1 = 13.
    
    Schoolbook was 18.
    12 or 13 is an improvement.
    
    Let's implement 2x2x1 with Rank 4 (schoolbook) to be safe, and 2x1x2 with Rank 1.
    
    Block 1: 2x2x2 (Rank 7)
    Block 2: 2x1x2 (Rank 1)
    Block 3: 2x2x1 (Rank 4)
    Block 4: 2x1x1 (Rank 1)
    
    Total Rank = 7 + 1 + 4 + 1 = 13.
    
    Let's verify 2x1x2 Rank 1.
    A (2x1), B (1x2). C (2x2).
    C_{i,k} = A_i B_k.
    U (2x1) shape (2, 1). V (1x2) shape (2, 1). W (2x2) shape (4, 1).
    U[0]=1, U[1]=1.
    V[0]=1, V[1]=1.
    W[0]=1 (C00), W[1]=1 (C01), W[2]=1 (C10), W[3]=1 (C11).
    
    Check C00: U0 V0 W0 = 1*1*1 = 1. Correct.
    Check C01: U0 V1 W1 = 1*1*1 = 1. Correct.
    Check C10: U1 V0 W2 = 1*1*1 = 1. Correct.
    Check C11: U1 V1 W3 = 1*1*1 = 1. Correct.
    
    Yes, Rank 1.
    
    So we have a decomposition of Rank 13.
    
    Can we do better?
    Maybe 2x2x1 can be Rank 3?
    If so, Rank 12.
    
    Let's try to find Rank 3 for 2x2x1.
    
    We need:
    c0 = a00 b0 + a01 b1
    c1 = a10 b0 + a11 b1
    
    Try:
    1. (a00+a10) b0 -> a00 b0 + a10 b0
    2. (a01+a11) b1 -> a01 b1 + a11 b1
    3. (a00-a10-a01+a11) (b0-b1)?
    
    Let's check:
    Term 1 + Term 2 + Term 3
    = a00 b0 + a10 b0 + a01 b1 + a11 b1 + (a00-a10-a01+a11)(b0-b1)
    = a00 b0 + a10 b0 + a01 b1 + a11 b1 + a00 b0 - a00 b1 - a10 b0 + a10 b1 - a01 b0 + a01 b1 + a11 b0 - a11 b1
    = 2 a00 b0 + 2 a01 b1 + a11 b0 - a00 b1 - a01 b0 + a10 b1
    
    This is not c0 + c1.
    
    Actually, there is a known identity for 2x2x1 rank 3.
    See "Tensor rank of matrix multiplication".
    
    Let's just use Rank 4 for now to ensure correctness. 13 < 18.
    
    """
    # Block 1: 2x2x2
    U1, V1, W1 = _build_block(2, 2, 2)
    
    # Block 2: 2x1x2
    U2, V2, W2 = _build_block(2, 1, 2)
    
    # Block 3: 2x2x1
    U3, V3, W3 = _build_block(2, 2, 1)
    
    # Block 4: 2x1x1
    U4, V4, W4 = _build_block(2, 1, 1)
    
    return _combine_blocks([(U1, V1, W1), (U2, V2, W2), (U3, V3, W3), (U4, V4, W4)])

def construct():
    start_time = time.time()
    deadline = 100.0  # 100s safety margin
    
    # Try to find a better decomposition?
    # For now, return the block decomposition.
    
    U, V, W = _decompose_2x3x3()
    
    if U is None:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

