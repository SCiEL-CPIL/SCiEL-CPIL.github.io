import time
import numpy as np

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 145

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p -> n x p."""
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
    return U, V, W, rank


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2. Rank 7.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # m1 = (a+d)(e+h)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0

    # m2 = (c+d)(e+f)
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0; V[1][1] = 1.0
    W[2][1] = 1.0; W[3][1] = -1.0

    # m3 = a(f-h)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0; W[3][2] = 1.0

    # m4 = d(g-e)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[2][3] = 1.0

    # m5 = (a+b)h
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0

    # m6 = (c-d)e
    U[2][5] = 1.0; U[3][5] = -1.0
    V[0][5] = 1.0
    W[3][5] = 1.0

    # m7 = (b-d)(f+g)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[1][6] = 1.0; V[2][6] = 1.0
    W[0][6] = 1.0

    return U, V, W, rank


def _block_decompose(n, m, p, n1, m1, p1):
    """
    Decompose matrix multiplication of size n x m * m x p -> n x p
    using a 2x2 block structure.
    """
    n2 = n - n1
    m2 = m - m1
    p2 = p - p1

    def _sub_decompose(nn, mm, pp):
        if nn == 2 and mm == 2 and pp == 2:
            return _strassen_2x2()
        else:
            return _schoolbook(nn, mm, pp)

    U1, V1, W1, r1 = _sub_decompose(n1, m1, p1)  # A11 * B11
    U2, V2, W2, r2 = _sub_decompose(n1, m2, p1)  # A12 * B21
    U3, V3, W3, r3 = _sub_decompose(n1, m1, p2)  # A11 * B12
    U4, V4, W4, r4 = _sub_decompose(n1, m2, p2)  # A12 * B22

    total_rank = r1 + r2 + r3 + r4

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0

    # Sub-problem 1: A11 * B11 -> C11
    for i in range(n1):
        for j in range(m1):
            for c in range(r1):
                U[i * m + j][col + c] = U1[i * m1 + j][c]
    for j in range(m1):
        for k in range(p1):
            for c in range(r1):
                V[j * p + k][col + c] = V1[j * p1 + k][c]
    for k in range(p1):
        for i in range(n1):
            for c in range(r1):
                W[k * n + i][col + c] = W1[k * n1 + i][c]
    col += r1

    # Sub-problem 2: A12 * B21 -> C11 (add)
    for i in range(n1):
        for j in range(m2):
            for c in range(r2):
                U[i * m + (m1 + j)][col + c] = U2[i * m2 + j][c]
    for j in range(m2):
        for k in range(p1):
            for c in range(r2):
                V[(m1 + j) * p + k][col + c] = V2[j * p1 + k][c]
    for k in range(p1):
        for i in range(n1):
            for c in range(r2):
                W[k * n + i][col + c] += W2[k * n1 + i][c]
    col += r2

    # Sub-problem 3: A11 * B12 -> C12
    for i in range(n1):
        for j in range(m1):
            for c in range(r3):
                U[i * m + j][col + c] = U3[i * m1 + j][c]
    for j in range(m1):
        for k in range(p2):
            for c in range(r3):
                V[j * p + (p1 + k)][col + c] = V3[j * p2 + k][c]
    for k in range(p2):
        for i in range(n1):
            for c in range(r3):
                W[(p1 + k) * n + i][col + c] = W3[k * n1 + i][c]
    col += r3

    # Sub-problem 4: A12 * B22 -> C12 (add)
    for i in range(n1):
        for j in range(m2):
            for c in range(r4):
                U[i * m + (m1 + j)][col + c] = U4[i * m2 + j][c]
    for j in range(m2):
        for k in range(p2):
            for c in range(r4):
                V[(m1 + j) * p + (p1 + k)][col + c] = V4[j * p2 + k][c]
    for k in range(p2):
        for i in range(n1):
            for c in range(r4):
                W[(p1 + k) * n + i][col + c] += W4[k * n1 + i][c]
    col += r4

    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i,j,k,r] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+r,c]
    # We want T[i,j,k,r] = 1 if i==r and j==k, else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    # 4x5x5
    
    # Strategy: Split 4x5x5 into blocks and use Strassen where possible.
    # Option 1: Split n=4 into 2+2, m=5 into 2+3, p=5 into 2+3
    # Sub-problems:
    # 1. 2x2x2 (Strassen, rank 7)
    # 2. 2x3x2 (Schoolbook, rank 12)
    # 3. 2x2x3 (Schoolbook, rank 12)
    # 4. 2x3x3 (Schoolbook, rank 18)
    # Total: 7 + 12 + 12 + 18 = 49
    
    # Option 2: Split n=4 into 2+2, m=5 into 3+2, p=5 into 3+2
    # Sub-problems:
    # 1. 2x3x3 (Schoolbook, rank 18)
    # 2. 2x2x3 (Schoolbook, rank 12)
    # 3. 2x3x2 (Schoolbook, rank 12)
    # 4. 2x2x2 (Strassen, rank 7)
    # Total: 18 + 12 + 12 + 7 = 49
    
    # Option 3: Split n=4 into 1+3, m=5 into 2+3, p=5 into 2+3
    # Sub-problems:
    # 1. 1x2x2 (Schoolbook, rank 4)
    # 2. 1x3x2 (Schoolbook, rank 6)
    # 3. 1x2x3 (Schoolbook, rank 6)
    # 4. 1x3x3 (Schoolbook, rank 9)
    # Total: 4 + 6 + 6 + 9 = 25
    # Plus the bottom part (n2=3) which needs to be handled.
    # Wait, the block decomposition only handles the top-left block C11 and C12.
    # The bottom blocks C21 and C22 are NOT handled in the current _block_decompose.
    # Let me re-read the _block_decompose function.
    
    # Looking at _block_decompose:
    # It only computes C11 and C12 (top part of C).
    # It does NOT compute C21 and C22.
    # This is a BUG in the original _block_decompose!
    # It only handles the first n1 rows of C.
    # The remaining n-n1 rows of C are not computed.
    
    # So the current _block_decompose is incomplete. It only computes part of C.
    # Let me verify by checking the verification.
    
    # Actually, looking more carefully at the original code:
    # _block_decompose only fills in W for the top n1 rows.
    # The bottom n2 rows of W are left as zeros.
    # This means the decomposition is incorrect unless n2=0.
    
    # So the original code has a bug. The _block_decompose function is incomplete.
    # It only handles C11 and C12, not C21 and C22.
    
    # Let me fix this by properly handling all 4 blocks.
    
    # For now, let me just use the schoolbook decomposition since the block
    # decomposition in the original code is buggy.
    
    # Actually, let me implement a correct block decomposition.
    
    def _full_block_decompose(n, m, p, n1, m1, p1):
        n2 = n - n1
        m2 = m - m1
        p2 = p - p1
        
        def _sub_decompose(nn, mm, pp):
            if nn == 2 and mm == 2 and pp == 2:
                return _strassen_2x2()
            else:
                return _schoolbook(nn, mm, pp)
        
        # 8 sub-problems for full 2x2 block decomposition:
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        U1, V1, W1, r1 = _sub_decompose(n1, m1, p1)  # A11 * B11
        U2, V2, W2, r2 = _sub_decompose(n1, m2, p1)  # A12 * B21
        U3, V3, W3, r3 = _sub_decompose(n1, m1, p2)  # A11 * B12
        U4, V4, W4, r4 = _sub_decompose(n1, m2, p2)  # A12 * B22
        U5, V5, W5, r5 = _sub_decompose(n2, m1, p1)  # A21 * B11
        U6, V6, W6, r6 = _sub_decompose(n2, m2, p1)  # A22 * B21
        U7, V7, W7, r7 = _sub_decompose(n2, m1, p2)  # A21 * B12
        U8, V8, W8, r8 = _sub_decompose(n2, m2, p2)  # A22 * B22
        
        total_rank = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8
        
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        # Sub-problem 1: A11 * B11 -> C11
        for i in range(n1):
            for j in range(m1):
                for c in range(r1):
                    U[i * m + j][col + c] = U1[i * m1 + j][c]
        for j in range(m1):
            for k in range(p1):
                for c in range(r1):
                    V[j * p + k][col + c] = V1[j * p1 + k][c]
        for k in range(p1):
            for i in range(n1):
                for c in range(r1):
                    W[k * n + i][col + c] = W1[k * n1 + i][c]
        col += r1
        
        # Sub-problem 2: A12 * B21 -> C11 (add)
        for i in range(n1):
            for j in range(m2):
                for c in range(r2):
                    U[i * m + (m1 + j)][col + c] = U2[i * m2 + j][c]
        for j in range(m2):
            for k in range(p1):
                for c in range(r2):
                    V[(m1 + j) * p + k][col + c] = V2[j * p1 + k][c]
        for k in range(p1):
            for i in range(n1):
                for c in range(r2):
                    W[k * n + i][col + c] += W2[k * n1 + i][c]
        col += r2
        
        # Sub-problem 3: A11 * B12 -> C12
        for i in range(n1):
            for j in range(m1):
                for c in range(r3):
                    U[i * m + j][col + c] = U3[i * m1 + j][c]
        for j in range(m1):
            for k in range(p2):
                for c in range(r3):
                    V[j * p + (p1 + k)][col + c] = V3[j * p2 + k][c]
        for k in range(p2):
            for i in range(n1):
                for c in range(r3):
                    W[(p1 + k) * n + i][col + c] = W3[k * n1 + i][c]
        col += r3
        
        # Sub-problem 4: A12 * B22 -> C12 (add)
        for i in range(n1):
            for j in range(m2):
                for c in range(r4):
                    U[i * m + (m1 + j)][col + c] = U4[i * m2 + j][c]
        for j in range(m2):
            for k in range(p2):
                for c in range(r4):
                    V[(m1 + j) * p + (p1 + k)][col + c] = V4[j * p2 + k][c]
        for k in range(p2):
            for i in range(n1):
                for c in range(r4):
                    W[(p1 + k) * n + i][col + c] += W4[k * n1 + i][c]
        col += r4
        
        # Sub-problem 5: A21 * B11 -> C21
        for i in range(n2):
            for j in range(m1):
                for c in range(r5):
                    U[(n1 + i) * m + j][col + c] = U5[i * m1 + j][c]
        for j in range(m1):
            for k in range(p1):
                for c in range(r5):
                    V[j * p + k][col + c] = V5[j * p1 + k][c]
        for k in range(p1):
            for i in range(n2):
                for c in range(r5):
                    W[k * n + (n1 + i)][col + c] = W5[k * n2 + i][c]
        col += r5
        
        # Sub-problem 6: A22 * B21 -> C21 (add)
        for i in range(n2):
            for j in range(m2):
                for c in range(r6):
                    U[(n1 + i) * m + (m1 + j)][col + c] = U6[i * m2 + j][c]
        for j in range(m2):
            for k in range(p1):
                for c in range(r6):
                    V[(m1 + j) * p + k][col + c] = V6[j * p1 + k][c]
        for k in range(p1):
            for i in range(n2):
                for c in range(r6):
                    W[k * n + (n1 + i)][col + c] += W6[k * n2 + i][c]
        col += r6
        
        # Sub-problem 7: A21 * B12 -> C22
        for i in range(n2):
            for j in range(m1):
                for c in range(r7):
                    U[(n1 + i) * m + j][col + c] = U7[i * m1 + j][c]
        for j in range(m1):
            for k in range(p2):
                for c in range(r7):
                    V[j * p + (p1 + k)][col + c] = V7[j * p2 + k][c]
        for k in range(p2):
            for i in range(n2):
                for c in range(r7):
                    W[(p1 + k) * n + (n1 + i)][col + c] = W7[k * n2 + i][c]
        col += r7
        
        # Sub-problem 8: A22 * B22 -> C22 (add)
        for i in range(n2):
            for j in range(m2):
                for c in range(r8):
                    U[(n1 + i) * m + (m1 + j)][col + c] = U8[i * m2 + j][c]
        for j in range(m2):
            for k in range(p2):
                for c in range(r8):
                    V[(m1 + j) * p + (p1 + k)][col + c] = V8[j * p2 + k][c]
        for k in range(p2):
            for i in range(n2):
                for c in range(r8):
                    W[(p1 + k) * n + (n1 + i)][col + c] += W8[k * n2 + i][c]
        col += r8
        
        return U, V, W, total_rank
    
    # Try different block splits and pick the one with lowest rank
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    splits = [
        (2, 2, 2),
        (2, 3, 3),
        (1, 2, 2),
        (1, 3, 3),
        (2, 2, 3),
        (2, 3, 2),
        (3, 2, 2),
        (3, 3, 3),
        (1, 1, 1),
        (1, 4, 4),
        (4, 1, 1),
        (2, 1, 1),
        (1, 2, 3),
        (3, 2, 3),
    ]
    
    for n1, m1, p1 in splits:
        if n1 < 1 or n1 >= n:
            continue
        if m1 < 1 or m1 >= m:
            continue
        if p1 < 1 or p1 >= p:
            continue
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        U, V, W, rank = _full_block_decompose(n, m, p, n1, m1, p1)
        
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    if best_U is None:
        U, V, W, rank = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W, rank = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
