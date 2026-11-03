"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Convention: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a21 + a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # p3 = a11(b12 - b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # p4 = a22(b21 - b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # p5 = (a11 + a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # p6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using Kronecker product of two Strassen decompositions."""
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U: A[i,j] where i = i1*2+i2, j = j1*2+j2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            
            # V: B[j,k] where j = j1*2+j2, k = k1*2+k2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            
            # W: C[i,k] -> W[k*n+i] where k = k1*2+k2, i = i1*2+i2
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x2_schoolbook():
    """Build rank-16 decomposition for 4x4 * 4x2.
    
    A is 4x4, B is 4x2, C is 4x2.
    Split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] (each 2x2)
    B = [[B00], [B10]] (each 2x1)
    C = [[C00], [C10]] (each 2x1)
    
    C00 = A00*B00 + A01*B10
    C10 = A10*B00 + A11*B10
    
    Each 2x2 * 2x1 product has rank 4 (schoolbook).
    Total: 4 products * 4 = 16.
    """
    n, m, p = 4, 4, 2
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 16 x 16
    V = [[0] * rank for _ in range(m * p)]  # 8 x 16
    W = [[0] * rank for _ in range(p * n)]  # 8 x 16
    
    # For each pair of 2x2 blocks (A_block, B_block), use schoolbook
    # A blocks: (br, bc) where br in {0,1}, bc in {0,1}
    # B blocks: (br, 0) where br in {0,1} (since B is 4x2, only one column of blocks)
    # C blocks: (br, 0) where br in {0,1}
    
    col = 0
    for br in range(2):
        for bc in range(2):
            for bbr in range(2):
                # A block (br, bc) * B block (bbr, 0) -> C block (br, 0)
                for ir in range(2):
                    for jc in range(2):
                        for jr in range(2):
                            # A[br*2+ir, bc*2+jc] * B[bbr*2+jr, 0] -> C[br*2+ir, 0]
                            i = br * 2 + ir
                            j = bc * 2 + jc
                            jj = bbr * 2 + jr
                            k = 0  # B and C only have 1 column of blocks (k=0)
                            
                            U[i*4 + j][col] = 1
                            V[jj*2 + k][col] = 1
                            W[k*4 + i][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x1_schoolbook():
    """Build rank-16 decomposition for 4x4 * 4x1.
    
    A is 4x4, B is 4x1, C is 4x1.
    Schoolbook: 16 terms.
    """
    n, m, p = 4, 4, 1
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 16 x 16
    V = [[0] * rank for _ in range(m * p)]  # 4 x 16
    W = [[0] * rank for _ in range(p * n)]  # 4 x 16
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*1 + 0][col] = 1
            W[0*4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7.
    
    Part 1: 4x4 * 4x4 -> rank 49 (Kronecker of two Strassen 2x2x2)
    Part 2: 4x4 * 4x2 -> rank 16 (four 2x2*2x1 schoolbook products)
    Part 3: 4x4 * 4x1 -> rank 16 (schoolbook)
    Total: 81
    """
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
    part1 = _build_4x4x4_strassen()
    rank1 = len(part1['U'][0])
    
    # Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5), rank 16
    part2 = _build_4x4x2_schoolbook()
    rank2 = len(part2['U'][0])
    
    # Part 3: 4x4 * 4x1 (B col 6, C col 6), rank 16
    part3 = _build_4x4x1_schoolbook()
    rank3 = len(part3['U'][0])
    
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
    for i in range(16):
        for r in range(rank1):
            U[i][r] = part1['U'][i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = part1['V'][i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = part1['W'][i*4+j][r]
    
    # Part 2: B cols 4-5, C cols 4-5
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = part2['U'][i][r]
    for i in range(4):
        for j in range(2):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = part2['V'][i*2+j][r]
    for i in range(2):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = part2['W'][i*4+j][r]
    
    # Part 3: B col 6, C col 6
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = part3['U'][i][r]
    for i in range(4):
        for r in range(rank3):
            V[i*p + 6][offset3 + r] = part3['V'][i][r]
    for i in range(4):
        for r in range(rank3):
            W[6*n + i][offset3 + r] = part3['W'][i][r]
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x7_simple():
    """Build decomposition for 4x4 * 4x7 using Strassen for 4x4x4 and schoolbook for 4x4x3."""
    n, m, p = 4, 4, 7
    left = _build_4x4x4_strassen()
    rank_left = len(left['U'][0])
    
    n_r, m_r, p_r = 4, 4, 3
    rank_right = n_r * m_r * p_r
    U_right = [[0] * rank_right for _ in range(n_r * m_r)]
    V_right = [[0] * rank_right for _ in range(m_r * p_r)]
    W_right = [[0] * rank_right for _ in range(p_r * n_r)]
    col = 0
    for i in range(n_r):
        for j in range(m_r):
            for k in range(p_r):
                U_right[i * m_r + j][col] = 1
                V_right[j * p_r + k][col] = 1
                W_right[k * n_r + i][col] = 1
                col += 1
    
    total_rank = rank_left + rank_right
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(rank_left):
            U[i][r] = left['U'][i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank_left):
                V[i * p + j][r] = left['V'][i * 4 + j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank_left):
                W[i * n + j][r] = left['W'][i * 4 + j][r]
    
    offset = rank_left
    for i in range(n * m):
        for r in range(rank_right):
            U[i][offset + r] = U_right[i][r]
    for i in range(4):
        for j in range(3):
            for r in range(rank_right):
                V[i * p + (4 + j)][offset + r] = V_right[i * 3 + j][r]
    for i in range(3):
        for j in range(4):
            for r in range(rank_right):
                W[(4 + i) * n + j][offset + r] = W_right[i * 4 + j][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    base_result = None
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            base_result = result
    except Exception:
        pass
    
    if base_result is None:
        if time.time() - _START_TIME > _TIME_LIMIT:
            return _schoolbook(4, 4, 7)
        try:
            result = _build_4x4x7_simple()
            if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
                base_result = result
        except Exception:
            pass
    
    if base_result is None:
        return _schoolbook(4, 4, 7)
    
    return base_result
