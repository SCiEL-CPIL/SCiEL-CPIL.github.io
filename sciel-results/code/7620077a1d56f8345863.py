"""Matrix multiplication 3x5x5 - structured decomposition via recursive Strassen."""
import time
import itertools

DIMENSIONS = [3, 5, 5]

def _matmul_small(a, b):
    """Exact matrix multiplication for small integer matrices."""
    n = len(a)
    m = len(a[0])
    p = len(b[0])
    result = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0
            for k in range(m):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result

def _build_schoolbook(n, m, p):
    """Return (U, V, W, rank) for schoolbook n x m x p."""
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
    return U, V, W, rank

def _build_strassen_2x2x2():
    """Return (U, V, W, rank=7) for 2x2x2 Strassen."""
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    # Strassen: M1=(a11-a22)(b21+b22), M2=(a21+a22)b11, M3=a11(b11-b22),
    # M4=a22(b22-b11), M5=(a11+a12)b22, M6=(a21-a11)(b11+b12), M7=(a12-a22)(b21+b22)
    # c11=M1+M4-M5+M7, c12=M3+M5, c21=M2+M4, c22=M1-M2+M3+M6
    # U rows: a11, a12, a21, a22 (index i*m+j)
    # V rows: b11, b12, b21, b22 (index j*p+k)
    # W rows: c11, c12, c21, c22 (index k*n+i)
    
    # M1 = (a11 - a22)(b21 + b22)
    U[0][0] = 1;  U[3][0] = -1
    V[2][0] = 1;  V[3][0] = 1
    W[0][0] = 1;  W[3][0] = 1
    
    # M2 = (a21 + a22)b11
    U[2][1] = 1;  U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1;  W[3][1] = -1
    
    # M3 = a11(b11 - b22)
    U[0][2] = 1
    V[0][2] = 1;  V[3][2] = -1
    W[1][2] = 1;  W[3][2] = 1
    
    # M4 = a22(b22 - b11)
    U[3][3] = 1
    V[3][3] = 1;  V[0][3] = -1
    W[0][3] = 1;  W[2][3] = 1
    
    # M5 = (a11 + a12)b22
    U[0][4] = 1;  U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1;  W[1][4] = 1
    
    # M6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1;  U[0][5] = -1
    V[0][5] = 1;  V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1;  U[3][6] = -1
    V[2][6] = 1;  V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, rank

def _build_strassen_3x3x3():
    """
    Build a 3x3x3 decomposition by padding to 4x4x4 with one zero row/col,
    then applying Strassen recursively.
    Actually, let's use a direct approach: decompose 3x3x3 using the known
    rank <= 23 result, or just use schoolbook for 3x3x3 (rank 27) since
    Strassen for 3x3 isn't standard.
    
    Alternative: Use the fact that we can embed 3x3 in 4x4 and use Strassen on 4x4.
    4x4x4 via Strassen has rank 7^2 = 49, but we only need the 3x3 part.
    Actually 4x4x4 Strassen: split into 2x2 blocks, each 2x2x2 uses rank 7.
    So 4x4x4 has rank 4*7 = 28. But we have a 3x3 matrix embedded in 4x4.
    
    Let me just use schoolbook for 3x3x3 = rank 27, and apply Strassen at the
    4x4 level for the 5x5 part.
    """
    return _build_schoolbook(3, 3, 3)

def _build_5x5_strassen():
    """
    Build 5x5x5 by embedding in 6x6x6 and using Strassen on 6x6.
    6x6x6: split into 3x3 blocks (2x2 grid), each 3x3x3 uses schoolbook rank 27.
    So 6x6x6 has rank 4*27 = 108. But we can do better.
    
    Actually, let's try: 5x5x5 embedded in 8x8x8.
    8x8x8 via Strassen: rank 7^3 = 343. That's worse than schoolbook 125.
    
    Better: 5x5x5 in 6x6x6.
    6x6x6: split into 2x2 blocks of 3x3. Each 3x3x3 is schoolbook rank 27.
    Total rank = 4 * 27 = 108.
    
    Or: 6x6x6: split into 3x3 blocks of 2x2. Each 2x2x2 is Strassen rank 7.
    Total rank = 9 * 7 = 63. That's better!
    
    So 5x5x5 embedded in 6x6x6 with rank 63.
    """
    # Build 6x6x6 by treating it as 3x3 blocks of 2x2x2 Strassen
    block_n = 3  # 3x3 blocks
    block_size = 2  # each block is 2x2
    total = block_n * block_size  # 6
    
    # First build one 2x2x2 Strassen
    U_s, V_s, W_s, rank_s = _build_strassen_2x2x2()
    
    # Now build 6x6x6
    rank_total = (block_n * block_n) * rank_s  # 9 * 7 = 63
    U = [[0] * rank_total for _ in range(total * total)]
    V = [[0] * rank_total for _ in range(total * total)]
    W = [[0] * rank_total for _ in range(total * total)]
    
    for bi in range(block_n):
        for bj in range(block_n):
            base_col = (bi * block_n + bj) * rank_s
            for li in range(block_size):
                for lj in range(block_size):
                    gi = bi * block_size + li  # global row of A
                    gj = bj * block_size + lj  # global col of A
                    # U index: gi * total + gj
                    u_idx = gi * total + gj
                    for li2 in range(block_size):
                        for lj2 in range(block_size):
                            gi2 = bi * block_size + li2
                            gj2 = bj * block_size + lj2
                            v_idx = gi2 * total + gj2
                            w_idx = gj2 * total + gi
                            for c in range(rank_s):
                                U[u_idx][base_col + c] += U_s[li * block_size + lj][c]
                                V[v_idx][base_col + c] += V_s[li2 * block_size + lj2][c]
                                W[w_idx][base_col + c] += W_s[li2 * block_size + lj2][c]
    
    return U, V, W, rank_total

def _build_3x5x5():
    """
    Build 3x5x5 by embedding in 6x6x6 (pad A from 3x5 to 6x6, B from 5x5 to 6x6).
    Use the 6x6x6 Strassen-based decomposition.
    """
    U6, V6, W6, rank6 = _build_5x5_strassen()
    total = 6
    
    # We need to project: A is 3x5 -> 6x6 (pad rows 3-5 and cols 5-5 with zeros)
    # B is 5x5 -> 6x6 (pad rows/cols 5-5 with zeros)
    # C is 3x5 -> 6x6 (we only care about rows 0-2, cols 0-4)
    
    # The tensor decomposition for 6x6x6 satisfies:
    # C[gk, gi] = sum_c U6[gi*6+gj, c] * V6[gj*6+gk, c] * W6[gk*6+gi, c]
    # Wait, the convention is W[k*n+i, c] where k is the column of C (which is col of B),
    # i is the row of C (which is row of A).
    
    # For 3x5x5: n=3, m=5, p=5
    # U has shape (n*m, rank) = (15, rank)
    # V has shape (m*p, rank) = (25, rank)  
    # W has shape (p*n, rank) = (15, rank)
    # W[k*n+i, c] where k in 0..4 (col of C), i in 0..2 (row of C)
    
    # In the 6x6x6 decomposition:
    # U6[gi*6+gj, c] for gi in 0..5, gj in 0..5
    # V6[gj*6+gk, c] for gj in 0..5, gk in 0..5
    # W6[gk*6+gi, c] for gk in 0..5, gi in 0..5
    
    # We want: C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..2, j in 0..4, k in 0..4
    # Using the 6x6x6 identity but restricting to valid indices:
    # C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k]
    # = sum_{j=0}^{5} A[i,j] B[j,k]  (since A[i,5]=0, B[5,k]=0 for our padding)
    
    # The 6x6x6 decomposition gives:
    # C6[gi,gk] = sum_{gj=0}^{5} U6[gi*6+gj, c] * V6[gj*6+gk, c] * W6[gk*6+gi, c] summed over c
    # But this equals A6[gi,gj] * B6[gj,gk] summed over gj.
    
    # So for gi in 0..2, gk in 0..4:
    # C[gi,gk] = sum_{gj=0}^{5} A6[gi,gj] B6[gj,gk] = sum_{gj=0}^{4} A[gi,gj] B[gj,gk]
    # (since A6[gi,5]=0 and B6[5,gk]=0)
    
    # Now extract the sub-decomposition:
    # U[i*m+j, c] for i in 0..2, j in 0..4 -> U6[i*6+j, c]
    # V[j*p+k, c] for j in 0..4, k in 0..4 -> V6[j*6+k, c]
    # W[k*n+i, c] for k in 0..4, i in 0..2 -> W6[k*6+i, c]
    
    n, m, p = 3, 5, 5
    U = [[0] * rank6 for _ in range(n * m)]
    V = [[0] * rank6 for _ in range(m * p)]
    W = [[0] * rank6 for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for c in range(rank6):
                U[i * m + j][c] = U6[i * total + j][c]
    
    for j in range(m):
        for k in range(p):
            for c in range(rank6):
                V[j * p + k][c] = V6[j * total + k][c]
    
    for k in range(p):
        for i in range(n):
            for c in range(rank6):
                W[k * n + i][c] = W6[k * total + i][c]
    
    return U, V, W, rank6

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Check: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_j U[i*m+j, c]) * (sum_k' V[j*p+k', c]) ... 
    # Actually: C[i,k] = sum_c U[i*m+j1, c] V[j1*p+k, c] W[k*n+i, c] summed over j1
    # More precisely: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Test with random integer matrices
    import random
    random.seed(42)
    for trial in range(5):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C_expected = _matmul_small(A, B)
        
        C_computed = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for c in range(rank):
                    uc = 0
                    for j in range(m):
                        uc += U[i * m + j][c] * A[i][j]
                    vc = 0
                    for j in range(m):
                        vc += V[j * p + k][c] * B[j][k]
                    s += uc * vc * W[k * n + i][c]
                C_computed[i][k] = s
        
        if C_computed != C_expected:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    # Try the structured 6x6x6-based decomposition for 3x5x5
    try:
        U, V, W, rank = _build_3x5x5()
        
        # Verify
        if _verify_decomposition(U, V, W, 3, 5, 5):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W, rank = _build_schoolbook(3, 5, 5)
    return {"U": U, "V": V, "W": W}
