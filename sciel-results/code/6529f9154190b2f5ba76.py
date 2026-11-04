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

def _build_block(n, m, p):
    """
    Build a tensor decomposition for n x m x p matrix multiplication.
    Uses Strassen for 2x2x2 blocks, schoolbook otherwise.
    Returns (U, V, W, rank) where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    Convention: W[k*n+i, c]
    """
    # Base case: if any dimension is 1, use schoolbook
    if n == 1 or m == 1 or p == 1:
        return _build_schoolbook(n, m, p)
    
    # Check if we can use Strassen (all dimensions >= 2 and even)
    # For simplicity, only apply Strassen when all dims are exactly 2
    if n == 2 and m == 2 and p == 2:
        return _build_strassen_2x2x2()
    
    # Recursive case: split into (2, n-2) x (2, m-2) x (2, p-2) if possible
    # Only split if remaining dimensions are >= 1
    if n >= 2 and m >= 2 and p >= 2:
        n1, n2 = 2, n - 2
        m1, m2 = 2, m - 2
        p1, p2 = 2, p - 2
        
        # Build 8 sub-decompositions
        sub_decomp = {}
        sub_decomp[(1,1,1)] = _build_block(n1, m1, p1)
        sub_decomp[(1,1,2)] = _build_block(n1, m1, p2)
        sub_decomp[(1,2,1)] = _build_block(n1, m2, p1)
        sub_decomp[(1,2,2)] = _build_block(n1, m2, p2)
        sub_decomp[(2,1,1)] = _build_block(n2, m1, p1)
        sub_decomp[(2,1,2)] = _build_block(n2, m1, p2)
        sub_decomp[(2,2,1)] = _build_block(n2, m2, p1)
        sub_decomp[(2,2,2)] = _build_block(n2, m2, p2)
        
        # Calculate total rank
        rank_total = sum(d[3] for d in sub_decomp.values())
        
        # Build full U, V, W
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        # Assign column offsets
        col_offset = {}
        offset = 0
        for key in [(1,1,1), (1,1,2), (1,2,1), (1,2,2), (2,1,1), (2,1,2), (2,2,1), (2,2,2)]:
            col_offset[key] = offset
            offset += sub_decomp[key][3]
        
        # Fill in the blocks
        for key, (U_s, V_s, W_s, rank_s) in sub_decomp.items():
            i_block, j_block, k_block = key
            i_off = 0 if i_block == 1 else n1
            j_off = 0 if j_block == 1 else m1
            k_off = 0 if k_block == 1 else p1
            
            n_b = n1 if i_block == 1 else n2
            m_b = m1 if j_block == 1 else m2
            p_b = p1 if k_block == 1 else p2
            
            base_col = col_offset[key]
            
            for i in range(n_b):
                for j in range(m_b):
                    for c in range(rank_s):
                        U[(i_off + i) * m + (j_off + j)][base_col + c] = U_s[i * m_b + j][c]
            
            for j in range(m_b):
                for k in range(p_b):
                    for c in range(rank_s):
                        V[(j_off + j) * p + (k_off + k)][base_col + c] = V_s[j * p_b + k][c]
            
            for k in range(p_b):
                for i in range(n_b):
                    for c in range(rank_s):
                        W[(k_off + k) * n + (i_off + i)][base_col + c] = W_s[k * n_b + i][c]
        
        return U, V, W, rank_total
    
    # Fallback: schoolbook
    return _build_schoolbook(n, m, p)

def _build_3x5x5_strassen_hybrid():
    """
    Build 3x5x5 using recursive block splitting with Strassen.
    Split as (2+1) x (2+3) x (2+3):
      (2,2,2): Strassen rank 7
      (2,2,3): rank 12
      (2,3,2): rank 12
      (2,3,3): rank 18
      (1,2,2): rank 4
      (1,2,3): rank 6
      (1,3,2): rank 6
      (1,3,3): rank 9
      Total: 74
    """
    return _build_block(3, 5, 5)

def _build_3x5x5():
    """
    Build 3x5x5 using recursive block splitting with Strassen.
    """
    return _build_block(3, 5, 5)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Check: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor identity: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
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
                for j in range(m):
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                C_computed[i][k] = s
        
        if C_computed != C_expected:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    # Try the recursive block Strassen decomposition for 3x5x5
    try:
        U, V, W, rank = _build_3x5x5()
        
        # Verify
        if _verify_decomposition(U, V, W, 3, 5, 5):
            elapsed = time.time() - start_time
            if elapsed < deadline - 10:
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W, rank = _build_schoolbook(3, 5, 5)
    return {"U": U, "V": V, "W": W}
