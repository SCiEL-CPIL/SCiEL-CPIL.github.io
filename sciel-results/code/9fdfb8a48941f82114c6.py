"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
    """Return (U, V, W) for schoolbook [n,m,p] with rank n*m*p."""
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

def _strassen_222_factors():
    """
    Return U (4x7), V (4x7), W (4x7) for 2x2*2x2=2x2 using Strassen (rank 7).
    A[i,j]: i=0,1 rows; j=0,1 cols
    B[j,k]: j=0,1 rows; k=0,1 cols  
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    U index: i*2+j (A entry)
    V index: j*2+k (B entry)
    W index: k*2+i (C entry)
    
    Strassen products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-b)(e+f)
    M7 = (a+d)(g+h)
    
    Output:
    C[0,0] = M1 - M4 + M6 - M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M5 - M2
    """
    # Verify Strassen numerically
    def strassen_2x2(A, B):
        a, b = A[0][0], A[0][1]
        c, d = A[1][0], A[1][1]
        e, f = B[0][0], B[0][1]
        g, h = B[1][0], B[1][1]
        M1 = (a + d) * (e + h)
        M2 = (c + d) * e
        M3 = a * (f - h)
        M4 = d * (g - e)
        M5 = (a + b) * h
        M6 = (c - b) * (e + f)
        M7 = (a + d) * (g + h)
        C00 = M1 - M4 + M6 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M3 + M5 - M2
        return [[C00, C01], [C10, C11]]
    
    # Verify with test case
    A_test = [[1, 2], [3, 4]]
    B_test = [[5, 6], [7, 8]]
    C_direct = [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]]
    C_strassen = strassen_2x2(A_test, B_test)
    
    if C_direct != C_strassen:
        # Fall back to schoolbook for 2x2
        U, V, W, rank = _schoolbook_factors(2, 2, 2)
        return U, V, W, rank
    
    # Build factor matrices for Strassen
    # Each M_r contributes to C[i,k] with coefficient W[k*2+i, r]
    # M_r = (sum_j U[i*2+j, r] * A[i,j]) * (sum_k V[j*2+k, r] * B[j,k])
    
    U = [[0]*7 for _ in range(4)]  # 4 A-entries
    V = [[0]*7 for _ in range(4)]  # 4 B-entries
    W = [[0]*7 for _ in range(4)]  # 4 C-entries
    
    # M1 = (a+d)(e+h): A-part: a+d, B-part: e+h
    # a = A[0,0] -> U[0*2+0, 0] = 1
    # d = A[1,1] -> U[1*2+1, 0] = 1
    # e = B[0,0] -> V[0*2+0, 0] = 1
    # h = B[1,1] -> V[1*2+1, 0] = 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C[0,0] gets +1, C[1,1] gets +1
    W[0*2+0][0] = 1  # C[0,0]
    W[1*2+1][0] = 1  # C[1,1]
    
    # M2 = (c+d)e: A-part: c+d, B-part: e
    # c = A[1,0] -> U[1*2+0, 1] = 1
    # d = A[1,1] -> U[1*2+1, 1] = 1
    # e = B[0,0] -> V[0*2+0, 1] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C[1,0] gets +1
    W[0*2+1][1] = 1  # C[1,0]
    
    # M3 = a(f-h): A-part: a, B-part: f-h
    # a = A[0,0] -> U[0*2+0, 2] = 1
    # f = B[0,1] -> V[0*2+1, 2] = 1
    # h = B[1,1] -> V[1*2+1, 2] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C[0,1] gets +1
    W[1*2+0][2] = 1  # C[0,1]
    
    # M4 = d(g-e): A-part: d, B-part: g-e
    # d = A[1,1] -> U[1*2+1, 3] = 1
    # g = B[1,0] -> V[1*2+0, 3] = 1
    # e = B[0,0] -> V[0*2+0, 3] = -1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C[0,0] gets -1, C[1,0] gets +1
    W[0*2+0][3] = -1  # C[0,0]
    W[0*2+1][3] = 1   # C[1,0]
    
    # M5 = (a+b)h: A-part: a+b, B-part: h
    # a = A[0,0] -> U[0*2+0, 4] = 1
    # b = A[0,1] -> U[0*2+1, 4] = 1
    # h = B[1,1] -> V[1*2+1, 4] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C[0,1] gets +1, C[1,1] gets +1
    W[1*2+0][4] = 1  # C[0,1]
    W[1*2+1][4] = 1  # C[1,1]
    
    # M6 = (c-b)(e+f): A-part: c-b, B-part: e+f
    # c = A[1,0] -> U[1*2+0, 5] = 1
    # b = A[0,1] -> U[0*2+1, 5] = -1
    # e = B[0,0] -> V[0*2+0, 5] = 1
    # f = B[0,1] -> V[0*2+1, 5] = 1
    U[2][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C[0,0] gets +1
    W[0*2+0][5] = 1  # C[0,0]
    
    # M7 = (a+d)(g+h): A-part: a+d, B-part: g+h
    # a = A[0,0] -> U[0*2+0, 6] = 1
    # d = A[1,1] -> U[1*2+1, 6] = 1
    # g = B[1,0] -> V[1*2+0, 6] = 1
    # h = B[1,1] -> V[1*2+1, 6] = 1
    U[0][6] = 1; U[3][6] = 1
    V[2][6] = 1; V[3][6] = 1
    # C[0,0] gets -1, C[1,1] gets -1
    W[0*2+0][6] = -1  # C[0,0]
    W[1*2+1][6] = -1  # C[1,1]
    
    return U, V, W, 7

def _strassen_221_factors():
    """
    Return U (2x2=4 entries), V (2x1=2 entries), W (1x2=2 entries) for [2,2,1] using rank 4 (schoolbook).
    Actually [2,2,1] is just dot products, so schoolbook rank = 2*2*1 = 4.
    But wait - can we do better? [2,2,1] means C[i,0] = sum_j A[i,j]*B[j,0].
    This is just two independent dot products of length 2, so rank = 4 (each needs 2 terms).
    Actually no - each C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0], so we need 4 rank-1 terms total.
    """
    n, m, p = 2, 2, 1
    rank = n * m * p  # 4
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

def _strassen_223_split_factors():
    """
    Decompose [2,2,3] by splitting output into [2,2,2] + [2,2,1].
    [2,2,2] uses Strassen (rank 7), [2,2,1] uses schoolbook (rank 4).
    Total rank: 11 (vs 12 schoolbook).
    """
    # Get Strassen for [2,2,2]
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    
    # Get schoolbook for [2,2,1]
    U_t, V_t, W_t, r_t = _strassen_221_factors()
    
    total_rank = r_s + r_t  # 7 + 4 = 11
    
    # Combine: U is 4 x 11, V is 6 x 11, W is 6 x 11
    U = [[0] * total_rank for _ in range(4)]   # 2*2 = 4 A-entries
    V = [[0] * total_rank for _ in range(6)]   # 2*3 = 6 B-entries
    W = [[0] * total_rank for _ in range(6)]   # 3*2 = 6 C-entries
    
    # First r_s columns: [2,2,2] part (output cols 0,1)
    for i in range(4):
        for r in range(r_s):
            U[i][r] = U_s[i][r]
    for j in range(2):
        for k in range(2):
            for r in range(r_s):
                V[j * 3 + k][r] = V_s[j * 2 + k][r]
    for k in range(2):
        for i in range(2):
            for r in range(r_s):
                W[k * 2 + i][r] = W_s[k * 2 + i][r]
    
    # Last r_t columns: [2,2,1] part (output col 2)
    for i in range(4):
        for r in range(r_t):
            U[i][r_s + r] = U_t[i][r]
    for j in range(2):
        for k in range(1):
            for r in range(r_t):
                V[j * 3 + 2][r_s + r] = V_t[j * 1 + k][r]
    for k in range(1):
        for i in range(2):
            for r in range(r_t):
                W[2 * 2 + i][r_s + r] = W_t[k * 2 + i][r]
    
    return U, V, W, total_rank

def _strassen_223_rank10_factors():
    """
    Attempt rank-10 decomposition of [2,2,3] matrix multiplication.
    A (2x2) * B (2x3) = C (2x3)
    
    We use a combination of Strassen-like ideas for the 2x2x2 part
    and exploit structure to reduce the 2x2x1 part.
    
    C[0,0] = a*e + c*h
    C[0,1] = a*f + c*i  
    C[0,2] = a*g + c*j
    C[1,0] = b*e + d*h
    C[1,1] = b*f + d*i
    C[1,2] = b*g + d*j
    
    Using 10 rank-1 terms:
    M1 = (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1  (wait, this is for 2x2x2)
    
    Let me think differently. We have 6 outputs, 12 inputs.
    Schoolbook: 12 terms (one per input pair that contributes)
    
    Actually for [2,2,3]:
    Each C[i,k] = sum_j A[i,j]*B[j,k]
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    That's 12 terms in schoolbook. Can we share computations?
    
    Let me try:
    M1 = A[0,0]*B[0,0] -> C[0,0]
    M2 = A[0,1]*B[1,0] -> C[0,0]
    M3 = A[1,0]*B[0,0] -> C[1,0]
    M4 = A[1,1]*B[1,0] -> C[1,0]
    
    For column 1:
    M5 = A[0,0]*B[0,1] -> C[0,1]
    M6 = A[0,1]*B[1,1] -> C[0,1]
    M7 = A[1,0]*B[0,1] -> C[1,1]
    M8 = A[1,1]*B[1,1] -> C[1,1]
    
    For column 2:
    M9 = A[0,0]*B[0,2] -> C[0,2]
    M10 = A[0,1]*B[1,2] -> C[0,2]
    M11 = A[1,0]*B[0,2] -> C[1,2]
    M12 = A[1,1]*B[1,2] -> C[1,2]
    
    Can we combine? Note that A[0,0] appears in M1,M5,M9.
    A[0,1] appears in M2,M6,M10.
    A[1,0] appears in M3,M7,M11.
    A[1,1] appears in M4,M8,M12.
    
    What if we define:
    P1 = A[0,0], P2 = A[0,1], P3 = A[1,0], P4 = A[1,1]
    Q1 = B[0,0], Q2 = B[0,1], Q3 = B[0,2], Q4 = B[1,0], Q5 = B[1,1], Q6 = B[1,2]
    
    C[0,0] = P1*Q1 + P2*Q4
    C[0,1] = P1*Q2 + P2*Q5
    C[0,2] = P1*Q3 + P2*Q6
    C[1,0] = P3*Q1 + P4*Q4
    C[1,1] = P3*Q2 + P4*Q5
    C[1,2] = P3*Q3 + P4*Q6
    
    This is like two independent [1,2,3] problems! Each needs rank 6 in schoolbook.
    But we can use Strassen-like tricks.
    
    For [1,2,3]: C[k] = P1*Qk+1 + P2*Qk+4 for k=0,1,2
    That's 6 terms. Can we do it in 5? Probably not since it's a bilinear form of rank 6.
    
    So [2,2,3] = two [1,2,3] problems, each of rank 6, total rank 12 in schoolbook.
    The current approach gets 11. Getting to 10 would require non-trivial sharing.
    
    Let me just keep the rank-11 version for now and try other optimizations.
    """
    # Fall back to the rank-11 version
    return _strassen_223_split_factors()

def _block_decomposition():
    """
    Decompose [3,5,5] using block structure with Strassen for 2x2x2 blocks.
    
    Split A (3x5) into A_top (2x5) rows 0,1 and A_bot (1x5) row 2
    Split B (5x5) into B_left (5x2) cols 0,1 and B_right (5x3) cols 2,3,4
    
    C = [A_top*B_left (2x2), A_top*B_right (2x3);
         A_bot*B_left (1x2), A_bot*B_right (1x3)]
    
    For each sub-problem, split middle dim 5 into 2+2+1:
    - [2,5,2]: Strassen for two [2,2,2] blocks + schoolbook [2,1,2] = 7+7+4 = 18
    - [2,5,3]: split each [2,2,3] into [2,2,2]+[2,2,1] = (7+4)+(7+4)+6 = 28
    - [1,5,2]: schoolbook = 10
    - [1,5,3]: schoolbook = 15
    Total: 18+28+10+15 = 71
    """
    n, m, p = 3, 5, 5
    
    # Calculate total rank
    # [2,5,2]: 7+7+4 = 18
    # [2,5,3]: (7+4)+(7+4)+6 = 28
    # [1,5,2]: 4+4+2 = 10
    # [1,5,3]: 6+6+3 = 15
    total_rank = 18 + 28 + 10 + 15
    U = [[0] * total_rank for _ in range(n * m)]  # 15 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 15 x total_rank
    
    col = 0
    
    # Helper: add a sub-decomposition to the main factors
    def add_subproblem(sub_n, sub_m, sub_p, row_offset_A, col_offset_B, mid_offset, 
                       sub_U, sub_V, sub_W, sub_rank):
        nonlocal col
        for i in range(sub_n):
            for j in range(sub_m):
                for r in range(sub_rank):
                    U[(row_offset_A + i) * m + (mid_offset + j)][col + r] = sub_U[i * sub_m + j][r]
        for j in range(sub_m):
            for k in range(sub_p):
                for r in range(sub_rank):
                    V[(mid_offset + j) * p + (col_offset_B + k)][col + r] = sub_V[j * sub_p + k][r]
        for k in range(sub_p):
            for i in range(sub_n):
                for r in range(sub_rank):
                    W[(col_offset_B + k) * n + (row_offset_A + i)][col + r] = sub_W[k * sub_n + i][r]
        col += sub_rank
    
    # Sub-problem 1: [2,5,2] -> C_top_left (2x2)
    # Split middle dim 5 into 2+2+1: indices 0,1 | 2,3 | 4
    
    # 1a: [2,2,2] using Strassen
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    add_subproblem(2, 2, 2, 0, 0, 0, U_s, V_s, W_s, r_s)
    
    # 1b: [2,2,2] using Strassen
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    add_subproblem(2, 2, 2, 0, 0, 2, U_s, V_s, W_s, r_s)
    
    # 1c: [2,1,2] schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 1, 2)
    add_subproblem(2, 1, 2, 0, 0, 4, U_s, V_s, W_s, r_s)
    
    # Sub-problem 2: [2,5,3] -> C_top_right (2x3)
    # Split middle dim 5 into 2+2+1: indices 0,1 | 2,3 | 4
    
    # 2a: [2,2,3] using split (Strassen [2,2,2] + schoolbook [2,2,1])
    U_s, V_s, W_s, r_s = _strassen_223_split_factors()
    add_subproblem(2, 2, 3, 0, 2, 0, U_s, V_s, W_s, r_s)
    
    # 2b: [2,2,3] using split
    U_s, V_s, W_s, r_s = _strassen_223_split_factors()
    add_subproblem(2, 2, 3, 0, 2, 2, U_s, V_s, W_s, r_s)
    
    # 2c: [2,1,3] schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 1, 3)
    add_subproblem(2, 1, 3, 0, 2, 4, U_s, V_s, W_s, r_s)
    
    # Sub-problem 3: [1,5,2] -> C_bot_left (1x2)
    # Split middle dim 5 into 2+2+1: indices 0,1 | 2,3 | 4
    
    # 3a: [1,2,2] schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 2)
    add_subproblem(1, 2, 2, 2, 0, 0, U_s, V_s, W_s, r_s)
    
    # 3b: [1,2,2] schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 2)
    add_subproblem(1, 2, 2, 2, 0, 2, U_s, V_s, W_s, r_s)
    
    # 3c: [1,1,2] schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 1, 2)
    add_subproblem(1, 1, 2, 2, 0, 4, U_s, V_s, W_s, r_s)
    
    # Sub-problem 4: [1,5,3] -> C_bot_right (1x3)
    # Split middle dim 5 into 2+2+1: indices 0,1 | 2,3 | 4
    
    # 4a: [1,2,3] schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 3)
    add_subproblem(1, 2, 3, 2, 2, 0, U_s, V_s, W_s, r_s)
    
    # 4b: [1,2,3] schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 3)
    add_subproblem(1, 2, 3, 2, 2, 2, U_s, V_s, W_s, r_s)
    
    # 4c: [1,1,3] schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 1, 3)
    add_subproblem(1, 1, 3, 2, 2, 4, U_s, V_s, W_s, r_s)
    
    return U, V, W, col

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition
    try:
        U, V, W, rank = _block_decomposition()
        
        # Verify correctness using vectorized approach for speed
        if time.time() < deadline:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Check: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Test with random matrices
            np.random.seed(42)
            A_test = np.random.randint(-3, 4, size=(n, m))
            B_test = np.random.randint(-3, 4, size=(m, p))
            C_expected = A_test @ B_test
            
            # Vectorized verification
            C_computed = np.zeros((n, p), dtype=float)
            for r in range(rank):
                # For each rank component, compute contribution
                u_col = U_arr[:, r]  # (n*m,)
                v_col = V_arr[:, r]  # (m*p,)
                w_col = W_arr[:, r]  # (p*n,)
                
                # Reshape to matrices
                u_mat = u_col.reshape(n, m)  # (n, m)
                v_mat = v_col.reshape(m, p)  # (m, p)
                w_mat = w_col.reshape(p, n)  # (p, n)
                
                # Contribution to C: w_mat * (u_mat * v_mat) element-wise? No.
                # C[i,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
                # = (U_col as n×m) * (V_col as m×p) gives n×p matrix, then element-wise * W_col reshaped
                # But W_col is indexed by k*n+i, so W_col.reshape(p,n) gives W[k,i]
                # C_contrib[i,k] = sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                
                # This is not a simple matrix product. Let's do it differently.
                # For fixed r: C[i,k] += W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
                
                # Compute sum_j U[i*m+j, r] * V[j*p+k, r] for all i,k
                # This is like: for each i,k, dot product of U[i*m:(i+1)*m, r] and V[:, k] reshaped
                # Actually V[j*p+k, r] for fixed k is V[:, k] but V is (m*p, rank)
                # V[j*p+k, r] = V_col[j*p+k]
                
                # Let's just do it with numpy operations
                for i in range(n):
                    u_row = u_col[i*m:(i+1)*m]  # (m,)
                    for k in range(p):
                        v_col_k = v_col[k::p]  # (m,) - V[j*p+k] for j=0..m-1
                        C_computed[i, k] += w_col[k*n+i] * np.dot(u_row, v_col_k)
            
            if np.allclose(C_computed, C_expected, atol=1e-9):
                # Second verification with different seed
                np.random.seed(123)
                A_test2 = np.random.randint(-3, 4, size=(n, m))
                B_test2 = np.random.randint(-3, 4, size=(m, p))
                C_expected2 = A_test2 @ B_test2
                
                C_computed2 = np.zeros((n, p), dtype=float)
                for r in range(rank):
                    u_col = U_arr[:, r]
                    v_col = V_arr[:, r]
                    w_col = W_arr[:, r]
                    for i in range(n):
                        u_row = u_col[i*m:(i+1)*m]
                        for k in range(p):
                            v_col_k = v_col[k::p]
                            C_computed2[i, k] += w_col[k*n+i] * np.dot(u_row, v_col_k)
                
                if np.allclose(C_computed2, C_expected2, atol=1e-9):
                    return {"U": U, "V": V, "W": W}
            # Fall through to schoolbook if verification fails
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W, rank = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
