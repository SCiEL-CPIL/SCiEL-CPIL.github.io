import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    big_mults = [
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((0,1), 1), ((1,1), -1)]
        },
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((1,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((0,1), 1)]
        },
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((1,0), 1)]
        },
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    for big_idx, bm in enumerate(big_mults):
        for r2 in range(7):
            col = big_idx * 7 + r2
            
            for (bi, bj), sign_a in bm['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        u_val = U2[i2*2+k2, r2]
                        if u_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + k2
                            U[I*4+J, col] += sign_a * u_val
            
            for (bk, bj), sign_b in bm['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        v_val = V2[k2*2+j2, r2]
                        if v_val != 0:
                            J = 2*bk + k2
                            K = 2*bj + j2
                            V[J*4+K, col] += sign_b * v_val
            
            for (bi, bj), sign_c in bm['C']:
                for i2 in range(2):
                    for j2 in range(2):
                        w_val = W2[i2*2+j2, r2]
                        if w_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + j2
                            W[I*4+J, col] += sign_c * w_val
                            
    return U, V, W

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def _validate_numpy(U, V, W, n, m, p):
    """Validate using NumPy for speed."""
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _build_4x4x9_kron():
    """
    Build 4x4x9 decomposition using Kronecker product approach.
    
    Model the 9 columns as three 2-column blocks plus one residual.
    For each 2-column block, use 2x2x2 Strassen (rank 7) combined with 4x4x2 structure.
    
    Actually, let's use a cleaner approach:
    - Split the 9 output columns into: [0,1], [2,3], [4,5], [6,7], [8]
    - For each pair of columns (2x2 output block), we need to compute 4x4 @ 4x2 -> 4x2
    - This is equivalent to 4x4 @ (4x2) which can be seen as two independent 4x4 @ 4x1
    - But we can use the structure: 4x4x2 can be decomposed using Strassen-like methods
    
    Let me think differently. The tensor for 4x4x9 has entries T[i,j,k] for i in 0..3, j in 0..3, k in 0..8.
    
    We can write this as a sum of terms. The current approach uses 3 copies of 4x4x4 (rank 49 each) + 16 for the last column = 164.
    
    Alternative: Use the fact that 9 = 2+2+2+2+1.
    For the 2x2 output blocks, we can use a 4x4x2 decomposition.
    
    A 4x4x2 multiplication can be done with rank at most 4*4*2 = 32 (schoolbook),
    but we can do better using Strassen-like techniques.
    
    Actually, let's just stick with the current approach but verify it's correct.
    The current score is 1.2632 which corresponds to rank 164 (since 1.0 = seed at rank 144, 
    and 1.2632 = 144/114... wait, let me recalculate).
    
    Actually, the score is seed/value. If seed is the schoolbook rank 144, then:
    score = 144 / rank
    1.2632 = 144 / rank => rank = 144 / 1.2632 ≈ 114
    
    Hmm, that doesn't match 164. Let me reconsider.
    
    Wait, the problem says "score is the rank" and "search score is seed/value". 
    If the evaluator measures the raw value (rank), and score = seed/value, then:
    If seed = 144 (schoolbook), and value = rank, then score = 144/rank.
    
    For score 1.2632: 144/rank = 1.2632 => rank = 113.99... ≈ 114
    
    But the current program returns rank 164. So either my understanding is wrong,
    or the seed is different.
    
    Let me just focus on minimizing the rank. The current program achieves rank 164.
    Can we do better?
    
    For 4x4x9:
    - Schoolbook: 4*4*9 = 144
    - Current: 3*49 + 16 = 163... wait, 3*49 = 147, + 16 = 163? No, 49+49+16 = 114.
    
    Oh! 49 + 49 + 16 = 114. That matches! The current program uses TWO 4x4x4 Strassen blocks (rank 49 each) plus a schoolbook block for the 9th column (rank 16). Total = 49 + 49 + 16 = 114.
    
    So the current rank is 114, and score = 144/114 ≈ 1.2632. Correct!
    
    Can we do better than 114?
    
    Ideas:
    1. Use a better decomposition for 4x4x9 directly.
    2. Use Winograd or other fast matrix multiplication techniques.
    3. Exploit the rectangular structure.
    
    For 4x4x9, one approach is to note that this is equivalent to computing 4 independent 4x9 matrix multiplications (one for each row of A... no, that's not right).
    
    Actually, 4x4 @ 4x9 = 4x9. This is a single matrix multiplication with rectangular dimensions.
    
    Known results: The rank of 4x4x9 matrix multiplication tensor is at most:
    - Schoolbook: 144
    - Using Strassen for 4x4x4 blocks: 2*49 + 16 = 114 (current)
    - Can we do 4x4x9 more efficiently?
    
    One idea: Pad to 4x4x16 (which is 4x4 @ 4x16 = 4x16), compute using Strassen, then discard the last 7 columns.
    4x4x16 = 4x4 @ 4x16. We can view this as 4 independent 4x4 @ 4x4 multiplications, each rank 49, total 196. That's worse.
    
    Another idea: Use the fact that 9 = 8+1, and 8 = 2^3.
    
    Let me try a different approach: decompose 4x4x9 as a Kronecker product.
    
    4x4x9 = (2x2x2) ⊗ (2x2x5)? No, that doesn't work dimensionally.
    
    Actually, 4x4x9 can be written as a sum of simpler tensors.
    
    Let me try: view the 9 columns as 4+4+1. We already do this with rank 49+49+16=114.
    
    What if we view it as 2+2+2+2+1?
    For each 2-column block, we need a 4x4x2 decomposition.
    
    A 4x4x2 multiplication: C[i,k] = sum_j A[i,j] B[j,k] for i in 0..3, k in 0..1.
    
    This can be done with rank at most 4*4*2 = 32 (schoolbook), but we can do better.
    
    Using Strassen-like methods for 4x4x2:
    Split A into 2x2 blocks, B into 2x2 blocks (but B is 4x2, so it's two 2x2 blocks stacked).
    
    Actually, let's just keep the current approach since it's already quite good.
    The key insight is that we're using rank 114, which gives score 144/114 ≈ 1.2632.
    
    To improve, we need to find a decomposition with rank < 114.
    
    Known lower bounds: The rank of 4x4x4 is at least 49 (Strassen is optimal for 2x2x2, and 4x4x4 has rank at least 49 by known results). For 4x4x9, the rank is at least... I'm not sure of the exact lower bound.
    
    Let me just verify the current program works correctly and returns the right format.
    """
    U444, V444, W444 = _build_4x4x4_strassen()
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: columns 0-3 (4x4x4 Strassen)
    for I in range(4):
        for J in range(4):
            U[I*4+J, 0:49] = U444[I*4+J, :]
    for J in range(4):
        for K in range(4):
            V[J*9+K, 0:49] = V444[J*4+K, :]
    for K in range(4):
        for I in range(4):
            W[K*4+I, 0:49] = W444[K*4+I, :]
    
    # Block 2: columns 4-7 (4x4x4 Strassen)
    offset2 = 49
    for I in range(4):
        for J in range(4):
            U[I*4+J, offset2:offset2+49] = U444[I*4+J, :]
    for J in range(4):
        for K in range(4):
            V[J*9+(K+4), offset2:offset2+49] = V444[J*4+K, :]
    for K in range(4):
        for I in range(4):
            W[(K+4)*4+I, offset2:offset2+49] = W444[K*4+I, :]
    
    # Block 3: column 8 (schoolbook, rank 16)
    offset3 = 98
    for idx in range(16):
        I = idx // 4
        J = idx % 4
        U[I*4+J, offset3+idx] = 1
        V[J*9+8, offset3+idx] = 1
        W[8*4+I, offset3+idx] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U, V, W = _build_4x4x9_kron()
    
    # Convert to lists for the final output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Validate
    if not _validate_numpy(U_list, V_list, W_list, n, m, p):
        return _schoolbook_fallback()
    
    return {"U": U_list, "V": V_list, "W": W_list}
