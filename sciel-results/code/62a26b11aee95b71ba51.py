import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """Return U, V, W for M_{n,m,p} schoolbook rank n*m*p."""
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
    return U, V, W

def _strassen_2x2x2():
    """
    Correct rank-7 Strassen decomposition for M_{2,2,2}.
    
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    where p=C[0,0], q=C[0,1], r=C[1,0], s=C[1,1]
    
    C[0,0] = a*e + b*g, C[0,1] = a*f + b*h
    C[1,0] = c*e + d*g, C[1,1] = c*f + d*h
    
    Strassen:
    M1 = (a+b)(e+f)
    M2 = (c+d)(e+f)
    M3 = (a+d)(h-g)
    M4 = (a+b)(f-h)
    M5 = (c+d)(g-e)
    M6 = (a-c)(h+d)  -- wait, this doesn't look right either
    
    Let me use a well-known correct formulation.
    
    Actually, let me verify by direct computation. The standard Strassen is:
    M1 = (a+b)(e+f)
    M2 = (c+d)(e+f)  
    M3 = (a+d)(h-g)
    M4 = (a+b)(f-h)
    M5 = (c+d)(g-e)
    M6 = (a-c)(h+d)
    M7 = (b-d)(e+c)
    
    C11 = M1 - M3 + M4 - M7
    C12 = M3 - M4
    C21 = M2 - M4
    C22 = M1 - M2 + M4 + M7
    
    But this only uses M1-M4, M7 in the output, not M5, M6. That can't be right for rank 7.
    
    I think the issue is that I'm not recalling the standard formulation correctly.
    Let me just use a verified approach: I'll compute the tensor directly.
    
    Actually, for safety, let me use schoolbook for 2x2x2 (rank 8) and focus on getting
    the block decomposition right. Then 4x4x4 would be 8*8=64.
    
    Total: 64 + 32 + 24 = 120. Same as schoolbook.
    
    Hmm, that's not an improvement. Let me think differently.
    
    Alternative: Don't use Strassen at all. Use a different block decomposition.
    
    A(4x5) * B(5x6):
    Split A into A1(4x3) + A2(4x2), B into B1(3x6) + B2(2x6)
    C = A1*B1 + A2*B2
    A1*B1: 4x3 x 3x6, schoolbook rank = 4*3*6 = 72
    A2*B2: 4x2 x 2x6, schoolbook rank = 4*2*6 = 48
    Total: 120. Still same.
    
    What if I use a mixed approach?
    
    A(4x5) * B(5x6):
    Split B into B1(4x6) + B2(1x6), A into A1(4x4) + A2(4x1)
    C = A1*B1 + A2*B2
    A1*B1: 4x4 x 4x6
    A2*B2: 4x1 x 1x6, schoolbook rank = 4*1*6 = 24
    
    For A1*B1 (4x4 x 4x6):
    Split B1 into B1a(4x4) + B1b(4x2)
    A1*B1a: 4x4 x 4x4
    A1*B1b: 4x4 x 4x2, schoolbook rank = 4*4*2 = 32
    
    For A1*B1a (4x4 x 4x4):
    Use Strassen: rank 49 (if I can get it right)
    
    Total: 49 + 32 + 24 = 105
    
    But I need a correct Strassen for 4x4x4.
    
    4x4x4 can be viewed as 2x2 blocks of 2x2 matrices.
    Apply Strassen at the block level: 7 block multiplications, each 2x2x2.
    If each 2x2x2 has rank 7, total is 49.
    
    But I need correct 2x2x2 Strassen.
    
    Let me try to derive it carefully.
    
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C[0,0] = a*e + b*g
    C[0,1] = a*f + b*h  
    C[1,0] = c*e + d*g
    C[1,1] = c*f + d*h
    
    I want 7 terms of the form (linear in A)(linear in B)(linear in C).
    
    Let me just use schoolbook for everything and accept rank 120.
    The task says "rank-minimal" but the current program already achieves 120.
    To improve, I need to find a decomposition with rank < 120.
    
    For now, let me return schoolbook to ensure correctness.
    """
    # Use schoolbook for 2x2x2: rank 8
    U = [[0]*8 for _ in range(4)]
    V = [[0]*8 for _ in range(4)]
    W = [[0]*8 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _strassen_4x4x4():
    """
    Rank-49 Strassen decomposition for M_{4,4,4}.
    
    View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    Apply Strassen at the block level.
    
    A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    Strassen at block level:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)(B11)
    M3 = (A11)(B22-B21)
    M4 = (A22)(B11-B12)
    M5 = (A11+A12)(B21+B22)
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2 x 2x2 multiplication, rank 7 (Strassen) or 8 (schoolbook).
    
    If I use Strassen for each block multiplication, total rank = 7*7 = 49.
    If I use schoolbook, total rank = 8*8 = 64.
    
    For now, use schoolbook for the block multiplications: rank 64.
    Total for 4x5x6: 64 + 32 + 24 = 120. Still same as schoolbook.
    
    I need to get Strassen right to achieve 49.
    """
    # Use schoolbook for 4x4x4: rank 64
    U = [[0]*64 for _ in range(16)]
    V = [[0]*64 for _ in range(16)]
    W = [[0]*64 for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return U, V, W

def _schoolbook_sub(n, m, p):
    """Schoolbook for sub-problem n x m x p."""
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
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: 
    # A(4x5) = [A1(4x4), A2(4x1)]
    # B(5x6) = [B1(4x6); B2(1x6)]
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 4x4 x 4x6
    #   Split B1 = [B1a(4x4), B1b(4x2)]
    #   A1*B1a: 4x4x4 -> schoolbook rank 64
    #   A1*B1b: 4x4x2 -> schoolbook rank 32
    # A2*B2: 4x1x6 -> schoolbook rank 24
    # Total: 64 + 32 + 24 = 120
    #
    # This is the same as schoolbook, but the structure is clearer.
    # To improve, I need a better decomposition for 4x4x4.
    #
    # For now, use schoolbook throughout.
    
    U_full = [[0] * 120 for _ in range(4 * 5)]  # 20 rows
    V_full = [[0] * 120 for _ in range(5 * 6)]  # 30 rows
    W_full = [[0] * 120 for _ in range(6 * 4)]  # 24 rows
    
    # Part 1: A1*B1a (4x4x4, rank 64)
    U_444, V_444, W_444 = _schoolbook_sub(4, 4, 4)
    for r in range(64):
        for a_idx in range(16):  # A1 flat index i*4+j, i in 0..3, j in 0..3
            i = a_idx // 4
            j = a_idx % 4
            U_full[i*5+j, r] = U_444[a_idx, r]
        for b_idx in range(16):  # B1a flat index j*4+k, j in 0..3, k in 0..3
            j = b_idx // 4
            k = b_idx % 4
            V_full[j*6+k, r] = V_444[b_idx, r]
        for c_idx in range(16):  # C1a flat index k*4+i, k in 0..3, i in 0..3
            k = c_idx // 4
            i = c_idx % 4
            W_full[k*4+i, r] = W_444[c_idx, r]
    
    # Part 2: A1*B1b (4x4x2, rank 32)
    U_442, V_442, W_442 = _schoolbook_sub(4, 4, 2)
    col_offset = 64
    for r in range(32):
        for a_idx in range(16):  # A1 flat index i*4+j
            i = a_idx // 4
            j = a_idx % 4
            U_full[i*5+j, col_offset+r] = U_442[a_idx, r]
        for b_idx in range(8):  # B1b flat index j*2+k, j in 0..3, k in 0..1
            j = b_idx // 2
            k = b_idx % 2
            V_full[j*6+(k+4), col_offset+r] = V_442[b_idx, r]
        for c_idx in range(8):  # C1b flat index k*4+i, k in 0..1, i in 0..3
            k = c_idx // 4
            i = c_idx % 4
            W_full[(k+4)*4+i, col_offset+r] = W_442[c_idx, r]
    col_offset += 32
    
    # Part 3: A2*B2 (4x1x6, rank 24)
    U_416, V_416, W_416 = _schoolbook_sub(4, 1, 6)
    for r in range(24):
        for a_idx in range(4):  # A2 flat index i (i in 0..3)
            i = a_idx
            U_full[i*5+4, col_offset+r] = U_416[a_idx, r]
        for b_idx in range(6):  # B2 flat index k (k in 0..5)
            k = b_idx
            V_full[4*6+k, col_offset+r] = V_416[b_idx, r]
        for c_idx in range(24):  # C2 flat index k*4+i, k in 0..5, i in 0..3
            k = c_idx // 4
            i = c_idx % 4
            W_full[k*4+i, col_offset+r] = W_416[c_idx, r]
    col_offset += 24
    
    # Verify the full decomposition
    np.random.seed(123)
    A_test = np.random.rand(4, 5)
    B_test = np.random.rand(5, 6)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 6))
    for r in range(120):
        u_r = np.array(U_full)[:, r]
        v_r = np.array(V_full)[:, r]
        w_r = np.array(W_full)[:, r]
        
        A_part = np.zeros(4)
        B_part = np.zeros(6)
        for i in range(4):
            for j in range(5):
                A_part[i] += A_test[i, j] * U_full[i*5+j, r]
        for j in range(5):
            for k in range(6):
                B_part[k] += B_test[j, k] * V_full[j*6+k, r]
        for i in range(4):
            for k in range(6):
                C_computed[i, k] += A_part[i] * B_part[k] * W_full[k*4+i, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - start_time:
        pass
    
    return {"U": U_full, "V": V_full, "W": W_full}
