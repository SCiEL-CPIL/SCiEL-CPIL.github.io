"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2 * 2x2.
    Returns U, V, W each of shape (4, 7) corresponding to:
    - U: 2x2 matrix A indices -> U[i*2+j, t]
    - V: 2x2 matrix B indices -> V[j*2+k, t]
    - W: 2x2 output C indices -> W[k*2+i, t]
    
    Strassen's 7 products:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 = index (0,0) -> U[0], A22 = index (1,1) -> U[3]
    # B11 = index (0,0) -> V[0], B22 = index (1,1) -> V[3]
    # t=0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C11 contribution: +1
    W[1][0] = 1  # C12: +1 (wait, M1 contributes to C11 and C22)
    # Let me redo this carefully.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # W[k*n+i, t] where output is C[i,k], so W index is k*2+i
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    
    # Reset
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # t=0: M1 = (A11 + A22)(B11 + B22)
    # A part: A[0,0] + A[1,1] -> U[0,0]=1, U[1*2+1,0]=U[3,0]=1
    U[0][0] = 1
    U[3][0] = 1
    # B part: B[0,0] + B[1,1] -> V[0*2+0,0]=V[0,0]=1, V[1*2+1,0]=V[3,0]=1
    V[0][0] = 1
    V[3][0] = 1
    # W: M1 contributes +1 to C11 (W[0,0]) and +1 to C22 (W[3,0])
    W[0][0] = 1
    W[3][0] = 1
    
    # t=1: M2 = (A21 + A22)B11
    # A part: A[1,0] + A[1,1] -> U[1*2+0,1]=U[2,1]=1, U[1*2+1,1]=U[3,1]=1
    U[2][1] = 1
    U[3][1] = 1
    # B part: B[0,0] -> V[0,1]=1
    V[0][1] = 1
    # W: M2 contributes +1 to C21 (W[1,1]) and -1 to C22 (W[3,1])
    W[1][1] = 1
    W[3][1] = -1
    
    # t=2: M3 = A11(B12 - B22)
    # A part: A[0,0] -> U[0,2]=1
    U[0][2] = 1
    # B part: B[0,1] - B[1,1] -> V[0*2+1,2]=V[1,2]=1, V[1*2+1,2]=V[3,2]=-1
    V[1][2] = 1
    V[3][2] = -1
    # W: M3 contributes +1 to C12 (W[2,2]) and +1 to C22 (W[3,2])
    W[2][2] = 1
    W[3][2] = 1
    
    # t=3: M4 = A22(B21 - B11)
    # A part: A[1,1] -> U[3,3]=1
    U[3][3] = 1
    # B part: B[1,0] - B[0,0] -> V[1*2+0,3]=V[2,3]=1, V[0*2+0,3]=V[0,3]=-1
    V[2][3] = 1
    V[0][3] = -1
    # W: M4 contributes +1 to C11 (W[0,3]) and +1 to C21 (W[1,3])
    W[0][3] = 1
    W[1][3] = 1
    
    # t=4: M5 = (A11 + A12)B22
    # A part: A[0,0] + A[0,1] -> U[0,4]=1, U[1,4]=1
    U[0][4] = 1
    U[1][4] = 1
    # B part: B[1,1] -> V[3,4]=1
    V[3][4] = 1
    # W: M5 contributes -1 to C11 (W[0,4]) and +1 to C12 (W[2,4])
    W[0][4] = -1
    W[2][4] = 1
    
    # t=5: M6 = (A21 - A11)(B11 + B12)
    # A part: A[1,0] - A[0,0] -> U[2,5]=1, U[0,5]=-1
    U[2][5] = 1
    U[0][5] = -1
    # B part: B[0,0] + B[0,1] -> V[0,5]=1, V[1,5]=1
    V[0][5] = 1
    V[1][5] = 1
    # W: M6 contributes +1 to C22 (W[3,5])
    W[3][5] = 1
    
    # t=6: M7 = (A12 - A22)(B21 + B22)
    # A part: A[0,1] - A[1,1] -> U[1,6]=1, U[3,6]=-1
    U[1][6] = 1
    U[3][6] = -1
    # B part: B[1,0] + B[1,1] -> V[2,6]=1, V[3,6]=1
    V[2][6] = 1
    V[3][6] = 1
    # W: M7 contributes +1 to C11 (W[0,6])
    W[0][6] = 1
    
    return U, V, W


def _schoolbook_2x2x2():
    """Schoolbook rank-4 decomposition for 2x2 * 2x2."""
    n = m = p = 2
    rank = 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W


def _build_4x4x7_strassen():
    """
    Build 4x4x7 decomposition using Strassen on 2x2 blocks.
    
    A (4x4) partitioned into 2x2 blocks:
        A = [[A11, A12],
             [A21, A22]]
    each block is 2x2.
    
    B (4x7) partitioned into 2x7 blocks:
        B = [[B1],   (2x7)
             [B2]]   (2x7)
    
    C (4x7) = [[C1],   (2x7)
               [C2]]   (2x7)
    
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    
    Each product is 2x2 * 2x7. We can split B into two 2x3.5 blocks? No.
    Better: treat 2x2 * 2x7 as four 2x2 * 2x1 blocks, but that's inefficient.
    
    Alternative: Use Strassen for the 2x2 * 2x2 part, and handle the 7 columns by
    splitting into 2+2+2+1 or similar.
    
    Actually, let's use a different approach:
    2x2 * 2x7: Think of B as [B_left (2x3), B_right (2x4)] or [B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x1)]
    
    Simplest: Split 2x2 * 2x7 into 7 instances of 2x2 * 2x1, but 2x2*2x1 has rank 2.
    Total: 7 * 2 = 14 per block product. 4 block products: 4 * 14 = 56.
    
    But can we do better? Let's use Strassen for 2x2*2x2 and handle the last column separately.
    2x2 * 2x7 = 2x2 * [2x2 | 2x2 | 2x2 | 2x1]
    First three 2x2*2x2 use Strassen (rank 7 each) = 21
    Last 2x2*2x1: rank 2
    Total per block: 23
    4 blocks: 92. That's worse.
    
    Let me just use the direct schoolbook-style for 2x2*2x7 which has rank 14.
    Total: 4 * 14 = 56.
    """
    n, m, p = 4, 4, 7
    # Use Strassen for each 2x2 * 2x2 sub-product, and handle the structure
    
    # Actually, the cleanest approach: 
    # C = A*B where A is 4x4, B is 4x7
    # Partition A into 2x2 blocks, B into 2x7 blocks
    # C1 = A11*B1 + A12*B2 (2x7)
    # C2 = A21*B1 + A22*B2 (2x7)
    #
    # For each 2x2 * 2x7 product, use schoolbook (rank 14)
    # Total rank = 4 * 14 = 56
    
    rank = 56
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # For each (bi, bj, bk) where bi is row block, bj is col block of A, bk is row block of B
    # With the constraint that bj == bk (since A's col must match B's row)
    for bi in range(2):
        for bj in range(2):
            # bk must equal bj
            bk = bj
            # A block (bi, bj): rows [bi*2, bi*2+2), cols [bj*2, bj*2+2)
            # B block bk: rows [bk*2, bk*2+2), cols 0..6
            # C block bi: rows [bi*2, bi*2+2), cols 0..6
            for i_local in range(2):
                for j_local in range(2):
                    for k in range(p):
                        i = bi * 2 + i_local
                        j = bj * 2 + j_local
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=4, m=4, p=7):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for t in range(rank):
        a_t = 0.0
        for i in range(n):
            for j in range(m):
                a_t += U_arr[i * m + j, t] * A[i, j]
        b_t = 0.0
        for j in range(m):
            for k in range(p):
                b_t += V_arr[j * p + k, t] * B[j, k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_arr[k * n + i, t] * a_t * b_t
    
    diff = np.max(np.abs(C_computed - C_expected))
    return diff < 1e-10, diff


def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Build the 4x4x7 decomposition
    result = _build_4x4x7_strassen()
    
    # Verify
    valid, diff = _verify_decomposition(result["U"], result["V"], result["W"])
    if not valid:
        # Fall back to schoolbook
        n, m, p = DIMENSIONS
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
    
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
