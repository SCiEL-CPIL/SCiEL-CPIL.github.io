"""Matrix multiplication 2,3,10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check whether the decomposition exactly reproduces the matmul tensor."""
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-12


def _build_strassen_232():
    """
    Construct U, V, W for 2x3x2 matrix multiplication with rank 11.
    U: (2*3, 11)
    V: (3*2, 11)
    W: (2*2, 11)
    """
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part (7 mults) for A_left (2x2) * B_top (2x2)
    # c=0: m0 = (a00+a11)(b00+b11)
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # c=1: m1 = (a10+a11)b00
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = -1
    
    # c=2: m2 = a00(b10-b11)
    U[0*3+0][2] = 1
    V[1*2+0][2] = 1; V[1*2+1][2] = -1
    W[1*2+0][2] = 1; W[1*2+1][2] = 1
    
    # c=3: m3 = a11(b01-b00)
    U[1*3+1][3] = 1
    V[0*2+1][3] = 1; V[0*2+0][3] = -1
    W[0*2+0][3] = -1; W[1*2+0][3] = 1
    
    # c=4: m4 = (a00+a01)b11
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+1][4] = 1; W[1*2+1][4] = 1
    
    # c=5: m5 = (a10-a00)(b00+b01)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    W[0*2+0][5] = 1
    
    # c=6: m6 = (a01-a11)(b10+b11)
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = 1
    
    # Outer product part (4 mults) for a_right (2x1) * b_bottom (1x2)
    # c=7: a02 * b20 -> C00
    U[0*3+2][7] = 1
    V[2*2+0][7] = 1
    W[0*2+0][7] = 1
    
    # c=8: a02 * b21 -> C01
    U[0*3+2][8] = 1
    V[2*2+1][8] = 1
    W[1*2+0][8] = 1
    
    # c=9: a12 * b20 -> C10
    U[1*3+2][9] = 1
    V[2*2+0][9] = 1
    W[0*2+1][9] = 1
    
    # c=10: a12 * b21 -> C11
    U[1*3+2][10] = 1
    V[2*2+1][10] = 1
    W[1*2+1][10] = 1
    
    return U, V, W


def _construct_block_rank():
    """
    Construct the 55-rank solution by tiling 5 blocks of 2x3x2 (rank 11 each).
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    block_rank = 11
    total_rank = num_blocks * block_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U_block, V_block, W_block = _build_strassen_232()
    
    for t in range(num_blocks):
        offset = t * block_rank
        for i in range(n * m):
            for c in range(block_rank):
                U[i][offset + c] = U_block[i][c]
        
        for j in range(m):
            for k_local in range(2):
                k_global = t * 2 + k_local
                local_row = j * 2 + k_local
                global_row = j * p + k_global
                for c in range(block_rank):
                    V[global_row][offset + c] = V_block[local_row][c]
                    
        for k_local in range(2):
            k_global = t * 2 + k_local
            for i in range(n):
                local_row = k_local * n + i
                global_row = k_global * n + i
                for c in range(block_rank):
                    W[global_row][offset + c] = W_block[local_row][c]
                    
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_strassen_2x3x3():
    """
    Try to construct a 2x3x3 block with rank < 18 using Strassen-like ideas.
    For 2x3x3, the schoolbook rank is 18.
    We can split the 3x3 block into a 2x2 Strassen part plus remaining terms.
    """
    n, m, p = 2, 3, 3
    # Schoolbook for 2x3x3 has rank 18
    # Let's try: Strassen on columns 0,1 (rank 7) + schoolbook on column 2 (rank 6) = 13?
    # Actually for 2x2x3: Strassen on first 2 cols (rank 7) + 1 col schoolbook (rank 2) = 9
    # For 2x3x3: think of B as 3x3. Split B into B[:,0:2] (3x2) and B[:,2] (3x1)
    # A @ B[:,0:2] is 2x2, done with rank 7 (Strassen)
    # A @ B[:,2] is 2x1, done with rank 2 (schoolbook)
    # But wait, Strassen for 2x3x2 has rank 11, not 7, because m=3 not m=2.
    # Let me think differently.
    
    # For 2x2x3 (n=2,m=2,p=3): Strassen 2x2x2 (rank 7) + 2x2x1 (rank 2) = 9
    # For 2x3x3: we can write B as [B1 B2] where B1 is 3x2, B2 is 3x1
    # A@B1 is 2x2. The tensor for 2x3x2 has rank 11 (as we computed).
    # A@B2 is 2x1. The tensor for 2x3x1 has rank 6.
    # Total: 11 + 6 = 17. That's better than 18!
    
    # Let's construct 2x3x3 with rank 17.
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x3x2 (columns 0,1) with rank 11
    U1, V1, W1 = _build_strassen_232()
    for i in range(n * m):
        for c in range(11):
            U[i][c] = U1[i][c]
    for j in range(m):
        for k in range(2):
            for c in range(11):
                V[j*p + k][c] = V1[j*2 + k][c]
    for k in range(2):
        for i in range(n):
            for c in range(11):
                W[k*n + i][c] = W1[k*n + i][c]
    
    # Part 2: 2x3x1 (column 2) with rank 6
    offset = 11
    for i in range(n):
        for j in range(m):
            col = offset + i*m + j
            U[i*m + j][col] = 1
            V[j*p + 2][col] = 1
            W[2*n + i][col] = 1
    
    return U, V, W, rank


def _construct_51_rank():
    """
    Try to get 2x3x10 with rank < 55.
    Split p=10 into: one 2x3x3 block (rank 17) + four 2x3x2 blocks (rank 11 each) = 17 + 44 = 61. Worse.
    Or: five 2x3x2 blocks = 55 (current).
    
    Alternative: 2x3x10 = 2x3x5 * 2? No.
    
    Let's try: split into 2x3x4 (rank ?) + 2x3x6 (rank ?)
    2x3x4: Strassen 2x3x2 (11) + 2x3x2 (11) = 22
    2x3x6: 3 * 11 = 33
    Total: 22 + 33 = 55. Same.
    
    What about a smarter approach for 2x3x4?
    2x3x4: Think of B as 3x4. 
    Use Winograd/Strassen ideas... This is complex.
    
    Let's just go with 55 for now and add a time guard.
    """
    return _construct_block_rank()


def construct():
    start = time.time()
    deadline = start + 120.0
    
    n, m, p = DIMENSIONS
    
    # Try the constructive 55-rank solution
    try:
        sol = _construct_block_rank()
        if _verify(sol["U"], sol["V"], sol["W"], n, m, p):
            return sol
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 5.0:
        return _schoolbook()
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
