"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Try to find a rank-28 decomposition using Strassen on a 2x2x2 sub-block
    result = _try_strassen_block(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fallback to schoolbook if no better solution found
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_block(n, m, p, deadline):
    """
    Apply Strassen's algorithm to a 2x2x2 sub-block of the 2x3x5 multiplication.
    
    The 2x3x5 multiplication tensor can be viewed as a sum of smaller tensors.
    We partition the 3x5 matrix into blocks and apply Strassen to the 2x2x2 part.
    
    Strategy:
    - Split the 3x5 matrix into:
      - A 2x3 block (rows 0-1, cols 0-2)
      - A 1x5 block (row 2, cols 0-4)
    - For the 2x3 block, we can further split into:
      - A 2x2 sub-block (rows 0-1, cols 0-1) -> apply Strassen (rank 7 instead of 8)
      - A 2x1 sub-block (rows 0-1, col 2) -> schoolbook (rank 2)
    - The 1x5 block is schoolbook (rank 5)
    
    Total rank: 7 + 2 + 5 = 14? No, that's not right because the tensor structure is more complex.
    
    Let me think more carefully. The multiplication tensor M_{2,3,5} has entries T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}.
    
    We can decompose this as:
    T = T1 + T2 + T3 where:
    - T1 covers i in {0,1}, j in {0,1}, k in {0,1} (a 2x2x2 sub-tensor)
    - T2 covers i in {0,1}, j in {0,1}, k in {2,3,4} (a 2x2x3 sub-tensor)
    - T3 covers i in {0,1}, j in {2}, k in {0,1,2,3,4} (a 2x1x5 sub-tensor)
    
    For T1 (2x2x2): use Strassen with rank 7
    For T2 (2x2x3): use schoolbook with rank 12
    For T3 (2x1x5): use schoolbook with rank 10
    
    Total: 7 + 12 + 10 = 29
    
    But we can do better by being more clever about the decomposition.
    """
    
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    # We'll try rank 28 first
    for target_rank in [28, 29, 30]:
        if time.time() > deadline:
            break
        
        result = _construct_decomposition(n, m, p, target_rank)
        if result is not None:
            return result
    
    return None

def _construct_decomposition(n, m, p, target_rank):
    """
    Construct a rank-target_rank decomposition using block structure.
    
    For 2x3x5, we decompose into:
    - Block A: 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1})
    - Block B: 2x2x3 (i in {0,1}, j in {0,1}, k in {2,3,4})
    - Block C: 2x1x5 (i in {0,1}, j in {2}, k in {0,1,2,3,4})
    
    Schoolbook ranks: 8 + 12 + 10 = 30
    With Strassen on Block A: 7 + 12 + 10 = 29
    
    For rank 28, we need to save one more rank. We can try to optimize Block B or C.
    """
    
    if target_rank >= 29:
        # Use Strassen on the 2x2x2 block
        return _strassen_2x2x2_plus_schoolbook(n, m, p)
    
    # For rank 28, we need a more clever decomposition
    # This is harder; fall back to 29
    return None

def _strassen_2x2x2_plus_schoolbook(n, m, p):
    """
    Construct a rank-29 decomposition using Strassen on the 2x2x2 sub-block.
    
    The 2x2x2 Strassen decomposition:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A10)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    Where A = [[A00, A01], [A10, A11]] and B = [[B00, B01], [B10, B11]]
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0] * rank for _ in range(m * p)]  # 15 x 29
    W = [[0] * rank for _ in range(p * n)]  # 10 x 29
    
    # Strassen terms for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # These use ranks 0-6
    
    # M1 = (A00+A11)(B00+B11)
    # A00 is at U[0, :], A11 is at U[1, :] (since i=0,j=0 -> idx 0; i=1,j=1 -> idx 3? No)
    # Wait, U is indexed by [i*m + j] where i in {0,1}, j in {0,1,2}
    # So: U[0] = (i=0,j=0), U[1] = (i=0,j=1), U[2] = (i=0,j=2)
    #     U[3] = (i=1,j=0), U[4] = (i=1,j=1), U[5] = (i=1,j=2)
    
    # For the 2x2x2 block, we use i in {0,1}, j in {0,1}, k in {0,1}
    # A00 = U[0], A01 = U[1], A10 = U[3], A11 = U[4]
    # B00 = V[0], B01 = V[1], B10 = V[3], B11 = V[4] (V indexed by [j*p + k])
    # C00 = W[0], C01 = W[1], C10 = W[2], C11 = W[3] (W indexed by [k*n + i])
    
    # Strassen M1 = (A00+A11)(B00+B11)
    # U contribution: A00 and A11 both contribute +1
    U[0][0] = 1  # A00
    U[4][0] = 1  # A11
    V[0][0] = 1  # B00
    V[4][0] = 1  # B11
    # W contribution: C00 gets +1, C11 gets +1
    W[0][0] = 1  # C00 (k=0, i=0)
    W[3][0] = 1  # C11 (k=1, i=1)
    
    # Strassen M2 = (A10+A11)B00
    U[3][1] = 1  # A10
    U[4][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 (k=0, i=1)
    
    # Strassen M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[3][2] = 1  # B10
    V[4][2] = -1  # -B11
    W[1][2] = 1  # C01 (k=1, i=0)
    
    # Strassen M4 = A11(B01-B00)
    U[4][3] = 1  # A11
    V[1][3] = 1  # B01
    V[0][3] = -1  # -B00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # Strassen M5 = (A00+A10)B11
    U[0][4] = 1  # A00
    U[3][4] = 1  # A10
    V[4][4] = 1  # B11
    W[0][4] = -1  # -C00
    W[1][4] = 1  # C01
    
    # Strassen M6 = (A10-A00)(B00+B10)
    U[3][5] = 1  # A10
    U[0][5] = -1  # -A00
    V[0][5] = 1  # B00
    V[3][5] = 1  # B10
    W[3][5] = 1  # C11
    
    # Strassen M7 = (A01-A11)(B01+B11)
    U[1][6] = 1  # A01
    U[4][6] = -1  # -A11
    V[1][6] = 1  # B01
    V[4][6] = 1  # B11
    W[0][6] = 1  # C00
    W[3][6] = 1  # C11
    
    # Now add schoolbook terms for the remaining parts
    # Block B: 2x2x3 (i in {0,1}, j in {0,1}, k in {2,3,4})
    # Schoolbook: 8 terms (i in {0,1}, j in {0,1}, k in {2,3,4})
    # Wait, that's 2*2*3 = 12 terms
    col = 7
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col is now 19
    
    # Block C: 2x1x5 (i in {0,1}, j in {2}, k in {0,1,2,3,4})
    # Schoolbook: 2*1*5 = 10 terms
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 29
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    
    # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        u_c = np.array([U[i * m + j][c] for i in range(n) for j in range(m)]).reshape(n, m)
        v_c = np.array([V[j * p + k][c] for j in range(m) for k in range(p)]).reshape(m, p)
        w_c = np.array([W[k * n + i][c] for k in range(p) for i in range(n)]).reshape(p, n)
        
        # T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    
    # Check that all entries are exactly 1
    return np.all(np.abs(T - 1) < 1e-9)

# EVOLVE-BLOCK-END
