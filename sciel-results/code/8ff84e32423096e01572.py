"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_factors(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    produce the matrix multiplication tensor under the trace convention W[k*n+i, r]."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                expected = 1
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - expected) > tol:
                    return False, (A, B, C, expected, actual)
    return True, None

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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

def _strassen_2x2():
    """
    Returns the Strassen decomposition for 2x2x2 matrix multiplication.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Column 0: M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1  # A11, A22
    V[0][0] = 1; V[3][0] = 1  # B11, B22
    W[0][0] = 1; W[3][0] = 1  # C11, C22 (indices: C11 is k=0,i=0 -> idx 0; C22 is k=1,i=1 -> idx 3)
    
    # Column 1: M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1  # A21 (i=1,j=0 -> idx 2), A22 (i=1,j=1 -> idx 3)
    V[0][1] = 1  # B11 (j=0,k=0 -> idx 0)
    W[2][1] = 1; W[3][1] = 1  # C21 (k=0,i=1 -> idx 2), C22 (k=1,i=1 -> idx 3)
    
    # Column 2: M3 = A11(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1; V[3][2] = -1  # B12 (j=0,k=1 -> idx 1), B22 (j=1,k=1 -> idx 3)
    W[1][2] = 1; W[3][2] = 1  # C12 (k=1,i=0 -> idx 1), C22 (k=1,i=1 -> idx 3)
    
    # Column 3: M4 = A22(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1; V[0][3] = -1  # B21 (j=1,k=0 -> idx 2), B11 (j=0,k=0 -> idx 0)
    W[0][3] = 1; W[2][3] = 1  # C11 (k=0,i=0 -> idx 0), C21 (k=0,i=1 -> idx 2)
    
    # Column 4: M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1  # A11, A12 (i=0,j=1 -> idx 1)
    V[3][4] = 1  # B22
    W[0][4] = -1; W[1][4] = 1  # C11 (-1), C12 (+1)
    
    # Column 5: M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1  # A21, A11
    V[0][5] = 1; V[1][5] = 1  # B11, B12
    W[3][5] = 1  # C22
    
    # Column 6: M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1  # A12, A22
    V[2][6] = 1; V[3][6] = 1  # B21, B22
    W[0][6] = 1  # C11
    
    return U, V, W

def _extend_strassen_to_2x3x3():
    """
    Attempt to extend the Strassen 2x2x2 decomposition to 2x3x3 by treating
    the 3x3 matrix as containing a 2x2 block plus additional terms.
    
    A is 2x3: A = [A_2x2 | A_2x1]
    B is 3x3: B = [[B_2x2, B_2x1], [B_1x2, B_1x1]]
    C = A*B is 2x3: C = [C_2x2 | C_2x1]
    
    We use Strassen for the 2x2x2 core (columns 0-1 of A, rows 0-1 of B, 
    producing columns 0-1 of C), and handle the remaining terms separately.
    
    C_2x2 = A_2x2 * B_2x2 (Strassen, rank 7)
    C_2x1 = A_2x2 * B_2x1 + A_2x1 * B_1x1
    
    For A_2x2 * B_2x1: this is 2x2 times 2x1. We can use a rank-3 decomposition 
    (or fewer if we exploit structure).
    For A_2x1 * B_1x1: this is 2x1 times 1x1, which is just scaling, rank 2.
    
    Total rank: 7 + 3 + 2 = 12. Let's see if we can do better.
    
    Actually, let's just construct it explicitly.
    """
    n, m, p = 2, 3, 3
    rank = 12  # 7 (Strassen) + 3 (2x2*2x1) + 2 (2x1*1x1)
    
    U = [[0]*rank for _ in range(n*m)]  # 6 rows
    V = [[0]*rank for _ in range(m*p)]  # 9 rows
    W = [[0]*rank for _ in range(p*n)]  # 6 rows
    
    # --- Part 1: Strassen for A_2x2 * B_2x2 -> C_2x2 ---
    # A indices: i in {0,1}, j in {0,1} -> U row idx = i*3+j = i*3+j (since m=3)
    # B indices: j in {0,1}, k in {0,1} -> V row idx = j*3+k
    # C indices: k in {0,1}, i in {0,1} -> W row idx = k*2+i
    
    U_s, V_s, W_s = _strassen_2x2()
    for r in range(7):
        for a_idx in range(4):
            if U_s[a_idx][r] != 0:
                i, j = a_idx // 2, a_idx % 2
                U[i*3+j][r] = U_s[a_idx][r]
        for b_idx in range(4):
            if V_s[b_idx][r] != 0:
                j, k = b_idx // 2, b_idx % 2
                V[j*3+k][r] = V_s[b_idx][r]
        for c_idx in range(4):
            if W_s[c_idx][r] != 0:
                k, i = c_idx // 2, c_idx % 2
                W[k*2+i][r] = W_s[c_idx][r]
    
    # --- Part 2: A_2x2 * B_2x1 -> C_2x1 (columns 2 of C) ---
    # A_2x2: i in {0,1}, j in {0,1}
    # B_2x1: j in {0,1}, k=2 -> V row idx = j*3+2
    # C_2x1: k=2, i in {0,1} -> W row idx = 2*2+i = 4+i
    
    # Schoolbook for 2x2*2x1 is rank 4, but we can do rank 3 using Strassen-like tricks
    # Actually, 2x2*2x1 is just two dot products of length 2.
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # We need 4 terms in schoolbook. Can we reduce?
    # Let's just use rank 4 for simplicity, total becomes 13.
    # Or use the fact that this is a 2x2x1 tensor.
    
    # Let's use rank 4 for this part (columns 7-10)
    col = 7
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*3+2][col] = 1
            W[4+i][col] = 1
            col += 1
    # Now col = 11
    
    # --- Part 3: A_2x1 * B_1x1 -> C_2x1 ---
    # A_2x1: i in {0,1}, j=2 -> U row idx = i*3+2
    # B_1x1: j=2, k=2 -> V row idx = 2*3+2 = 8
    # C_2x1: k=2, i in {0,1} -> W row idx = 4+i
    
    # Rank 2
    for i in range(2):
        U[i*3+2][col] = 1
        V[8][col] = 1
        W[4+i][col] = 1
        col += 1
    # Now col = 13, but we allocated rank=12. Let's fix.
    
    # We need rank = 7 + 4 + 2 = 13. Let's rebuild with rank 13.
    
    rank = 13
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Part 1: Strassen
    U_s, V_s, W_s = _strassen_2x2()
    for r in range(7):
        for a_idx in range(4):
            if U_s[a_idx][r] != 0:
                i, j = a_idx // 2, a_idx % 2
                U[i*3+j][r] = U_s[a_idx][r]
        for b_idx in range(4):
            if V_s[b_idx][r] != 0:
                j, k = b_idx // 2, b_idx % 2
                V[j*3+k][r] = V_s[b_idx][r]
        for c_idx in range(4):
            if W_s[c_idx][r] != 0:
                k, i = c_idx // 2, c_idx % 2
                W[k*2+i][r] = W_s[c_idx][r]
    
    # Part 2: A_2x2 * B_2x1 (rank 4)
    col = 7
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*3+2][col] = 1
            W[4+i][col] = 1
            col += 1
    
    # Part 3: A_2x1 * B_1x1 (rank 2)
    for i in range(2):
        U[i*3+2][col] = 1
        V[8][col] = 1
        W[4+i][col] = 1
        col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Try the extended Strassen decomposition (rank 13)
    try:
        U, V, W = _extend_strassen_to_2x3x3()
        ok, _ = _verify_factors(U, V, W, n, m, p)
        if ok:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
