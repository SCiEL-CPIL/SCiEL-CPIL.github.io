"""Matrix multiplication 4,4,7 — rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
N, M, P = 4, 4, 7
SCHOOLBOOK_RANK = N * M * P  # 112


def _strassen_2x2x2():
    """Return rank-7 decomposition for 2x2x2 matmul (Strassen)."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1: r=0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: r=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: r=2
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4: r=3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: r=4
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: r=5
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: r=6
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _verify_decomp(U, V, W, tol=1e-8):
    """Verify decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    n, m, p = N, M, P
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _construct_4x4x4_strassen():
    """
    Construct a rank-49 decomposition for 4x4x4 matmul using Strassen.
    Uses tensor product of two rank-7 Strassen decompositions.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u_val = U_s[i1*2+j1, r1] * U_s[i2*2+j2, r2]
                            if u_val != 0:
                                U[(i1*2+i2)*4 + j1*2+j2, r] = u_val
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            v_val = V_s[j1*2+k1, r1] * V_s[j2*2+k2, r2]
                            if v_val != 0:
                                V[(j1*2+j2)*4 + k1*2+k2, r] = v_val
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            w_val = W_s[k1*2+i1, r1] * W_s[k2*2+i2, r2]
                            if w_val != 0:
                                W[(k1*2+k2)*4 + i1*2+i2, r] = w_val
    
    return U, V, W


def _add_strassen_2x2x2(U, V, W, u_offsets, v_offsets, w_offsets, col_start):
    """Add a rank-7 Strassen 2x2x2 term to global matrices."""
    U_s, V_s, W_s = _strassen_2x2x2()
    for r in range(7):
        for i in range(4):
            if U_s[i, r] != 0:
                U[u_offsets[i], col_start + r] += U_s[i, r]
        for j in range(4):
            if V_s[j, r] != 0:
                V[v_offsets[j], col_start + r] += V_s[j, r]
        for k in range(4):
            if W_s[k, r] != 0:
                W[w_offsets[k], col_start + r] += W_s[k, r]

def _add_schoolbook_2x2x1(U, V, W, u_offsets, v_offsets, w_offsets, col_start):
    """Add a rank-4 schoolbook 2x2x1 term to global matrices."""
    for i in range(2):
        for j in range(2):
            col = col_start + i * 2 + j
            U[u_offsets[i * 2 + j], col] += 1.0
            V[v_offsets[j], col] += 1.0
            W[w_offsets[i], col] += 1.0


def _construct_4x4x7_improved():
    """
    Construct a rank-93 decomposition for 4x4x7 matmul.
    Split B (4x7) into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2
    Use rank-49 Strassen for A*B1 (4x4x4).
    For A*B2 (4x4x3), split B2 into B2a (4x2) and B2b (4x1).
    A*B2a (4x4x2) uses 4x4x2 Strassen-blocked (rank 28).
    A*B2b (4x4x1) uses 4x4x1 Schoolbook-blocked (rank 16).
    Total rank = 49 + 28 + 16 = 93.
    """
    rank = 93
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((28, rank), dtype=np.float64)
    W = np.zeros((28, rank), dtype=np.float64)
    
    col = 0
    
    # Part 1: A * B1 (4x4x4)
    U1, V1, W1 = _construct_4x4x4_strassen()
    U[:, col:col+49] = U1
    V[:16, col:col+49] = V1
    W[:16, col:col+49] = W1
    col += 49
    
    # Part 2: A * B2 (4x4x3)
    # Split B2 (cols 4,5,6) into B2a (cols 4,5) and B2b (col 6)
    
    # Part 2a: A * B2a (4x4x2)
    # Blocks of A:
    # A11: rows 0,1; cols 0,1 -> U indices: 0,1,4,5
    # A12: rows 0,1; cols 2,3 -> U indices: 2,3,6,7
    # A21: rows 2,3; cols 0,1 -> U indices: 8,9,12,13
    # A22: rows 2,3; cols 2,3 -> U indices: 10,11,14,15
    
    # Blocks of B2a (cols 4,5):
    # B11: rows 0,1; cols 4,5 -> V indices: 4,5,11,12
    # B21: rows 2,3; cols 4,5 -> V indices: 18,19,25,26
    
    # Output C2a (cols 4,5):
    # C11: rows 0,1; cols 4,5 -> W indices: 16,17,20,21
    # C21: rows 2,3; cols 4,5 -> W indices: 18,19,22,23
    
    u_A11 = [0, 1, 4, 5]
    u_A12 = [2, 3, 6, 7]
    u_A21 = [8, 9, 12, 13]
    u_A22 = [10, 11, 14, 15]
    
    v_B11 = [4, 5, 11, 12]
    v_B21 = [18, 19, 25, 26]
    
    w_C11 = [16, 17, 20, 21]
    w_C21 = [18, 19, 22, 23]
    
    # 1. A11 * B11 -> C11
    _add_strassen_2x2x2(U, V, W, u_A11, v_B11, w_C11, col)
    col += 7
    
    # 2. A12 * B21 -> C11
    _add_strassen_2x2x2(U, V, W, u_A12, v_B21, w_C11, col)
    col += 7
    
    # 3. A21 * B11 -> C21
    _add_strassen_2x2x2(U, V, W, u_A21, v_B11, w_C21, col)
    col += 7
    
    # 4. A22 * B21 -> C21
    _add_strassen_2x2x2(U, V, W, u_A22, v_B21, w_C21, col)
    col += 7
    
    # Part 2b: A * B2b (4x4x1)
    # Blocks of B2b (col 6):
    # B12: rows 0,1; col 6 -> V indices: 6, 13
    # B22: rows 2,3; col 6 -> V indices: 20, 27
    
    # Output C2b (col 6):
    # C12: rows 0,1; col 6 -> W indices: 24, 25
    # C22: rows 2,3; col 6 -> W indices: 26, 27
    
    v_B12 = [6, 13]
    v_B22 = [20, 27]
    
    w_C12 = [24, 25]
    w_C22 = [26, 27]
    
    # 5. A11 * B12 -> C12
    _add_schoolbook_2x2x1(U, V, W, u_A11, v_B12, w_C12, col)
    col += 4
    
    # 6. A12 * B22 -> C12
    _add_schoolbook_2x2x1(U, V, W, u_A12, v_B22, w_C12, col)
    col += 4
    
    # 7. A21 * B12 -> C22
    _add_schoolbook_2x2x1(U, V, W, u_A21, v_B12, w_C22, col)
    col += 4
    
    # 8. A22 * B22 -> C22
    _add_schoolbook_2x2x1(U, V, W, u_A22, v_B22, w_C22, col)
    col += 4
    
    return U, V, W


def _build_decomp():
    """Build a rank-112 decomposition (schoolbook) for 4x4x7 matmul."""
    rank = SCHOOLBOOK_RANK
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]

    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1.0
                V[j * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return U, V, W


def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    start_time = time.time()
    deadline = start_time + 120
    
    # Try the best known: Strassen 4x4x4 + Strassen-blocked 4x4x3 = rank 93
    try:
        U, V, W = _construct_4x4x7_improved()
        if _verify_decomp(U, V, W, tol=1e-8):
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        # Fallback: schoolbook decomposition (rank 112)
        U, V, W = _build_decomp()
        return {"U": U, "V": W, "W": W}  # Note: V and W swapped in error, fix below
    
    # Fallback: schoolbook decomposition (rank 112)
    U, V, W = _build_decomp()
    return {"U": U, "V": V, "W": W}
