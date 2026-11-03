"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
from collections import defaultdict

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_integer_halfinteger(U, V, W):
    """Check all entries are integer or half-integer (multiples of 0.5)."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) where:
    - U is 4x7 (rows: A[0,0], A[0,1], A[1,0], A[1,1])
    - V is 4x7 (rows: B[0,0], B[0,1], B[1,0], B[1,1])
    - W is 4x7 (rows: C[0,0], C[0,1], C[1,0], C[1,1])
    
    Strassen formulas:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U: rows are A entries, columns are M1..M7
    # A00=row0, A01=row1, A10=row2, A11=row3
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    # V: rows are B entries, columns are M1..M7
    # B00=row0, B01=row1, B10=row2, B11=row3
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [1, 0, 1, 0, 0, 1, 0],   # B01
        [1, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ]
    # W: rows are C entries, columns are M1..M7
    # C00=row0, C01=row1, C10=row2, C11=row3
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],    # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],    # C10 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0],   # C11 = M1 - M2 + M3 + M6
    ]
    return U, V, W

def _strassen_2x2_2x3_decomposition():
    """
    Return a decomposition for 2x2 * 2x3 -> 2x3 using Strassen-like approach.
    Split B into B_left (2x2) and B_right (2x1).
    
    For the 2x2 part, use Strassen (rank 7).
    For the 2x1 part, we need to compute:
    C_left_col = A * B_col where A is 2x2, B_col is 2x1
    
    Actually, let's think differently. We can use the full Strassen approach
    but with B being 2x3. Let's split B into two 2x2 blocks? No, 2x3 doesn't split evenly.
    
    Alternative: Use the fact that 2x2 * 2x3 can be done with rank 11 or so.
    Let me try a direct approach with some clever combinations.
    
    Actually, for 2x2 * 2x3, we can think of it as computing 6 output entries.
    The schoolbook rank would be 4*3 = 12. Let's see if we can do better.
    
    Let me try to use the Strassen idea but adapted. Split A into 2x2 (itself) and
    B into columns. 
    
    For now, let me just use schoolbook for the 2x2*2x3 case with rank 12,
    but try to find a better decomposition.
    
    Actually, let me try a different approach: use the known fact that
    R(2x2 * 2x3) <= 11 or similar. Let me try to construct one.
    
    For simplicity in this first pass, let me use rank 12 (schoolbook) for 2x2*2x3
    and see if the overall construction works. Then I can optimize.
    """
    # For now, use schoolbook: rank = 2*2*3 = 12
    # U: 4x12 (A entries)
    # V: 6x12 (B entries: B[0,0],B[0,1],B[0,2],B[1,0],B[1,1],B[1,2])
    # W: 6x12 (C entries: C[0,0],C[0,1],C[0,2],C[1,0],C[1,1],C[1,2])
    
    U = [[0]*12 for _ in range(4)]
    V = [[0]*12 for _ in range(6)]
    W = [[0]*12 for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_4x4_4x5():
    """
    Build a Strassen-based decomposition for 4x4 * 4x5 -> 4x5.
    
    Split A (4x4) into 2x2 blocks:
    A = [[A11, A12],
         [A21, A22]]
    where each block is 2x2.
    
    Split B (4x5) into blocks:
    B = [[B11, B12],
         [B21, B22]]
    where B11, B21 are 2x2 and B12, B22 are 2x3.
    
    C = [[C11, C12],
         [C21, C22]]
    where C11, C21 are 2x2 and C12, C22 are 2x3.
    
    C11 = A11*B11 + A12*B21  (2x2*2x2 + 2x2*2x2)
    C12 = A11*B12 + A12*B22  (2x2*2x3 + 2x2*2x3)
    C21 = A21*B11 + A22*B21  (2x2*2x2 + 2x2*2x2)
    C22 = A21*B12 + A22*B22  (2x2*2x3 + 2x2*2x3)
    
    For each product, we can use Strassen for 2x2*2x2 (rank 7) 
    and a decomposition for 2x2*2x3.
    
    For 2x2*2x3, let me try to find a rank-11 or rank-12 decomposition.
    For now, let me use rank 12 (schoolbook) for 2x2*2x3.
    
    Total rank:
    C11: 7 + 7 = 14
    C12: 12 + 12 = 24
    C21: 7 + 7 = 14
    C22: 12 + 12 = 24
    Total: 14 + 24 + 14 + 24 = 76
    
    That's worse than schoolbook (80)? No, 76 < 80. So it's an improvement.
    
    But wait, we can do better. Let me try to find a better 2x2*2x3 decomposition.
    
    Actually, let me try a different approach. Instead of separate Strassen for each block,
    let me try to use a more integrated approach.
    
    For now, let me construct the rank-76 decomposition and verify it.
    """
    n, m, p = 4, 4, 5
    
    # Get the Strassen 2x2*2x2 decomposition
    S_U, S_V, S_W = _strassen_2x2_decomposition()
    
    # Get the 2x2*2x3 decomposition (schoolbook for now, rank 12)
    T_U, T_V, T_W = _strassen_2x2_2x3_decomposition()
    
    # Total rank = 14 + 24 + 14 + 24 = 76
    total_rank = 76
    
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 76
    V = [[0]*total_rank for _ in range(m*p)]  # 20 x 76
    W = [[0]*total_rank for _ in range(p*n)]  # 20 x 76
    
    # Column offsets for each block product
    # C11 = A11*B11 + A12*B21
    #   A11*B11: columns 0-6 (Strassen)
    #   A12*B21: columns 7-13 (Strassen)
    # C12 = A11*B12 + A12*B22
    #   A11*B12: columns 14-25 (schoolbook 2x2*2x3)
    #   A12*B22: columns 26-37 (schoolbook 2x2*2x3)
    # C21 = A21*B11 + A22*B21
    #   A21*B11: columns 38-44 (Strassen)
    #   A22*B21: columns 45-51 (Strassen)
    # C22 = A21*B12 + A22*B22
    #   A21*B12: columns 52-63 (schoolbook 2x2*2x3)
    #   A22*B22: columns 64-75 (schoolbook 2x2*2x3)
    
    # Define block mappings
    # A block row indices in the full U matrix (16 rows: A[i,j] -> row i*4+j)
    # A11: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # A12: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # A21: A[2,0]=8, A[2,1]=9, A[3,0]=12, A[3,1]=13
    # A22: A[2,2]=10, A[2,3]=11, A[3,2]=14, A[3,3]=15
    
    A11_rows = [0, 1, 4, 5]
    A12_rows = [2, 3, 6, 7]
    A21_rows = [8, 9, 12, 13]
    A22_rows = [10, 11, 14, 15]
    
    # B block row indices in the full V matrix (20 rows: B[j,k] -> row j*5+k)
    # B11: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    # B12: B[0,2]=2, B[0,3]=3, B[0,4]=4, B[1,2]=7, B[1,3]=8, B[1,4]=9
    # B21: B[2,0]=10, B[2,1]=11, B[3,0]=15, B[3,1]=16
    # B22: B[2,2]=12, B[2,3]=13, B[2,4]=14, B[3,2]=17, B[3,3]=18, B[3,4]=19
    
    B11_rows = [0, 1, 5, 6]
    B12_rows = [2, 3, 4, 7, 8, 9]
    B21_rows = [10, 11, 15, 16]
    B22_rows = [12, 13, 14, 17, 18, 19]
    
    # C block row indices in the full W matrix (20 rows: C[i,k] -> row k*4+i)
    # C11: C[0,0]=0, C[0,1]=4, C[1,0]=1, C[1,1]=5
    # C12: C[0,2]=8, C[0,3]=12, C[0,4]=16, C[1,2]=9, C[1,3]=13, C[1,4]=17
    # C21: C[2,0]=2, C[2,1]=6, C[3,0]=3, C[3,1]=7
    # C22: C[2,2]=10, C[2,3]=14, C[2,4]=18, C[3,2]=11, C[3,3]=15, C[3,4]=19
    
    C11_rows = [0, 4, 1, 5]
    C12_rows = [8, 12, 16, 9, 13, 17]
    C21_rows = [2, 6, 3, 7]
    C22_rows = [10, 14, 18, 11, 15, 19]
    
    def _add_strassen_block(A_rows, B_rows, C_rows, col_offset):
        """Add a Strassen 2x2*2x2 block at the given column offset."""
        for local_i in range(4):
            for c in range(7):
                U[A_rows[local_i]][col_offset + c] += S_U[local_i][c]
        for local_j in range(4):
            for c in range(7):
                V[B_rows[local_j]][col_offset + c] += S_V[local_j][c]
        for local_k in range(4):
            for c in range(7):
                W[C_rows[local_k]][col_offset + c] += S_W[local_k][c]
    
    def _add_schoolbook_2x2_2x3(A_rows, B_rows, C_rows, col_offset):
        """Add a schoolbook 2x2*2x3 block at the given column offset."""
        for local_i in range(4):
            for c in range(12):
                U[A_rows[local_i]][col_offset + c] += T_U[local_i][c]
        for local_j in range(6):
            for c in range(12):
                V[B_rows[local_j]][col_offset + c] += T_V[local_j][c]
        for local_k in range(6):
            for c in range(12):
                W[C_rows[local_k]][col_offset + c] += T_W[local_k][c]
    
    # C11 = A11*B11 + A12*B21
    _add_strassen_block(A11_rows, B11_rows, C11_rows, 0)
    _add_strassen_block(A12_rows, B21_rows, C11_rows, 7)
    
    # C12 = A11*B12 + A12*B22
    _add_schoolbook_2x2_2x3(A11_rows, B12_rows, C12_rows, 14)
    _add_schoolbook_2x2_2x3(A12_rows, B22_rows, C12_rows, 26)
    
    # C21 = A21*B11 + A22*B21
    _add_strassen_block(A21_rows, B11_rows, C21_rows, 38)
    _add_strassen_block(A22_rows, B21_rows, C21_rows, 45)
    
    # C22 = A21*B12 + A22*B22
    _add_schoolbook_2x2_2x3(A21_rows, B12_rows, C22_rows, 52)
    _add_schoolbook_2x2_2x3(A22_rows, B22_rows, C22_rows, 64)
    
    return {"U": U, "V": V, "W": W}

def _try_improve_2x2_2x3():
    """
    Try to find a better than schoolbook (rank 12) decomposition for 2x2 * 2x3.
    Use a randomized search with small integer/half-integer entries.
    """
    n, m, p = 2, 2, 3
    target_rank = 11  # Try to beat schoolbook's 12
    
    rng = random.Random(123)
    candidates = [-1, -0.5, 0, 0.5, 1]
    
    for attempt in range(200):
        U = [[rng.choice(candidates) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[rng.choice(candidates) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[rng.choice(candidates) for _ in range(target_rank)] for _ in range(p*n)]
        
        if _verify_decomposition(U, V, W, n, m, p):
            if _verify_integer_halfinteger(U, V, W):
                return U, V, W
    
    return None

def _build_strassen_4x4_4x5_improved():
    """
    Build an improved Strassen-based decomposition for 4x4 * 4x5 -> 4x5.
    Uses Strassen for 2x2*2x2 (rank 7) and tries to use a better 2x2*2x3 decomposition.
    """
    n, m, p = 4, 4, 5
    
    S_U, S_V, S_W = _strassen_2x2_decomposition()
    
    # Try to get an improved 2x2*2x3 decomposition
    improved = _try_improve_2x2_2x3()
    if improved is not None:
        T_U, T_V, T_W = improved
        t_rank = len(T_U[0])
    else:
        T_U, T_V, T_W = _strassen_2x2_2x3_decomposition()
        t_rank = 12
    
    total_rank = 14 + 2*t_rank + 14 + 2*t_rank  # 28 + 4*t_rank
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    A11_rows = [0, 1, 4, 5]
    A12_rows = [2, 3, 6, 7]
    A21_rows = [8, 9, 12, 13]
    A22_rows = [10, 11, 14, 15]
    
    B11_rows = [0, 1, 5, 6]
    B12_rows = [2, 3, 4, 7, 8, 9]
    B21_rows = [10, 11, 15, 16]
    B22_rows = [12, 13, 14, 17, 18, 19]
    
    C11_rows = [0, 4, 1, 5]
    C12_rows = [8, 12, 16, 9, 13, 17]
    C21_rows = [2, 6, 3, 7]
    C22_rows = [10, 14, 18, 11, 15, 19]
    
    def _add_strassen_block(A_rows, B_rows, C_rows, col_offset):
        for local_i in range(4):
            for c in range(7):
                U[A_rows[local_i]][col_offset + c] += S_U[local_i][c]
        for local_j in range(4):
            for c in range(7):
                V[B_rows[local_j]][col_offset + c] += S_V[local_j][c]
        for local_k in range(4):
            for c in range(7):
                W[C_rows[local_k]][col_offset + c] += S_W[local_k][c]
    
    def _add_2x2_2x3_block(A_rows, B_rows, C_rows, col_offset):
        for local_i in range(4):
            for c in range(t_rank):
                U[A_rows[local_i]][col_offset + c] += T_U[local_i][c]
        for local_j in range(6):
            for c in range(t_rank):
                V[B_rows[local_j]][col_offset + c] += T_V[local_j][c]
        for local_k in range(6):
            for c in range(t_rank):
                W[C_rows[local_k]][col_offset + c] += T_W[local_k][c]
    
    col = 0
    # C11 = A11*B11 + A12*B21
    _add_strassen_block(A11_rows, B11_rows, C11_rows, col)
    col += 7
    _add_strassen_block(A12_rows, B21_rows, C11_rows, col)
    col += 7
    
    # C12 = A11*B12 + A12*B22
    _add_2x2_2x3_block(A11_rows, B12_rows, C12_rows, col)
    col += t_rank
    _add_2x2_2x3_block(A12_rows, B22_rows, C12_rows, col)
    col += t_rank
    
    # C21 = A21*B11 + A22*B21
    _add_strassen_block(A21_rows, B11_rows, C21_rows, col)
    col += 7
    _add_strassen_block(A22_rows, B21_rows, C21_rows, col)
    col += 7
    
    # C22 = A21*B12 + A22*B22
    _add_2x2_2x3_block(A21_rows, B12_rows, C22_rows, col)
    col += t_rank
    _add_2x2_2x3_block(A22_rows, B22_rows, C22_rows, col)
    col += t_rank
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, build the schoolbook as fallback
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p  # 80
    
    # Try the Strassen-based construction
    if time.time() < deadline:
        try:
            strassen_result = _build_strassen_4x4_4x5()
            if _verify_decomposition(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                if _verify_integer_halfinteger(strassen_result["U"], strassen_result["V"], strassen_result["W"]):
                    rank = len(strassen_result["U"][0])
                    if rank < best_rank:
                        best_rank = rank
                        best_result = strassen_result
        except Exception:
            pass
    
    # Try the improved Strassen-based construction
    if time.time() < deadline:
        try:
            improved_result = _build_strassen_4x4_4x5_improved()
            if _verify_decomposition(improved_result["U"], improved_result["V"], improved_result["W"], n, m, p):
                if _verify_integer_halfinteger(improved_result["U"], improved_result["V"], improved_result["W"]):
                    rank = len(improved_result["U"][0])
                    if rank < best_rank:
                        best_rank = rank
                        best_result = improved_result
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
