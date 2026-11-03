"""Direct block decomposition for 3x4 * 4x6 -> 3x6 matrix multiplication.
Uses Strassen for 2x2*2x2 (rank 7) and direct for 1x2*2x2 (rank 4).
Total rank = 66.
"""
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    
    # Target rank is 66
    # A (3x4) split rows: 1, 2
    # B (4x6) split rows: 2, 2; cols: 2, 2, 2
    # C (3x6)
    
    # Row 0 of A (1x4) * B (4x6) -> C (1x6)
    # Split B rows into 2,2. Split B cols into 2,2,2.
    # For each B row-block (2) and B col-block (2), we have 1x2 * 2x2.
    # Rank of 1x2*2x2 is 4 (schoolbook).
    # 2 row-blocks * 3 col-blocks * 4 rank = 24.
    
    # Rows 1-2 of A (2x4) * B (4x6) -> C (2x6)
    # Split B rows into 2,2. Split B cols into 2,2,2.
    # For each B row-block (2) and B col-block (2), we have 2x2 * 2x2.
    # Rank of 2x2*2x2 is 7 (Strassen).
    # 2 row-blocks * 3 col-blocks * 7 rank = 42.
    
    total_rank = 24 + 42 # 66
    
    U = [[0] * total_rank for _ in range(n * m)]  # (12, 66)
    V = [[0] * total_rank for _ in range(m * p)]  # (24, 66)
    W = [[0] * total_rank for _ in range(p * n)]  # (18, 66)
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add Strassen 2x2*2x2 block at given offsets."""
        nonlocal col
        # A indices (U is flattened row-major: i*m + j)
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        
        # B indices (V is flattened row-major: j*p + k)
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (W is trace convention: k*n + i)
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
        
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 1x2*2x2 block (rank 4) at given offsets."""
        nonlocal col
        # A indices (1x2)
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        
        # B indices (2x2)
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (1x2)
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        # M1: A0*B00
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        # M2: A1*B10
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # M3: A0*B01
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        # M4: A1*B11
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1

    # --- Construction ---
    
    # Top row of A (row 0)
    # B row blocks: 0-1, 2-3
    # B col blocks: 0-1, 2-3, 4-5
    
    # A[0, 0:2] * B[0:2, 0:2] -> C[0, 0:2]
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    # A[0, 0:2] * B[0:2, 2:4] -> C[0, 2:4]
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    # A[0, 0:2] * B[0:2, 4:6] -> C[0, 4:6]
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    
    # A[0, 2:4] * B[2:4, 0:2] -> C[0, 0:2]
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    # A[0, 2:4] * B[2:4, 2:4] -> C[0, 2:4]
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    # A[0, 2:4] * B[2:4, 4:6] -> C[0, 4:6]
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    
    # Bottom rows of A (rows 1-2)
    # B row blocks: 0-1, 2-3
    # B col blocks: 0-1, 2-3, 4-5
    
    # A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    add_strassen(1, 0, 0, 0, 1, 0)
    # A[1:3, 0:2] * B[0:2, 2:4] -> C[1:3, 2:4]
    add_strassen(1, 0, 0, 2, 1, 2)
    # A[1:3, 0:2] * B[0:2, 4:6] -> C[1:3, 4:6]
    add_strassen(1, 0, 0, 4, 1, 4)
    
    # A[1:3, 2:4] * B[2:4, 0:2] -> C[1:3, 0:2]
    add_strassen(1, 2, 2, 0, 1, 0)
    # A[1:3, 2:4] * B[2:4, 2:4] -> C[1:3, 2:4]
    add_strassen(1, 2, 2, 2, 1, 2)
    # A[1:3, 2:4] * B[2:4, 4:6] -> C[1:3, 4:6]
    add_strassen(1, 2, 2, 4, 1, 4)
    
    if col != total_rank:
        # Fallback to schoolbook if something went wrong with column counting
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

    # Optional: Quick verification with random matrices
    # This is cheap enough (small matrices) and ensures correctness.
    import random
    random.seed(42)
    
    def mat_mult(A, B):
        rows_a = len(A)
        cols_a = len(A[0])
        cols_b = len(B[0])
        C = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
        for i in range(rows_a):
            for k in range(cols_b):
                s = 0
                for j in range(cols_a):
                    s += A[i][j] * B[j][k]
                C[i][k] = s
        return C

    # Generate random integer matrices
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    C_expected = mat_mult(A, B)
    
    # Compute C_decomposed
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])
    # Note: U, V, W are lists of lists.
    # U[i*m+j][c]
    # V[j*p+k][c]
    # W[k*n+i][c]
    
    C_decomposed = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for c in range(total_rank):
                w_val = W[k * n + i][c]
                if w_val == 0:
                    continue
                u_v_prod = 0
                for j in range(m):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    if u_val == 0 or v_val == 0:
                        continue
                    u_v_prod += u_val * v_val
                s += w_val * u_v_prod
            C_decomposed[i][k] = s
            
    if C_decomposed != C_expected:
        # Fallback to schoolbook
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

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Time taken: {time.time() - start:.4f}s")
    print(f"Rank: {len(res['U'][0])}")
