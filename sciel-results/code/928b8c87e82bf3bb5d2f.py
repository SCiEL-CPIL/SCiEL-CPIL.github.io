DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a low-rank tensor decomposition for 2x5 by 5x5 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 sub-blocks to reduce rank.
    """
    n, m, p = DIMENSIONS
    
    # Rank 48 Decomposition
    # Split columns k into pairs (0,1), (2,3) and single (4)
    # Split rows j into (0,1) and (2,3,4)
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # 10 x 48
    V = [[0] * rank for _ in range(m * p)]  # 25 x 48
    W = [[0] * rank for _ in range(p * n)]  # 10 x 48
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2 block
    # A indices: (0,0), (0,1), (1,0), (1,1) -> U idx: 0, 1, 5, 6
    # B indices: (0,0), (0,1), (1,0), (1,1) -> V idx: 0, 1, 5, 6 (relative to block)
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W idx: 0, 1, 2, 3 (relative to block)
    
    def add_strassen(j_start, k_start):
        nonlocal col
        # j_start, k_start are 0 or 2
        # A part: rows 0,1 of A, cols j_start, j_start+1
        # B part: rows j_start, j_start+1 of B, cols k_start, k_start+1
        # C part: rows 0,1 of C, cols k_start, k_start+1
        
        # U indices for A[0, j_start], A[0, j_start+1], A[1, j_start], A[1, j_start+1]
        u_00 = 0 * m + j_start
        u_01 = 0 * m + (j_start + 1)
        u_10 = 1 * m + j_start
        u_11 = 1 * m + (j_start + 1)
        
        # V indices for B[j_start, k_start], B[j_start, k_start+1], B[j_start+1, k_start], B[j_start+1, k_start+1]
        v_00 = j_start * p + k_start
        v_01 = j_start * p + (k_start + 1)
        v_10 = (j_start + 1) * p + k_start
        v_11 = (j_start + 1) * p + (k_start + 1)
        
        # W indices for C[0, k_start], C[0, k_start+1], C[1, k_start], C[1, k_start+1]
        w_00 = k_start * n + 0
        w_01 = k_start * n + 1
        w_10 = (k_start + 1) * n + 0
        w_11 = (k_start + 1) * n + 1
        
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        U[u_00][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[u_10][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[u_00][col] += 1
        V[v_01][col] += 1
        V[v_11][col] -= 1
        W[w_01][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[u_11][col] += 1
        V[v_10][col] += 1
        V[v_00][col] -= 1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[u_00][col] += 1
        U[u_01][col] += 1
        V[v_11][col] += 1
        W[w_00][col] -= 1
        W[w_01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_10][col] += 1
        U[u_00][col] -= 1
        V[v_00][col] += 1
        V[v_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_01][col] += 1
        U[u_11][col] -= 1
        V[v_10][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

    # Add schoolbook terms for 2x3x2 block
    # A rows 0,1; B rows j_start..j_end; C cols k_start, k_start+1
    def add_schoolbook_2x3x2(j_start, j_end, k_start):
        nonlocal col
        # j_start, j_end are indices in B (and A cols)
        # Here we assume j range is 2,3,4
        for j in range(j_start, j_end + 1):
            for k in range(k_start, k_start + 2):
                # U: A[0, j], A[1, j]
                # V: B[j, k]
                # W: C[0, k], C[1, k]
                # Wait, schoolbook is one term per (i,j,k).
                # We need 2 (for i) * 3 (for j) * 2 (for k) = 12 terms.
                for i in range(2):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Add schoolbook terms for 2x3x1 block
    def add_schoolbook_2x3x1(j_start, j_end, k):
        nonlocal col
        for j in range(j_start, j_end + 1):
            for i in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Add schoolbook terms for 2x2x1 block
    def add_schoolbook_2x2x1(j_start, j_end, k):
        nonlocal col
        for j in range(j_start, j_end + 1):
            for i in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Block 1: k in {0, 1}
    add_strassen(0, 0) # j in {0,1}, k in {0,1}
    add_schoolbook_2x3x2(2, 4, 0) # j in {2,3,4}, k in {0,1}
    
    # Block 2: k in {2, 3}
    add_strassen(0, 2) # j in {0,1}, k in {2,3}
    add_schoolbook_2x3x2(2, 4, 2) # j in {2,3,4}, k in {2,3}
    
    # Block 3: k in {4}
    add_schoolbook_2x2x1(0, 1, 4) # j in {0,1}, k in {4}
    add_schoolbook_2x3x1(2, 4, 4) # j in {2,3,4}, k in {4}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
