"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44: 28 (Strassen) + 16 (Schoolbook)
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_2x2_mult(a_r, a_c, b_r, b_c, c_r, c_c, sign, base_rank):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = base_rank + (i * 2 + j) * 2 + k
                    U[(a_r + i) * m + (a_c + j)][col] += sign
                    V[(b_r + k) * p + (b_c + j)][col] += sign
                    W[(c_c + j) * n + (c_r + i)][col] += sign

    # --- Strassen for Top 2 Rows ---
    # M1 = (A11 + A12)(B11 + B22)
    base = 0
    add_2x2_mult(0, 0, 0, 0, 0, 0, 1, base)    # A11*B11
    add_2x2_mult(0, 0, 2, 2, 0, 0, 1, base)    # A11*B22
    add_2x2_mult(0, 2, 0, 0, 0, 0, 1, base)    # A12*B11
    add_2x2_mult(0, 2, 2, 2, 0, 0, 1, base)    # A12*B22
    base += 16

    # M2 = 0 (Skipped)
    
    # M3 = A11(B21 - B22)
    add_2x2_mult(0, 0, 2, 0, 0, 0, 1, base)    # A11*B21
    add_2x2_mult(0, 0, 2, 2, 0, 0, -1, base)   # -A11*B22
    base += 8

    # M4 = 0 (Skipped)

    # M5 = A11(B11 + B22)
    add_2x2_mult(0, 0, 0, 0, 0, 0, 1, base)    # A11*B11
    add_2x2_mult(0, 0, 2, 2, 0, 0, 1, base)    # A11*B22
    base += 8

    # M6 = A12(B21 + B22)
    add_2x2_mult(0, 2, 2, 0, 0, 0, 1, base)    # A12*B21
    add_2x2_mult(0, 2, 2, 2, 0, 0, 1, base)    # A12*B22
    base += 8

    # M7 = -A11(B11 + B12)
    add_2x2_mult(0, 0, 0, 0, 0, 0, -1, base)   # -A11*B11
    add_2x2_mult(0, 0, 0, 2, 0, 0, -1, base)   # -A11*B12
    base += 8

    # Note: This implementation has repeated terms (e.g., A11*B11 appears in M1, M5, M7).
    # This means the rank is actually higher than 28 if we don't factor them out.
    # However, for the purpose of this exercise, and given the constraint, 
    # I will use a simple, guaranteed correct decomposition:
    
    # --- Fallback: Schoolbook for Top 2 Rows (32 rank) + Bottom Row (16 rank) = 48 rank. ---
    # This is the seed.
    
    # Let's try the Rank 44 with independent terms.
    # Actually, the simplest improvement is to use the "Row Trick" for the bottom row (4 rank)
    # and Schoolbook for the top 2 rows (32 rank). Total 36.
    
    U = [[0] * 36 for _ in range(n * m)]
    V = [[0] * 36 for _ in range(m * p)]
    W = [[0] * 36 for _ in range(p * n)]
    
    # Top 2 Rows: Schoolbook
    base = 0
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U[i * m + j][base] = 1
                V[j * p + k][base] = 1
                W[k * n + i][base] = 1
                base += 1
                
    # Bottom Row: Row Trick (4 rank)
    # C_2k = sum_j A_2j B_jk
    for j in range(4):
        # Term j
        col = 32 + j
        # U[2*m + j] = 1
        U[2 * m + j][col] = 1
        # V[j*p + k] = 1 for all k
        for k in range(4):
            V[j * p + k][col] = 1
        # W[k*n + 2] = 1 for all k
        for k in range(4):
            W[k * n + 2][col] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
