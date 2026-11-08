"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Rank 32 decomposition:
    # Part 1: 2x4x2 (cols 0-1 of B)
    #   Block A_L (0-1) x B_T (0-1): Rank 7
    #   Block A_R (2-3) x B_B (2-3): Rank 7
    #   Total 14
    # Part 2: 2x4x3 (cols 2-4 of B)
    #   Block A_L (0-1) x B_T (0-1): Rank 9 (Strassen 7 + Std 2)
    #   Block A_R (2-3) x B_B (2-3): Rank 9 (Strassen 7 + Std 2)
    #   Total 18
    # Grand Total: 32
    
    rank = 32
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def strassen_2x2x2(ua, ub, ua2, ub2, va, vb, va2, vb2, c00, c01, c10, c11):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[ua][col] += 1
        U[ub2][col] += 1 # Wait, indices passed carefully
        # Let's use explicit parameters to avoid confusion
        # A: a00, a01, a10, a11
        # B: b00, b01, b10, b11
        # C: c00, c01, c10, c11
        
        # Re-define inputs clearly inside call or pass tuples
        
    # Better: pass indices explicitly to the helper
    
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    def add_std_2x2x3(a00, a01, a10, a11, b00, b01, b10, b11, b02, b12, c00, c01, c10, c11, c02, c12):
        nonlocal col
        # First part: Strassen on cols 0,1
        add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
        
        # Second part: Standard for col 2
        # C02 += A00*B02 + A01*B12
        # C12 += A10*B02 + A11*B12
        
        # Term 1: A00 * B02
        U[a00][col] = 1
        V[b02][col] = 1
        W[c02][col] += 1
        col += 1
        
        # Term 2: A01 * B12
        U[a01][col] = 1
        V[b12][col] = 1
        W[c02][col] += 1
        col += 1
        
        # Term 3: A10 * B02
        U[a10][col] = 1
        V[b02][col] = 1
        W[c12][col] += 1
        col += 1
        
        # Term 4: A11 * B12
        U[a11][col] = 1
        V[b12][col] = 1
        W[c12][col] += 1
        col += 1

    # --- Part 1: 2x4x2 (B cols 0,1) ---
    
    # A_L (0,1) x B_T (0,1) -> C(:, 0:2)
    # A: u(0,0), u(0,1), u(1,0), u(1,1)
    # B: v(0,0), v(0,1), v(1,0), v(1,1)
    # C: w(0,0), w(0,1), w(1,0), w(1,1)
    add_strassen(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    )
    
    # A_R (2,3) x B_B (2,3) -> C(:, 0:2)
    # A: u(0,2), u(0,3), u(1,2), u(1,3)
    # B: v(2,0), v(2,1), v(3,0), v(3,1)
    # C: w(0,0), w(0,1), w(1,0), w(1,1)
    add_strassen(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    )

    # --- Part 2: 2x4x3 (B cols 2,3,4) ---
    
    # A_L (0,1) x B_T (0,1) -> C(:, 2:5)
    # A: u(0,0), u(0,1), u(1,0), u(1,1)
    # B: v(0,2), v(0,3), v(0,4), v(1,2), v(1,3), v(1,4)
    # C: w(2,0), w(2,1), w(2,2), w(3,0), w(3,1), w(3,2), w(4,0), w(4,1), w(4,2)
    # Note: B_T is 2x3. Rows 0,1 of B. Cols 2,3,4 of B.
    # b00=v(0,2), b01=v(0,3), b02=v(0,4)
    # b10=v(1,2), b11=v(1,3), b12=v(1,4)
    # c00=w(2,0), c01=w(2,1), c02=w(2,2)
    # c10=w(3,0), c11=w(3,1), c12=w(3,2)
    # Wait, C is 2x5. C[k, i].
    # w_idx(k, i) = k * 2 + i.
    # k is column of B (0-4). i is row of A (0-1).
    # So C[2, 0] is w_idx(2,0). C[2,1] is w_idx(2,1).
    # The output of 2x2x3 mult is 2 rows (A) x 3 cols (B).
    # So c00 corresponds to C[0, 0] of the local 2x3 block?
    # Local C row 0 is global C row 0. Local C col 0 is global C col 2.
    # So local c00 -> global w_idx(2,0).
    # local c01 -> global w_idx(3,0).
    # local c02 -> global w_idx(4,0).
    # local c10 -> global w_idx(2,1).
    # local c11 -> global w_idx(3,1).
    # local c12 -> global w_idx(4,1).
    
    add_std_2x2x3(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,2), v_idx(0,3), v_idx(0,4), # b00, b01, b02 (Row 0 of B_T, cols 2,3,4)
        v_idx(1,2), v_idx(1,3), v_idx(1,4), # b10, b11, b12 (Row 1 of B_T, cols 2,3,4)
        w_idx(2,0), w_idx(3,0), w_idx(4,0), # c00, c01, c02 (Row 0 of C, cols 2,3,4)
        w_idx(2,1), w_idx(3,1), w_idx(4,1)  # c10, c11, c12 (Row 1 of C, cols 2,3,4)
    )
    
    # A_R (2,3) x B_B (2,3) -> C(:, 2:5)
    # A: u(0,2), u(0,3), u(1,2), u(1,3)
    # B: v(2,2), v(2,3), v(2,4), v(3,2), v(3,3), v(3,4)
    # C: same as above
    add_std_2x2x3(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,2), v_idx(2,3), v_idx(2,4),
        v_idx(3,2), v_idx(3,3), v_idx(3,4),
        w_idx(2,0), w_idx(3,0), w_idx(4,0),
        w_idx(2,1), w_idx(3,1), w_idx(4,1)
    )
    
    # Trim if necessary (should be exactly 32)
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    elif col > rank:
        # Error in counting
        raise ValueError(f"Col {col} > Rank {rank}")

    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
