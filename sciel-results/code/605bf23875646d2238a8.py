import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # We will build a rank-44 decomposition.
    # Structure:
    # C (3x4) is split into C_left (3x2) and C_right (3x2).
    # C_left = A_left * B_topleft + A_right * B_bottomleft
    # C_right = A_left * B_topright + A_right * B_bottomright
    # A_left is 3x2 (cols 0,1), A_right is 3x2 (cols 2,3).
    # B_topleft is 2x2 (rows 0,1 cols 0,1), etc.
    
    # Each term is a 3x2 by 2x2 multiplication.
    # Rank of 3x2x2 multiplication:
    # Split A (3x2) into A_top (2x2) and A_bot (1x2).
    # A_top * B (2x2 by 2x2) -> Rank 7 (Strassen).
    # A_bot * B (1x2 by 2x2) -> Rank 4 (Trivial).
    # Total Rank 11 per term.
    # 4 terms -> Total Rank 44.
    
    RANK = 44
    U = np.zeros((n*m, RANK))
    V = np.zeros((m*p, RANK))
    W = np.zeros((p*n, RANK))
    
    col_offset = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B12 - B21)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    # The 4 blocks are:
    # 1. A_cols [0,1], B_rows [0,1], C_cols [0,1]
    # 2. A_cols [2,3], B_rows [2,3], C_cols [0,1]
    # 3. A_cols [0,1], B_rows [0,1], C_cols [2,3]
    # 4. A_cols [2,3], B_rows [2,3], C_cols [2,3]
    
    # Note: B is 4x4.
    # B_topleft: rows 0,1 cols 0,1.
    # B_bottomleft: rows 2,3 cols 0,1.
    # B_topright: rows 0,1 cols 2,3.
    # B_bottomright: rows 2,3 cols 2,3.
    
    # A is 3x4.
    # A_left: cols 0,1.
    # A_right: cols 2,3.
    
    # For each of the 4 combinations (A_part, B_part, C_part):
    # A_part is 3x2.
    # B_part is 2x2.
    # C_part is 3x2.
    
    a_parts = [
        [0, 1], # Left
        [2, 3]  # Right
    ]
    
    b_parts = [
        [0, 1], # Top
        [2, 3]  # Bottom
    ]
    
    c_parts = [
        [0, 1], # Left
        [2, 3]  # Right
    ]
    
    # We iterate over the 4 blocks
    # Block 1: A_left, B_top, C_left
    # Block 2: A_right, B_bottom, C_left
    # Block 3: A_left, B_top, C_right
    # Block 4: A_right, B_bottom, C_right
    
    # Wait, C_left = A_left B_top + A_right B_bottom.
    # C_right = A_left B_top? No.
    # C = A B.
    # C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2].
    # C[:, 2:4] = A[:, 0:2] B[0:2, 2:4] + A[:, 2:4] B[2:4, 2:4].
    
    # So the pairs are:
    # (A_left, B_top, C_left)
    # (A_right, B_bottom, C_left)
    # (A_left, B_top, C_right) -> Wait, B for C_right top part is B[0:2, 2:4] which is B_topright.
    # (A_right, B_bottom, C_right) -> B[2:4, 2:4] which is B_bottomright.
    
    # So we need 4 distinct B blocks:
    # B00 (Top-Left), B10 (Bottom-Left), B01 (Top-Right), B11 (Bottom-Right).
    
    # And 2 A blocks:
    # A0 (Left), A1 (Right).
    
    # And 2 C blocks:
    # C0 (Left), C1 (Right).
    
    # Combinations:
    # 1. A0 * B00 -> C0
    # 2. A1 * B10 -> C0
    # 3. A0 * B01 -> C1
    # 4. A1 * B11 -> C1
    
    # Each combination is a 3x2 * 2x2 -> 3x2 multiplication.
    
    def add_strassen_block(U, V, W, offset, a_cols, b_rows, c_cols):
        """
        Adds a 3x2 x 2x2 -> 3x2 decomposition.
        a_cols: indices of A columns (2 indices)
        b_rows: indices of B rows (2 indices)
        c_cols: indices of C columns (2 indices)
        """
        # A is 3x2. Rows 0,1,2.
        # B is 2x2. Rows b_rows[0], b_rows[1]. Cols 0,1 (local).
        # C is 3x2. Rows 0,1,2. Cols c_cols[0], c_cols[1].
        
        # Strassen part for A_top (2x2) * B (2x2) -> C_top (2x2)
        # A_top rows 0,1.
        # B rows b_rows.
        # C_top rows 0,1.
        
        # Local indices for A_top:
        # A[0, a_cols[0]], A[0, a_cols[1]]
        # A[1, a_cols[0]], A[1, a_cols[1]]
        
        # Local indices for B:
        # B[b_rows[0], 0], B[b_rows[0], 1]
        # B[b_rows[1], 0], B[b_rows[1], 1]
        # Note: B is 4x4 global. Local cols 0,1 correspond to global cols c_cols[0], c_cols[1]?
        # No, B is 4x4. The multiplication is A(3x2) * B(2x2).
        # The "2x2" B block uses columns c_cols[0] and c_cols[1] of the global B?
        # Yes, because C cols are c_cols.
        # So B block is B[b_rows, c_cols].
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11 = A[0, a_cols[0]]
        # A22 = A[1, a_cols[1]]
        # B11 = B[b_rows[0], c_cols[0]]
        # B22 = B[b_rows[1], c_cols[1]]
        
        # We need to add 7 terms for Strassen and 4 terms for Trivial.
        
        # Strassen Terms
        # 1. (A00 + A11) * (B00 + B11)
        U[u_idx(0, a_cols[0]), offset] += 1.0
        U[u_idx(1, a_cols[1]), offset] += 1.0
        V[v_idx(b_rows[0], c_cols[0]), offset] += 1.0
        V[v_idx(b_rows[1], c_cols[1]), offset] += 1.0
        W[w_idx(c_cols[0], 0), offset] += 1.0
        W[w_idx(c_cols[1], 0), offset] += 1.0
        W[w_idx(c_cols[0], 1), offset] += 1.0
        W[w_idx(c_cols[1], 1), offset] += 1.0
        offset += 1
        
        # 2. (A10 + A11) * B00
        # A10 = A[1, a_cols[0]]
        # A11 = A[1, a_cols[1]]
        # B00 = B[b_rows[0], c_cols[0]]
        U[u_idx(1, a_cols[0]), offset] += 1.0
        U[u_idx(1, a_cols[1]), offset] += 1.0
        V[v_idx(b_rows[0], c_cols[0]), offset] += 1.0
        W[w_idx(c_cols[0], 1), offset] += 1.0
        offset += 1
        
        # 3. A00 * (B10 - B00)
        # A00 = A[0, a_cols[0]]
        # B10 = B[b_rows[1], c_cols[0]]
        # B00 = B[b_rows[0], c_cols[0]]
        U[u_idx(0, a_cols[0]), offset] += 1.0
        V[v_idx(b_rows[1], c_cols[0]), offset] += 1.0
        V[v_idx(b_rows[0], c_cols[0]), offset] -= 1.0
        W[w_idx(c_cols[1], 0), offset] += 1.0
        W[w_idx(c_cols[1], 1), offset] += 1.0
        offset += 1
        
        # 4. A11 * (B01 - B10)
        # A11 = A[1, a_cols[1]]
        # B01 = B[b_rows[0], c_cols[1]]
        # B10 = B[b_rows[1], c_cols[0]]
        U[u_idx(1, a_cols[1]), offset] += 1.0
        V[v_idx(b_rows[0], c_cols[1]), offset] += 1.0
        V[v_idx(b_rows[1], c_cols[0]), offset] -= 1.0
        W[w_idx(c_cols[0], 1), offset] += 1.0
        W[w_idx(c_cols[1], 1), offset] += 1.0
        offset += 1
        
        # 5. (A00 + A01) * B11
        # A00 = A[0, a_cols[0]]
        # A01 = A[0, a_cols[1]]
        # B11 = B[b_rows[1], c_cols[1]]
        U[u_idx(0, a_cols[0]), offset] += 1.0
        U[u_idx(0, a_cols[1]), offset] += 1.0
        V[v_idx(b_rows[1], c_cols[1]), offset] += 1.0
        W[w_idx(c_cols[0], 0), offset] -= 1.0
        W[w_idx(c_cols[1], 1), offset] += 1.0
        offset += 1
        
        # 6. (A10 - A00) * (B00 + B01)
        # A10 = A[1, a_cols[0]]
        # A00 = A[0, a_cols[0]]
        # B00 = B[b_rows[0], c_cols[0]]
        # B01 = B[b_rows[0], c_cols[1]]
        U[u_idx(1, a_cols[0]), offset] += 1.0
        U[u_idx(0, a_cols[0]), offset] -= 1.0
        V[v_idx(b_rows[0], c_cols[0]), offset] += 1.0
        V[v_idx(b_rows[0], c_cols[1]), offset] += 1.0
        W[w_idx(c_cols[1], 1), offset] += 1.0
        offset += 1
        
        # 7. (A01 - A11) * (B10 + B11)
        # A01 = A[0, a_cols[1]]
        # A11 = A[1, a_cols[1]]
        # B10 = B[b_rows[1], c_cols[0]]
        # B11 = B[b_rows[1], c_cols[1]]
        U[u_idx(0, a_cols[1]), offset] += 1.0
        U[u_idx(1, a_cols[1]), offset] -= 1.0
        V[v_idx(b_rows[1], c_cols[0]), offset] += 1.0
        V[v_idx(b_rows[1], c_cols[1]), offset] += 1.0
        W[w_idx(c_cols[0], 0), offset] += 1.0
        offset += 1
        
        # Trivial part for A_bot (1x2) * B (2x2) -> C_bot (1x2)
        # A_bot is row 2 of A.
        # A20 = A[2, a_cols[0]]
        # A21 = A[2, a_cols[1]]
        # C_bot rows 2.
        
        # 8. A20 * B00
        U[u_idx(2, a_cols[0]), offset] += 1.0
        V[v_idx(b_rows[0], c_cols[0]), offset] += 1.0
        W[w_idx(c_cols[0], 2), offset] += 1.0
        offset += 1
        
        # 9. A20 * B01
        U[u_idx(2, a_cols[0]), offset] += 1.0
        V[v_idx(b_rows[0], c_cols[1]), offset] += 1.0
        W[w_idx(c_cols[1], 2), offset] += 1.0
        offset += 1
        
        # 10. A21 * B10
        U[u_idx(2, a_cols[1]), offset] += 1.0
        V[v_idx(b_rows[1], c_cols[0]), offset] += 1.0
        W[w_idx(c_cols[0], 2), offset] += 1.0
        offset += 1
        
        # 11. A21 * B11
        U[u_idx(2, a_cols[1]), offset] += 1.0
        V[v_idx(b_rows[1], c_cols[1]), offset] += 1.0
        W[w_idx(c_cols[1], 2), offset] += 1.0
        offset += 1
        
        return offset

    # Block 1: A_left [0,1], B_top [0,1], C_left [0,1]
    col_offset = add_strassen_block(U, V, W, col_offset, [0, 1], [0, 1], [0, 1])
    
    # Block 2: A_right [2,3], B_bottom [2,3], C_left [0,1]
    col_offset = add_strassen_block(U, V, W, col_offset, [2, 3], [2, 3], [0, 1])
    
    # Block 3: A_left [0,1], B_top [0,1], C_right [2,3]
    # Note: B for C_right uses cols 2,3. So B block is B[0:2, 2:4].
    # This is B_topright.
    col_offset = add_strassen_block(U, V, W, col_offset, [0, 1], [0, 1], [2, 3])
    
    # Block 4: A_right [2,3], B_bottom [2,3], C_right [2,3]
    # B block is B[2:4, 2:4].
    col_offset = add_strassen_block(U, V, W, col_offset, [2, 3], [2, 3], [2, 3])
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
