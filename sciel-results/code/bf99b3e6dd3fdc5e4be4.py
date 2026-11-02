"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 51 decomposition
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # We decompose B into 3 blocks of 3x3: B_1, B_2, B_3
    # A is 2x3.
    # For each block B_j (3x3), we split B_j into B_j_left (3x2) and B_j_right (3x1)
    # A is split into A_L (2x1) and A_R (2x2)
    
    for block in range(3):
        # Indices for A: A is 2x3.
        # A_L uses col 0 of A. A_R uses cols 1, 2 of A.
        # A indices:
        # A[0,0] -> U[0], A[0,1] -> U[1], A[0,2] -> U[2]
        # A[1,0] -> U[3], A[1,1] -> U[4], A[1,2] -> U[5]
        
        # B block j starts at row 0,1,2 and cols 3*block, 3*block+1, 3*block+2
        # V indices:
        # B[r, c] -> V[r*9 + c]
        
        # Part 1: A_L * B_j_left (2x1 * 1x2)
        # A_L is A[:, 0] -> U[0], U[3]
        # B_j_left is B[:, 3*block : 3*block+2]
        # B rows are 0,1,2. Only row 0 is used for B_T (1x2)?
        # Wait, A_L is 2x1. B_j_left is 3x2.
        # A_L * B_j_left is not defined directly as 2x1 * 3x2.
        # The multiplication is A (2x3) * B (3x3).
        # Sum over k=0..2 of A[:,k] * B[k,:].
        # My previous decomposition:
        # A = [A_L, A_R]. A_L is col 0. A_R is cols 1,2.
        # B = [B_T; B_B]. B_T is row 0. B_B is rows 1,2.
        # C = A_L * B_T + A_R * B_B.
        # A_L (2x1) * B_T (1x2) -> 2x2.
        # A_R (2x2) * B_B (2x2) -> 2x2.
        
        # So for each block, we need:
        # 1. A_L * B_T (where B_T is the 1x2 slice of the block's first row)
        # 2. A_R * B_B (where B_B is the 2x2 slice of the block's last two rows)
        
        # Block j corresponds to B columns 3j, 3j+1, 3j+2.
        # B_T is B[0, 3j], B[0, 3j+1].
        # B_B is B[1, 3j], B[1, 3j+1], B[2, 3j], B[2, 3j+1] (2x2 matrix).
        # Wait, B is 3x3 block.
        # B_T is row 0, cols 0,1 of the block.
        # B_B is rows 1,2, cols 0,1 of the block.
        # What about col 2 of the block?
        # My previous split was B (3x3) into B_left (3x2) and B_right (3x1).
        # B_left cols 0,1. B_right col 2.
        # A_L * B_T_part (from B_left row 0)
        # A_R * B_B_part (from B_left rows 1,2)
        # A_L * B_T_right (from B_right row 0)
        # A_R * B_B_right (from B_right rows 1,2)
        
        # Let's re-verify the rank counts.
        # B_left (3x2):
        #   B_T_left (1x2): A_L * B_T_left (2x1 * 1x2) -> Rank 4.
        #   B_B_left (2x2): A_R * B_B_left (2x2 * 2x2) -> Rank 7 (Strassen).
        # B_right (3x1):
        #   B_T_right (1x1): A_L * B_T_right (2x1 * 1x1) -> Rank 2.
        #   B_B_right (2x1): A_R * B_B_right (2x2 * 2x1) -> Rank 4.
        # Total 17. Correct.

        # --- Implementing the 17 terms for this block ---
        
        # 1. A_L * B_T_left (2x1 * 1x2)
        # A_L indices: U[0] (A[0,0]), U[3] (A[1,0])
        # B_T_left indices: V[0*9 + 3*block] (B[0, 3j]), V[0*9 + 3*block + 1] (B[0, 3j+1])
        # Output W indices:
        # W is p*n = 9*2 = 18.
        # W[k*n + i].
        # Output of A_L * B_T_left is 2x2.
        # C[0,0] = A[0,0]B[0,3j] -> W[3j*2 + 0]
        # C[0,1] = A[0,0]B[0,3j+1] -> W[(3j+1)*2 + 0]
        # C[1,0] = A[1,0]B[0,3j] -> W[3j*2 + 1]
        # C[1,1] = A[1,0]B[0,3j+1] -> W[(3j+1)*2 + 1]
        
        # Term 1: A[0,0] * B[0, 3j]
        add_term(0, 3*block, 3*block*2 + 0)
        # Term 2: A[0,0] * B[0, 3j+1]
        add_term(0, 3*block + 1, (3*block + 1)*2 + 0)
        # Term 3: A[1,0] * B[0, 3j]
        add_term(3, 3*block, 3*block*2 + 1)
        # Term 4: A[1,0] * B[0, 3j+1]
        add_term(3, 3*block + 1, (3*block + 1)*2 + 1)
        
        # 2. A_R * B_B_left (2x2 * 2x2) Strassen
        # A_R indices:
        # A[0,1] -> U[1], A[0,2] -> U[2]
        # A[1,1] -> U[4], A[1,2] -> U[5]
        # B_B_left indices:
        # B[1, 3j] -> V[1*9 + 3j], B[1, 3j+1] -> V[1*9 + 3j + 1]
        # B[2, 3j] -> V[2*9 + 3j], B[2, 3j+1] -> V[2*9 + 3j + 1]
        
        # Strassen for 2x2 * 2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B11 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M4 + M7
        # C21 = M1 - M2 + M6 + M7
        # C22 = M1 - M3 - M5 + M6
        
        # We need to map these to rank-1 terms.
        # Each M is a product of linear forms.
        # M1: (U[1]+U[5]) * (V[9+3j]+V[18+3j])
        # This expands to 4 terms:
        # U[1]*V[9+3j], U[1]*V[18+3j], U[5]*V[9+3j], U[5]*V[18+3j]
        
        # It is easier to just list the 7 Strassen products as rank-1 tensors in the "Strassen basis"
        # and then map them to W.
        # However, the interface requires U, V, W matrices.
        # We can just add the 4 expanded terms for each M?
        # No, Strassen reduces the number of multiplications.
        # The rank is the number of multiplications.
        # So we should have 7 terms for this block part.
        # But the terms are linear combinations of inputs.
        # U, V, W entries can be any integer/half-integer.
        
        # Let's define the 7 Strassen terms explicitly.
        # Let u1, u2, u3, u4 be A_R entries.
        # u1=U[1], u2=U[2], u3=U[4], u4=U[5]
        # Let v1, v2, v3, v4 be B_B_left entries.
        # v1=V[9+3j], v2=V[9+3j+1], v3=V[18+3j], v4=V[18+3j+1]
        
        # M1 = (u1+u4)(v1+v4)
        # M2 = (u3+u4)v1
        # M3 = u1(v3-v4)
        # M4 = u4(v1-v2)
        # M5 = (u1+u2)v4
        # M6 = (u3-u1)(v1+v2)
        # M7 = (u2-u4)(v3+v4)
        
        # Output C (2x2) from A_R B_B_left:
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M4 + M7
        # C21 = M1 - M2 + M6 + M7
        # C22 = M1 - M3 - M5 + M6
        
        # We need to add 7 rank-1 terms corresponding to M1..M7.
        # For each M, we add a column to U, V, W.
        # U values for M1: [1, 0, 0, 0, 1, 0] (for A_R indices 1,2,4,5? No, indices in U are global)
        # A_R indices in U: 1, 2, 4, 5.
        # M1 uses U[1] and U[5].
        
        # Let's create a helper to add a Strassen term.
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs, val=1):
            nonlocal col
            # u_coeffs: dict {global_U_idx: coeff}
            # v_coeffs: dict {global_V_idx: coeff}
            # w_coeffs: dict {global_W_idx: coeff}
            for idx, c in u_coeffs.items():
                U[idx][col] += c * val
            for idx, c in v_coeffs.items():
                V[idx][col] += c * val
            for idx, c in w_coeffs.items():
                W[idx][col] += c * val
            col += 1

        # M1
        add_strassen_term({1: 1, 5: 1}, {9+3*block: 1, 18+3*block: 1}, {3*block*2+0: 1, (3*block+1)*2+0: 0, 3*block*2+1: 1, (3*block+1)*2+1: 0})
        # Wait, W coeffs for M1 contribute to C11 and C21.
        # C11 += M1, C21 += M1.
        # C11 is W[3j*2+0], C21 is W[3j*2+1].
        # So w_coeffs for M1: {3j*2+0: 1, 3j*2+1: 1}.
        
        # Let's restart the Strassen mapping carefully.
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M4 + M7
        # C21 = M1 - M2 + M6 + M7
        # C22 = M1 - M3 - M5 + M6
        
        # W indices for this 2x2 output block (cols 3j, 3j+1):
        # C11 -> W[3j*2 + 0]
        # C12 -> W[(3j+1)*2 + 0]
        # C21 -> W[3j*2 + 1]
        # C22 -> W[(3j+1)*2 + 1]
        
        # M1:
        # U: {1: 1, 5: 1}
        # V: {9+3j: 1, 18+3j: 1}
        # W: {3j*2+0: 1, 3j*2+1: 1}
        add_strassen_term({1: 1, 5: 1}, {9+3*block: 1, 18+3*block: 1}, {3*block*2+0: 1, 3*block*2+1: 1})
        
        # M2:
        # U: {4: 1, 5: 1}
        # V: {9+3j: 1}
        # W: {3j*2+1: -1}
        add_strassen_term({4: 1, 5: 1}, {9+3*block: 1}, {3*block*2+1: -1})
        
        # M3:
        # U: {1: 1}
        # V: {18+3j: 1, 18+3j+1: -1}
        # W: {(3j+1)*2+0: 1, (3j+1)*2+1: -1}
        add_strassen_term({1: 1}, {18+3*block: 1, 18+3*block+1: -1}, {(3*block+1)*2+0: 1, (3*block+1)*2+1: -1})
        
        # M4:
        # U: {5: 1}
        # V: {9+3j: 1, 9+3j+1: -1}
        # W: {3j*2+0: -1, (3j+1)*2+0: 1}
        add_strassen_term({5: 1}, {9+3*block: 1, 9+3*block+1: -1}, {3*block*2+0: -1, (3*block+1)*2+0: 1})
        
        # M5:
        # U: {1: 1, 2: 1}
        # V: {18+3j+1: 1}
        # W: {3j*2+0: 1, (3j+1)*2+1: -1}
        add_strassen_term({1: 1, 2: 1}, {18+3*block+1: 1}, {3*block*2+0: 1, (3*block+1)*2+1: -1})
        
        # M6:
        # U: {4: 1, 1: -1}
        # V: {9+3j: 1, 9+3j+1: 1}
        # W: {3j*2+1: 1, (3j+1)*2+1: 1}
        add_strassen_term({4: 1, 1: -1}, {9+3*block: 1, 9+3*block+1: 1}, {3*block*2+1: 1, (3*block+1)*2+1: 1})
        
        # M7:
        # U: {2: 1, 5: -1}
        # V: {18+3j: 1, 18+3j+1: 1}
        # W: {3j*2+0: 1, (3j+1)*2+0: 1, 3j*2+1: 1, (3j+1)*2+1: 1}
        add_strassen_term({2: 1, 5: -1}, {18+3*block: 1, 18+3*block+1: 1}, {3*block*2+0: 1, (3*block+1)*2+0: 1, 3*block*2+1: 1, (3*block+1)*2+1: 1})
        
        # 3. A_L * B_T_right (2x1 * 1x1)
        # A_L: U[0], U[3]
        # B_T_right: V[0*9 + 3*block + 2] (B[0, 3j+2])
        # Output: C[0, 2], C[1, 2]
        # C[0,2] = A[0,0] B[0, 3j+2] -> W[(3j+2)*2 + 0]
        # C[1,2] = A[1,0] B[0, 3j+2] -> W[(3j+2)*2 + 1]
        
        add_term(0, 3*block + 2, (3*block + 2)*2 + 0)
        add_term(3, 3*block + 2, (3*block + 2)*2 + 1)
        
        # 4. A_R * B_B_right (2x2 * 2x1)
        # A_R: U[1], U[2], U[4], U[5]
        # B_B_right: V[1*9 + 3*block + 2] (B[1, 3j+2]), V[2*9 + 3*block + 2] (B[2, 3j+2])
        # Output: C[0, 2], C[1, 2]
        # C[0,2] += A[0,1]B[1, 3j+2] + A[0,2]B[2, 3j+2]
        # C[1,2] += A[1,1]B[1, 3j+2] + A[1,2]B[2, 3j+2]
        
        # This is 4 terms.
        add_term(1, 9 + 3*block + 2, (3*block + 2)*2 + 0)
        add_term(2, 18 + 3*block + 2, (3*block + 2)*2 + 0)
        add_term(4, 9 + 3*block + 2, (3*block + 2)*2 + 1)
        add_term(5, 18 + 3*block + 2, (3*block + 2)*2 + 1)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
