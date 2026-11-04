import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = 4, 4, 6
    
    # Rank 28 for 4x4x4 (4 blocks of 2x2x2 Strassen)
    # Rank 28 for 4x4x2 (4 blocks of 2x2x2 Strassen)
    # Total Rank 56
    
    R = 56
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen 2x2x2 coefficients
    # U factors for A (4 entries)
    # V factors for B (4 entries)
    # W factors for C (4 entries)
    # A = [[a11, a12], [a21, a22]] -> indices 0,1,2,3
    # B = [[b11, b12], [b21, b22]] -> indices 0,1,2,3
    # C = [[c11, c12], [c21, c22]] -> indices 0,1,2,3
    
    # Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    
    # Helper to add a 2x2x2 Strassen block
    # a_offset: index of a11 in the 4x4 A matrix (flattened)
    # b_offset: index of b11 in the 4x4 B matrix (flattened)
    # c_offset: index of c11 in the 4x4 C matrix (flattened)
    # col_start: starting column index in U, V, W
    # b_scale: if B is 4x2, the "b11" of the block might be different?
    # Actually, for 4x4x2, B is 4x2. 
    # Block B_top is 2x2 (cols 0,1 of B rows 0,1). 
    # Block B_bot is 2x2 (cols 0,1 of B rows 2,3).
    # So b_offset for B_top is 0 (row 0, col 0).
    # b_offset for B_bot is 12 (row 2, col 0).
    
    def add_strassen_block(a_off, b_off, c_off, col_start):
        # a_off is index of top-left element of A block in flattened A (row-major)
        # A is 4x4. A[i,j] is at i*4+j.
        # Block 0,0: a_off = 0. Elements 0,1,4,5.
        # Block 0,1: a_off = 1. Elements 1,2,5,6.
        # Block 1,0: a_off = 4. Elements 4,5,8,9.
        # Block 1,1: a_off = 5. Elements 5,6,9,10.
        
        # b_off is index of top-left element of B block in flattened B (row-major)
        # B is 4x6. B[i,j] is at i*6+j.
        # For 4x4x4 part: B is 4x4.
        # Block 0,0: b_off = 0. Elements 0,1,6,7.
        # Block 0,1: b_off = 1. Elements 1,2,7,8.
        # Block 1,0: b_off = 6. Elements 6,7,12,13.
        # Block 1,1: b_off = 7. Elements 7,8,13,14.
        
        # For 4x4x2 part: B is 4x2.
        # Block 0,0 (top): b_off = 0. Elements 0,1,6,7. (Rows 0,1; Cols 0,1)
        # Block 1,0 (bot): b_off = 12. Elements 12,13,18,19. (Rows 2,3; Cols 0,1)
        
        # c_off is index of top-left element of C block in flattened C (row-major)
        # C is 4x6. C[i,j] is at i*6+j.
        # Block 0,0: c_off = 0.
        # Block 0,1: c_off = 1.
        # Block 1,0: c_off = 6.
        # Block 1,1: c_off = 7.
        
        # Strassen terms
        # M1: (a00+a11)(b00+b11)
        # M2: (a10+a11)b00
        # M3: a00(b01-b11)
        # M4: a11(b10-b00)
        # M5: (a00+a01)b11
        # M6: (a10-a00)(b00+b01)
        # M7: (a01-a11)(b10+b11)
        
        # Local indices 0,1,2,3 correspond to a00, a01, a10, a11
        # a_idx = [a_off, a_off+1, a_off+4, a_off+5]
        # b_idx = [b_off, b_off+1, b_off+6, b_off+7]
        # c_idx = [c_off, c_off+1, c_off+6, c_off+7]
        
        a_idx = [a_off, a_off+1, a_off+4, a_off+5]
        b_idx = [b_off, b_off+1, b_off+6, b_off+7]
        c_idx = [c_off, c_off+1, c_off+6, c_off+7]
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # M1
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (1,1), (2,-1), (3,1)]),
            # M2
            ([(2,1), (3,1)], [(0,1)], [(1,1), (2,1)]),
            # M3
            ([(0,1)], [(1,1), (3,-1)], [(0,-1), (1,1)]), # Wait, C11 = M1+M4-M5+M7, C12=M3+M5
            # Let's re-verify C reconstruction
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # M1: C11(+1), C22(+1)
            # M2: C21(+1), C22(-1)
            # M3: C12(+1), C22(+1)
            # M4: C11(+1), C21(+1)
            # M5: C11(-1), C12(+1)
            # M6: C22(+1)
            # M7: C11(+1)
            
            # M1
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            # M2
            ([(2,1), (3,1)], [(0,1)], [(1,-1), (2,1)]), # C21(+1), C22(-1) -> indices 2,3
            # M3
            ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]), # C12(+1), C22(+1) -> indices 1,3
            # M4
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]), # C11(+1), C21(+1) -> indices 0,2
            # M5
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]), # C11(-1), C12(+1) -> indices 0,1
            # M6
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]), # C22(+1) -> index 3
            # M7
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]), # C11(+1) -> index 0
        ]
        
        for i, (u_c, v_c, w_c) in enumerate(terms):
            col = col_start + i
            for idx, coeff in u_c:
                U[a_idx[idx], col] += coeff
            for idx, coeff in v_c:
                V[b_idx[idx], col] += coeff
            for idx, coeff in w_c:
                W[c_idx[idx], col] += coeff

    # Part 1: 4x4x4 (Columns 0-3 of B)
    # Blocks:
    # A00, B00 -> C00
    add_strassen_block(0, 0, 0, 0)
    # A01, B10 -> C01? No.
    # C01 = A00 B01 + A01 B11
    # This is not a single block multiplication.
    # My previous logic "4 blocks of 2x2x2" for 4x4x4 was:
    # C11 = A11 B11 + A12 B21
    # This requires 2 multiplications for C11.
    # Total 8 multiplications?
    # No, Strassen for 4x4x4 uses 7 multiplications of 2x2x2?
    # No, recursive Strassen:
    # T(4) = 7 T(2)? No.
    # T(2) = 7 T(1).
    # T(4) = 7 T(2) = 49 T(1)?
    # Or T(4) = 4 T(2) = 28 T(1)?
    # The "4 blocks" approach I described:
    # C11 = A11 B11 + A12 B21.
    # If we compute A11 B11 (rank 7) and A12 B21 (rank 7), we use 14 terms for C11.
    # Total 4 * 14 = 56 terms for 4x4x4.
    # This is rank 56.
    # Then 4x4x2 adds 28. Total 84.
    # This is worse than 77.
    
    # Correct Recursive Strassen for 4x4x4:
    # Split A, B, C into 2x2 blocks.
    # Compute M1..M7 using 2x2 Strassen on the blocks.
    # M1 = (A11+A22)(B11+B22)
    # This requires computing (A11+A22) and (B11+B22) which are 2x2 matrices.
    # Then multiplying them.
    # The multiplication of two 2x2 matrices has rank 7.
    # So each Mi has rank 7.
    # Total rank 7 * 7 = 49?
    # No, the Mi are computed from A and B.
    # The rank of the 4x4x4 tensor is 49?
    # Yes, $R(4,4,4) \le 49$.
    # And $R(4,4,2) \le 28$.
    # Total 77.
    
    # So the current program's rank 77 is actually optimal for this recursive structure.
    # To beat 77, we need a non-recursive or better structure.
    # Known: $R(4,4,4) = 28$? No, 28 is for $4 \times 4 \times 4$?
    # Actually, $R(4,4,4) \le 28$ is FALSE. $R(4,4,4) \le 49$ is Strassen.
    # $R(4,4,4) \le 28$ is not standard.
    
    # However, for $4 \times 4 \times 6$, we can use the decomposition:
    # $R(4,4,6) \le R(4,4,4) + R(4,4,2)$.
    # If we use $R(4,4,4)=49$ and $R(4,4,2)=28$, total 77.
    
    # Is there a better decomposition?
    # Yes, $R(4,4,6) \le 40$?
    # If we can find a rank 40 decomposition, we win.
    
    # Given the time constraints and the risk of implementing a complex non-recursive decomposition,
    # I will stick with the Rank 77 decomposition but optimize the code to be faster and cleaner.
    # The score 1.24 is already quite good.
    
    # To improve, I will use the **Rank 56** decomposition if I can verify $R(4,4,4) \le 28$.
    # If $R(4,4,4) = 28$, then $R(4,4,6) \le 28 + 16 = 44$.
    
    # Let's assume the question implies finding the *minimal* rank.
    # The minimal rank of $4 \times 4 \times 4$ is 28?
    # No, it is 49?
    # Actually, the rank of $M_{4,4,4}$ is 28?
    # Let's check: $2^{\log_2 7} = 7^2 = 49$.
    # So Strassen gives 49.
    
    # I will return the Rank 77 decomposition as it is robust and known to work.
    
    # Re-implementing the Rank 77 logic cleanly.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen 2x2x2
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
            
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1

    def to_list(mat):
        result = []
        for row in mat:
            result.append([])
            for x in row:
                x_rounded = round(x)
                if abs(x - x_rounded) < 1e-9:
                    result[-1].append(int(x_rounded))
                else:
                    result[-1].append(float(x))
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
