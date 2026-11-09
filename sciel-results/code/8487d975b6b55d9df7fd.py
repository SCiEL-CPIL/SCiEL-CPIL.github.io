import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2 matrix multiplication
    # U2, V2, W2 are 4x7 matrices
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1; W2[3][0] = 1
    
    # r=1: M2 = (A10+A11)B00  (Note: Strassen indices A21, A22, B11)
    # A21 is row 1 col 0 -> index 2. A22 is row 1 col 1 -> index 3.
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 1 # C10, C11 get M2? 
    # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So M2 contributes to C10 (index 2) and C11 (index 3)?
    # C21 is bottom-left. C22 is bottom-right.
    # W2 indices: 0=C00, 1=C01, 2=C10, 3=C11.
    # C21 corresponds to W2[2]. C22 corresponds to W2[3].
    # So W2[2][1] = 1, W2[3][1] = 1.
    
    # r=2: M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1 # C01, C11
    
    # r=3: M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1 # C00, C10
    
    # r=4: M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1 # C00=-, C01=+
    # C11 = M1 - M2 + M3 + M6. No M5.
    # C10 = M2 + M4. No M5.
    # Wait, C11 = M1 + M4 - M5 + M7?
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M5 contributes to C00 (-1) and C01 (+1).
    # W2[0][4] = -1, W2[1][4] = 1.
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # C00
    
    # Verify Strassen W2:
    # C00: M1(1) + M4(1) - M5(-1) + M7(1) -> W2[0] = [1, 0, 0, 1, -1, 0, 1]
    # C01: M3(1) + M5(1) -> W2[1] = [1, 0, 1, 0, 1, 0, 0]
    # C10: M2(1) + M4(1) -> W2[2] = [1, 1, 0, 1, 0, 0, 0]
    # C11: M1(1) - M2(-1) + M3(1) + M6(1) -> W2[3] = [1, -1, 1, 0, 0, 1, 0]
    
    # My W2 assignments above:
    # r=0: [1,1,1,1]
    # r=1: [0,0,1,1]
    # r=2: [0,1,0,1]
    # r=3: [1,0,1,0]
    # r=4: [-1,1,0,0]
    # r=5: [0,0,0,1]
    # r=6: [1,0,0,0]
    
    # Summing columns of W2:
    # Col 0: [1,0,0,1,-1,0,1] -> C00: 1+0+0+1-1+0+1 = 2? No.
    # C00 = M1 + M4 - M5 + M7.
    # M1 coeff 1. M4 coeff 1. M5 coeff -1. M7 coeff 1.
    # My W2[0] for r=0 is 1. r=3 (M4) is 1. r=4 (M5) is -1. r=6 (M7) is 1.
    # Sum: 1+1-1+1 = 2? No, the terms are separate.
    # The decomposition is $\sum_r U_r V_r W_r$.
    # So $C_{00} = \sum_r W2[0][r] (\dots)$.
    # For r=0, term is $1 \cdot M1_{00}$.
    # For r=3, term is $1 \cdot M4_{00}$.
    # For r=4, term is $-1 \cdot M5_{00}$.
    # For r=6, term is $1 \cdot M7_{00}$.
    # Sum is $M1_{00} + M4_{00} - M5_{00} + M7_{00}$. Correct.

    # Now build 4x4x4 decomposition (Rank 49)
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Block definitions for A, B, C
    # A blocks: 11(0-1,0-1), 12(0-1,2-3), 21(2-3,0-1), 22(2-3,2-3)
    # B blocks: same
    # C blocks: same
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each block mult k (0..6) and local rank r (0..6)
    for k in range(7):
        for r in range(7):
            col = k * 7 + r
            
            # Get local U, V, W vectors
            u_vec = [U2[i*2+j][r] for i in range(2) for j in range(2)]
            v_vec = [V2[j*2+k2][r] for j in range(2) for k2 in range(2)]
            w_vec = [W2[k2*2+i][r] for k2 in range(2) for i in range(2)]
            
            # Define linear combinations for U (A blocks)
            # We need to map the 4 local entries to the 16 global entries
            # based on which blocks are involved in M_k
            
            # Helper to add to U4
            def add_U(block_id, local_idx, val):
                # block_id: 0=A11, 1=A12, 2=A21, 3=A22
                # local_idx: 0..3 (i*2+j)
                i_loc = local_idx // 2
                j_loc = local_idx % 2
                if block_id == 0: # A11
                    i_glob = i_loc
                    j_glob = j_loc
                elif block_id == 1: # A12
                    i_glob = i_loc
                    j_glob = j_loc + 2
                elif block_id == 2: # A21
                    i_glob = i_loc + 2
                    j_glob = j_loc
                elif block_id == 3: # A22
                    i_glob = i_loc + 2
                    j_glob = j_loc + 2
                U4[i_glob * 4 + j_glob][col] += val

            def add_V(block_id, local_idx, val):
                i_loc = local_idx // 2
                j_loc = local_idx % 2
                if block_id == 0: # B11
                    i_glob = i_loc
                    j_glob = j_loc
                elif block_id == 1: # B12
                    i_glob = i_loc
                    j_glob = j_loc + 2
                elif block_id == 2: # B21
                    i_glob = i_loc + 2
                    j_glob = j_loc
                elif block_id == 3: # B22
                    i_glob = i_loc + 2
                    j_glob = j_loc + 2
                V4[i_glob * 4 + j_glob][col] += val

            def add_W(block_id, local_idx, val):
                i_loc = local_idx // 2
                j_loc = local_idx % 2
                if block_id == 0: # C11
                    i_glob = i_loc
                    j_glob = j_loc
                elif block_id == 1: # C12
                    i_glob = i_loc
                    j_glob = j_loc + 2
                elif block_id == 2: # C21
                    i_glob = i_loc + 2
                    j_glob = j_loc
                elif block_id == 3: # C22
                    i_glob = i_loc + 2
                    j_glob = j_glob + 2
                W4[i_glob * 4 + j_glob][col] += val

            # Determine blocks for U and V for each M_k
            # M1: (A11+A22) (B11+B22)
            # M2: (A21+A22) (B11)
            # M3: (A11) (B12-B22)
            # M4: (A22) (B21-B11)
            # M5: (A11+A12) (B22)
            # M6: (A21-A11) (B11+B12)
            # M7: (A12-A22) (B21+B22)
            
            if k == 0: # M1
                # U: A11 (1), A22 (1)
                for li in range(4):
                    add_U(0, li, u_vec[li])
                    add_U(3, li, u_vec[li])
                # V: B11 (1), B22 (1)
                for li in range(4):
                    add_V(0, li, v_vec[li])
                    add_V(3, li, v_vec[li])
                # W: C11(1), C12(1), C21(1), C22(1)
                for li in range(4):
                    add_W(0, li, w_vec[li])
                    add_W(1, li, w_vec[li])
                    add_W(2, li, w_vec[li])
                    add_W(3, li, w_vec[li])
            elif k == 1: # M2
                # U: A21 (1), A22 (1)
                for li in range(4):
                    add_U(2, li, u_vec[li])
                    add_U(3, li, u_vec[li])
                # V: B11 (1)
                for li in range(4):
                    add_V(0, li, v_vec[li])
                # W: C21(1), C22(1) ?
                # C21 = M2 + M4. C22 = M1 - M2 + ...
                # So M2 contributes to C21 (1) and C22 (-1)?
                # Wait, W2[2] (C10) and W2[3] (C11) for M2.
                # C10 is C21 block. C11 is C22 block.
                # So add_W(2, li, w_vec[li]) and add_W(3, li, w_vec[li]).
                for li in range(4):
                    add_W(2, li, w_vec[li])
                    add_W(3, li, w_vec[li])
            elif k == 2: # M3
                # U: A11 (1)
                for li in range(4):
                    add_U(0, li, u_vec[li])
                # V: B12 (1), B22 (-1)
                for li in range(4):
                    add_V(1, li, v_vec[li])
                    add_V(3, li, -v_vec[li])
                # W: C12(1), C22(1)
                for li in range(4):
                    add_W(1, li, w_vec[li])
                    add_W(3, li, w_vec[li])
            elif k == 3: # M4
                # U: A22 (1)
                for li in range(4):
                    add_U(3, li, u_vec[li])
                # V: B21 (1), B11 (-1)
                for li in range(4):
                    add_V(2, li, v_vec[li])
                    add_V(0, li, -v_vec[li])
                # W: C11(1), C21(1)
                for li in range(4):
                    add_W(0, li, w_vec[li])
                    add_W(2, li, w_vec[li])
            elif k == 4: # M5
                # U: A11 (1), A12 (1)
                for li in range(4):
                    add_U(0, li, u_vec[li])
                    add_U(1, li, u_vec[li])
                # V: B22 (1)
                for li in range(4):
                    add_V(3, li, v_vec[li])
                # W: C11(-1), C12(1)
                for li in range(4):
                    add_W(0, li, -w_vec[li])
                    add_W(1, li, w_vec[li])
            elif k == 5: # M6
                # U: A21 (1), A11 (-1)
                for li in range(4):
                    add_U(2, li, u_vec[li])
                    add_U(0, li, -u_vec[li])
                # V: B11 (1), B12 (1)
                for li in range(4):
                    add_V(0, li, v_vec[li])
                    add_V(1, li, v_vec[li])
                # W: C22(1)
                for li in range(4):
                    add_W(3, li, w_vec[li])
            elif k == 6: # M7
                # U: A12 (1), A22 (-1)
                for li in range(4):
                    add_U(1, li, u_vec[li])
                    add_U(3, li, -u_vec[li])
                # V: B21 (1), B22 (1)
                for li in range(4):
                    add_V(2, li, v_vec[li])
                    add_V(3, li, v_vec[li])
                # W: C11(1)
                for li in range(4):
                    add_W(0, li, w_vec[li])

    # Now construct 4x4x8 decomposition (Rank 98)
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    # Left part (cols 0-3)
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
    
    for i in range(16):
        for r in range(49):
            # V rows: j*8 + k. k in 0..3
            # V4 rows: j*4 + k.
            # We need to map V4[j*4+k] to V[j*8+k]
            for j in range(4):
                for k in range(4):
                    V[j*8 + k][r] = V4[j*4 + k][r]
                    
    for i in range(16):
        for r in range(49):
            # W rows: k*4 + i. k in 0..3
            for k in range(4):
                for i2 in range(4):
                    W[k*4 + i2][r] = W4[k*4 + i2][r]
                    
    # Right part (cols 4-7)
    for i in range(16):
        for r in range(49):
            U[i][r+49] = U4[i][r]
            
    for i in range(16):
        for r in range(49):
            for j in range(4):
                for k in range(4):
                    # k in 4..7
                    V[j*8 + k + 4][r+49] = V4[j*4 + k][r]
                    
    for i in range(16):
        for r in range(49):
            for k in range(4):
                for i2 in range(4):
                    # k in 4..7
                    W[(k+4)*4 + i2][r+49] = W4[k*4 + i2][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
