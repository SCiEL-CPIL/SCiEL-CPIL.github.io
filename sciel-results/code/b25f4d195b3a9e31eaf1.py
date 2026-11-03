import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # 1. Build Schoolbook for the tail part (k=4,5,6)
    # Rank contribution: 4 * 4 * 3 = 48
    # Indices for tail:
    # U: i*m + j (i in 0..3, j in 0..3)
    # V: j*p + k (j in 0..3, k in 4..6)
    # W: k*n + i (k in 4..6, i in 0..3)
    
    tail_rank = n * m * 3  # 48
    
    # 2. Build Strassen for the head part (k=0,1,2,3)
    # We split the 4x4x4 multiplication into 4 blocks of 2x2x2?
    # No, standard Strassen for 4x4 applies to the matrix entries.
    # Let's use the explicit tensor decomposition for 4x4x4 based on 2x2 Strassen.
    # The 4x4 matrix A can be viewed as 2x2 blocks of 2x2 matrices?
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # But matrix multiplication is linear.
    # If we use Strassen on the 2x2 block structure, we need 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7 (Strassen).
    # So total rank for 4x4x4 is 7 * 7 = 49?
    # Let's verify.
    # The tensor T_4x4x4 = sum_{block_mult} T_{block_mult}.
    # Actually, this is correct. The rank is 49.
    
    head_rank = 49
    
    total_rank = tail_rank + head_rank  # 97
    
    # Initialize arrays
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 97
    
    # --- Tail Part (Schoolbook) ---
    # k in 4, 5, 6
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    # col should be 48 now
    
    # --- Head Part (Strassen 4x4x4) ---
    # We need to construct the 49 terms for k in 0..3.
    # The 4x4 matrix multiplication can be decomposed by treating it as multiplication of 2x2 blocks of 2x2 matrices.
    # Let A be 4x4, B be 4x4.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]] where Aij, Bkl are 2x2.
    # C = AB = [[C00, C01], [C10, C11]].
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    #
    # Using Strassen on the block level (2x2 blocks):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mj is a product of 2x2 matrices.
    # Let Xj be the first factor, Yj the second factor.
    # Each Mj = Xj * Yj has rank 7 (Strassen 2x2).
    # So we have 7 terms, each contributing rank 7. Total 49.
    #
    # We need to map the indices of the 2x2 Strassen decomposition to the global 4x4 indices.
    #
    # Strassen 2x2 decomposition details:
    # For a 2x2 matrix multiplication P = Q * R:
    # s1 = (q00+q11)(r00+r11)
    # s2 = (q10+q11)r00
    # s3 = q00(r01-r11)
    # s4 = q11(r10-r00)
    # s5 = (q00+q01)r11
    # s6 = (q10-q00)(r00+r01)
    # s7 = (q01-q11)(r10+r11)
    #
    # p00 = s1 + s4 - s5 + s7
    # p01 = s3 + s5
    # p10 = s2 + s4
    # p11 = s1 - s2 + s3 + s6
    #
    # The tensor T_2x2 for s_t (t=1..7) has:
    # U vector u_t (size 4), V vector v_t (size 4), W vector w_t (size 4).
    # u_t is the coefficient vector for Q entries.
    # v_t is the coefficient vector for R entries.
    # w_t is the coefficient vector for P entries.
    
    # Let's define the 7 vectors for 2x2 Strassen.
    # Indices for 2x2:
    # Q: (0,0), (0,1), (1,0), (1,1) -> idx 0,1,2,3
    # R: (0,0), (0,1), (1,0), (1,1) -> idx 0,1,2,3
    # P: (0,0), (0,1), (1,0), (1,1) -> idx 0,1,2,3
    
    # s1 = (q00+q11)(r00+r11)
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # w1 = [1, 0, 0, 1] (since p00+=1, p11+=1) -> wait, p00 = s1... p11 = ... - s1?
    # p00 = s1 + ...
    # p11 = s1 - ...
    # So w1 has 1 at idx 0 (p00) and 1 at idx 3 (p11)?
    # Let's check the formula again.
    # p00 = s1 + s4 - s5 + s7
    # p11 = s1 - s2 + s3 + s6
    # So s1 contributes to p00 and p11.
    # w1 = [1, 0, 0, 1]
    
    u_2x2 = [
        [1, 0, 0, 1],  # s1: (q00+q11)
        [0, 0, 1, 1],  # s2: (q10+q11)
        [1, 0, 0, 0],  # s3: q00
        [0, 0, 0, 1],  # s4: q11
        [1, 1, 0, 0],  # s5: (q00+q01)
        [-1, 0, 1, 0], # s6: (q10-q00)
        [0, 1, 0, -1]  # s7: (q01-q11)
    ]
    
    v_2x2 = [
        [1, 0, 0, 1],  # s1: (r00+r11)
        [1, 0, 0, 0],  # s2: r00
        [0, 1, 0, -1], # s3: (r01-r11)
        [0, 0, 1, -1], # s4: (r10-r00)
        [0, 0, 0, 1],  # s5: r11
        [1, 1, 0, 0],  # s6: (r00+r01)
        [0, 0, 1, 1]   # s7: (r10+r11)
    ]
    
    w_2x2 = [
        [1, 0, 0, 1],  # s1: p00, p11
        [0, 0, 1, -1], # s2: p10, p11
        [0, 1, 0, 1],  # s3: p01, p11
        [1, 0, 1, 0],  # s4: p00, p10
        [-1, 1, 0, 0], # s5: p00, p01
        [0, 0, 0, 1],  # s6: p11
        [1, 0, 0, 0]   # s7: p00
    ]
    
    # Now, we define the 7 block multiplications for the 4x4 case.
    # Block indices:
    # A blocks: A00 (top-left), A01 (top-right), A10 (bottom-left), A11 (bottom-right)
    # B blocks: B00 (top-left), B01 (top-right), B10 (bottom-left), B11 (bottom-right)
    # Global indices for 4x4:
    # A[i,j] where i,j in 0..3.
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # Define the 7 M terms.
    # Each term is defined by:
    # X_block: which A blocks are involved and with what signs.
    # Y_block: which B blocks are involved and with what signs.
    # Z_map: how the resulting 2x2 product maps to the 4x4 C blocks.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let's define a function to add a Strassen term to the global tensors.
    # Args:
    #   x_blocks: list of (block_idx, coeff) for A. block_idx: 0=A00, 1=A01, 2=A10, 3=A11.
    #   y_blocks: list of (block_idx, coeff) for B.
    #   z_map: list of (c_block_idx, coeff) for C.
    #   col_offset: starting column in U, V, W.
    
    def add_strassen_term(x_blocks, y_blocks, z_map, col_offset, n, m, p):
        # We have 7 sub-terms (s1..s7) for the 2x2 multiplication.
        for s in range(7):
            u_s = u_2x2[s]
            v_s = v_2x2[s]
            w_s = w_2x2[s]
            
            # Construct global U vector for this sub-term
            # U is size 16 (4x4).
            # We need to place the 2x2 pattern u_s into the appropriate 2x2 blocks.
            # The block is a 2x2 submatrix.
            # For A, the blocks are 2x2.
            # A00: rows 0-1, cols 0-1.
            # A01: rows 0-1, cols 2-3.
            # A10: rows 2-3, cols 0-1.
            # A11: rows 2-3, cols 2-3.
            
            for bi, bc in x_blocks:
                if bc == 0: continue
                # Determine row and col offsets for the block
                if bi == 0: # A00
                    r0, c0 = 0, 0
                elif bi == 1: # A01
                    r0, c0 = 0, 2
                elif bi == 2: # A10
                    r0, c0 = 2, 0
                else: # bi == 3 # A11
                    r0, c0 = 2, 2
                
                # Add u_s scaled by bc to the block
                for dr in range(2):
                    for dc in range(2):
                        global_r = r0 + dr
                        global_c = c0 + dc
                        global_idx = global_r * m + global_c # m=4
                        U[global_idx][col_offset + s] += u_s[dr*2+dc] * bc
            
            # Construct global V vector for this sub-term
            # V is size 28 (4x7).
            # But this term only involves k=0..3 (head part).
            # B blocks are 2x2, cols 0-1 (B00, B10) and 2-3 (B01, B11).
            # B00: rows 0-1, cols 0-1. (j in 0,1; k in 0,1)
            # B01: rows 0-1, cols 2-3. (j in 0,1; k in 2,3)
            # B10: rows 2-3, cols 0-1. (j in 2,3; k in 0,1)
            # B11: rows 2-3, cols 2-3. (j in 2,3; k in 2,3)
            
            for bi, bc in y_blocks:
                if bc == 0: continue
                if bi == 0: # B00
                    r0, c0 = 0, 0
                elif bi == 1: # B01
                    r0, c0 = 0, 2
                elif bi == 2: # B10
                    r0, c0 = 2, 0
                else: # bi == 3 # B11
                    r0, c0 = 2, 2
                
                # Add v_s scaled by bc to the block
                for dr in range(2):
                    for dc in range(2):
                        global_r = r0 + dr # j index
                        global_c = c0 + dc # k index (0..3)
                        global_idx = global_r * p + global_c # p=7
                        V[global_idx][col_offset + s] += v_s[dr*2+dc] * bc
            
            # Construct global W vector for this sub-term
            # W is size 28 (7x4).
            # C blocks:
            # C00: rows 0-1, cols 0-1. (k in 0,1; i in 0,1) -> W index k*n+i
            # C01: rows 2-3, cols 0-1. (k in 2,3; i in 0,1)
            # C10: rows 0-1, cols 2-3. (k in 0,1; i in 2,3)
            # C11: rows 2-3, cols 2-3. (k in 2,3; i in 2,3)
            
            for bi, bc in z_map:
                if bc == 0: continue
                if bi == 0: # C00
                    r0, c0 = 0, 0 # k offset, i offset
                elif bi == 1: # C01
                    r0, c0 = 2, 0
                elif bi == 2: # C10
                    r0, c0 = 0, 2
                else: # bi == 3 # C11
                    r0, c0 = 2, 2
                
                # Add w_s scaled by bc to the block
                # w_s is for P (C block). P is 2x2.
                # P entries: (0,0), (0,1), (1,0), (1,1).
                # C block indices:
                # r in 0,1 corresponds to local row of P (which is k local).
                # c in 0,1 corresponds to local col of P (which is i local).
                for dr in range(2):
                    for dc in range(2):
                        global_r = r0 + dr # k index
                        global_c = c0 + dc # i index
                        global_idx = global_r * n + global_c # n=4
                        W[global_idx][col_offset + s] += w_s[dr*2+dc] * bc
    
    # Define the 7 M terms for 4x4x4
    # Block indices: 0:00, 1:01, 2:10, 3:11
    
    # M1 = (A00+A11)(B00+B11)
    # C00 = M1 + ...
    # C11 = M1 - ...
    # So M1 contributes to C00 and C11.
    # Wait, the Strassen formula gives C00, C01, C10, C11 in terms of M1..M7.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each Mj, we need to specify which C blocks it contributes to and with what coefficient.
    
    # M1:
    x1 = [(0, 1), (3, 1)]  # A00 + A11
    y1 = [(0, 1), (3, 1)]  # B00 + B11
    z1 = [(0, 1), (3, 1)]  # C00 + C11
    
    # M2:
    x2 = [(2, 1), (3, 1)]  # A10 + A11
    y2 = [(0, 1)]          # B00
    z2 = [(2, 1), (3, -1)] # C10 - C11 ? 
    # Wait.
    # C10 = M2 + M4. So M2 -> C10 (+1).
    # C11 = M1 - M2 + M3 + M6. So M2 -> C11 (-1).
    # So z2 = [(2, 1), (3, -1)]
    
    # M3:
    x3 = [(0, 1)]          # A00
    y3 = [(1, 1), (3, -1)] # B01 - B11
    z3 = [(1, 1), (3, 1)]  # C01 + C11
    # C01 = M3 + M5. So M3 -> C01 (+1).
    # C11 = ... + M3. So M3 -> C11 (+1).
    
    # M4:
    x4 = [(3, 1)]          # A11
    y4 = [(2, 1), (0, -1)] # B10 - B00
    z4 = [(0, 1), (2, 1)]  # C00 + C10
    # C00 = M1 + M4 - M5 + M7. So M4 -> C00 (+1).
    # C10 = M2 + M4. So M4 -> C10 (+1).
    
    # M5:
    x5 = [(0, 1), (1, 1)]  # A00 + A01
    y5 = [(3, 1)]          # B11
    z5 = [(0, -1), (1, 1)] # -C00 + C01
    # C00 = ... - M5. So M5 -> C00 (-1).
    # C01 = M3 + M5. So M5 -> C01 (+1).
    
    # M6:
    x6 = [(2, 1), (0, -1)] # A10 - A00
    y6 = [(0, 1), (1, 1)]  # B00 + B01
    z6 = [(3, 1)]          # C11
    # C11 = ... + M6. So M6 -> C11 (+1).
    
    # M7:
    x7 = [(1, 1), (3, -1)] # A01 - A11
    y7 = [(2, 1), (3, 1)]  # B10 + B11
    z7 = [(0, 1)]          # C00
    # C00 = ... + M7. So M7 -> C00 (+1).
    
    terms = [
        (x1, y1, z1),
        (x2, y2, z2),
        (x3, y3, z3),
        (x4, y4, z4),
        (x5, y5, z5),
        (x6, y6, z6),
        (x7, y7, z7)
    ]
    
    for t_idx, (x_t, y_t, z_t) in enumerate(terms):
        col_start = 48 + t_idx * 7
        add_strassen_term(x_t, y_t, z_t, col_start, n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
