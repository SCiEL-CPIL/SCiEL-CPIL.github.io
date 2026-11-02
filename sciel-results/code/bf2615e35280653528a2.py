"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 65.
    # Part 1: 4x4x4 multiplication using recursive Strassen (Rank 49)
    # Part 2: 4x4x1 multiplication using schoolbook (Rank 16)
    
    # Helper to add a term to U, V, W
    # U is list of lists, we will use numpy arrays for construction and convert at end?
    # Or just lists.
    
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Recursive Strassen for 4x4x4 ---
    # We need to generate the factors for the 4x4x4 tensor.
    # The tensor is T[i,j,k] = 1 for i,j,k in 0..3.
    # We decompose it into 7 blocks of 2x2x2.
    
    # Strassen 2x2x2 factors (Rank 7)
    # Inputs: A (2x2), B (2x2)
    # U indices: (i, j) for i,j in 0,1
    # V indices: (j, k) for j,k in 0,1
    # W indices: (k, i) for k,i in 0,1
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We will define the 2x2x2 Strassen decomposition explicitly.
    # Each term r in 0..6 corresponds to one of the 7 multiplications.
    # For each term, we define U, V, W entries for the 2x2x2 sub-tensor.
    
    # Sub-tensor indices are local (0,1).
    # We need to map these to the global 4x4x4 indices.
    
    # The 4x4 matrices are split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 7 multiplications operate on the blocks.
    # M1 = (A00+A11)(B00+B11)
    # This means we need to compute the product of two 2x2 matrices.
    # Let X = A00+A11, Y = B00+B11.
    # X is a 2x2 matrix, Y is a 2x2 matrix.
    # We use Strassen 2x2x2 to compute Z = X*Y.
    # Then we add Z to C00 and C11.
    
    # This recursive structure is hard to flatten into a single set of U, V, W factors directly without tracking the linear combinations.
    # However, we can just generate the factors for the 7 sub-problems and then combine them.
    
    # Let's define a function to get the factors for a 2x2x2 Strassen multiplication.
    def get_strassen_2x2_factors():
        # Returns U, V, W for 2x2x2, each of shape (4, 7)
        # U: (i,j), V: (j,k), W: (k,i)
        U_sub = [[0]*7 for _ in range(4)]
        V_sub = [[0]*7 for _ in range(4)]
        W_sub = [[0]*7 for _ in range(4)]
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1) -- Wait, M1 contributes to C00 and C11.
        # But Strassen 2x2x2 computes ONE 2x2 matrix product.
        # The output is a 2x2 matrix Z.
        # The decomposition of the 2x2x2 tensor T_sub[i,j,k]=1 is:
        # T_sub = sum_r u_r v_r w_r
        # For Strassen, the factors are:
        
        # r=0: M1
        # U: A00 (1), A11 (1) -> U[0,0]=1, U[3,0]=1? No.
        # Indices:
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (k,i order? W is p*n = 4. Index k*n+i)
        # W index: k*2 + i.
        # C00: k=0,i=0 -> 0
        # C01: k=0,i=1 -> 1
        # C10: k=1,i=0 -> 2
        # C11: k=1,i=1 -> 3
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00 (idx 0) = 1, A11 (idx 3) = 1
        # V: B00 (idx 0) = 1, B11 (idx 3) = 1
        # W: C00 (idx 0) = 1, C11 (idx 3) = 1
        U_sub[0][0] = 1; U_sub[3][0] = 1
        V_sub[0][0] = 1; V_sub[3][0] = 1
        W_sub[0][0] = 1; W_sub[3][0] = 1
        
        # M2 = (A10+A11)B00
        # U: A10 (idx 2) = 1, A11 (idx 3) = 1
        # V: B00 (idx 0) = 1
        # W: C10 (idx 2) = 1
        U_sub[2][1] = 1; U_sub[3][1] = 1
        V_sub[0][1] = 1
        W_sub[2][1] = 1
        
        # M3 = A00(B01-B11)
        # U: A00 (idx 0) = 1
        # V: B01 (idx 1) = 1, B11 (idx 3) = -1
        # W: C01 (idx 1) = 1
        U_sub[0][2] = 1
        V_sub[1][2] = 1; V_sub[3][2] = -1
        W_sub[1][2] = 1
        
        # M4 = A11(B10-B00)
        # U: A11 (idx 3) = 1
        # V: B10 (idx 2) = 1, B00 (idx 0) = -1
        # W: C00 (idx 0) = 1, C10 (idx 2) = 1
        U_sub[3][3] = 1
        V_sub[2][3] = 1; V_sub[0][3] = -1
        W_sub[0][3] = 1; W_sub[2][3] = 1
        
        # M5 = (A00+A01)B11
        # U: A00 (idx 0) = 1, A01 (idx 1) = 1
        # V: B11 (idx 3) = 1
        # W: C00 (idx 0) = -1, C11 (idx 3) = 1
        U_sub[0][4] = 1; U_sub[1][4] = 1
        V_sub[3][4] = 1
        W_sub[0][4] = -1; W_sub[3][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # U: A10 (idx 2) = 1, A00 (idx 0) = -1
        # V: B00 (idx 0) = 1, B01 (idx 1) = 1
        # W: C11 (idx 3) = 1
        U_sub[2][5] = 1; U_sub[0][5] = -1
        V_sub[0][5] = 1; V_sub[1][5] = 1
        W_sub[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01 (idx 1) = 1, A11 (idx 3) = -1
        # V: B10 (idx 2) = 1, B11 (idx 3) = 1
        # W: C00 (idx 0) = 1, C01 (idx 1) = 1
        U_sub[1][6] = 1; U_sub[3][6] = -1
        V_sub[2][6] = 1; V_sub[3][6] = 1
        W_sub[0][6] = 1; W_sub[1][6] = 1
        
        return U_sub, V_sub, W_sub

    U_2, V_2, W_2 = get_strassen_2x2_factors()
    
    # Now, construct the 4x4x4 factors using 7 instances of the 2x2x2 factors.
    # We need to map the local indices (0,1) of the blocks to the global indices (0..3).
    
    # Block mapping for A (4x4):
    # A00: i in 0,1; j in 0,1 -> global i, j
    # A01: i in 0,1; j in 2,3 -> global i, j+2
    # A10: i in 2,3; j in 0,1 -> global i+2, j
    # A11: i in 2,3; j in 2,3 -> global i+2, j+2
    
    # Block mapping for B (4x4):
    # B00: j in 0,1; k in 0,1 -> global j, k
    # B01: j in 0,1; k in 2,3 -> global j, k+2
    # B10: j in 2,3; k in 0,1 -> global j+2, k
    # B11: j in 2,3; k in 2,3 -> global j+2, k+2
    
    # Block mapping for C (4x4):
    # C00: k in 0,1; i in 0,1 -> global k, i
    # C01: k in 0,1; i in 2,3 -> global k, i+2
    # C10: k in 2,3; i in 0,1 -> global k+2, i
    # C11: k in 2,3; i in 2,3 -> global k+2, i+2
    
    # The 7 Strassen multiplications:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00) -> C10 += M2
    # M3 = (A00)(B01-B11) -> C01 += M3
    # M4 = (A11)(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)(B11) -> C00 -= M5, C11 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C01 += M7
    
    # We will iterate through the 7 multiplications.
    # For each multiplication, we use the 7 factors from U_2, V_2, W_2.
    # Total rank for 4x4x4 is 7 * 7 = 49.
    
    current_rank = 0
    
    # Define the block offsets
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # For each Strassen term s in 0..6:
    #   For each sub-factor r in 0..6:
    #     global_rank = current_rank + s*7 + r
    
    # We need to define the linear combinations for U, V, W for each of the 7 Strassen terms.
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00 (1), A11 (1)
    # V: B00 (1), B11 (1)
    # W: C00 (1), C11 (1)
    
    # Term 1: M2 = (A10+A11)(B00)
    # U: A10 (1), A11 (1)
    # V: B00 (1)
    # W: C10 (1)
    
    # Term 2: M3 = (A00)(B01-B11)
    # U: A00 (1)
    # V: B01 (1), B11 (-1)
    # W: C01 (1)
    
    # Term 3: M4 = (A11)(B10-B00)
    # U: A11 (1)
    # V: B10 (1), B00 (-1)
    # W: C00 (1), C10 (1)
    
    # Term 4: M5 = (A00+A01)(B11)
    # U: A00 (1), A01 (1)
    # V: B11 (1)
    # W: C00 (-1), C11 (1)
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # U: A10 (1), A00 (-1)
    # V: B00 (1), B01 (1)
    # W: C11 (1)
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # U: A01 (1), A11 (-1)
    # V: B10 (1), B11 (1)
    # W: C00 (1), C01 (1)
    
    # We will process each term.
    
    def add_strassen_term(term_idx, u_blocks, v_blocks, w_blocks):
        # u_blocks: list of (block_idx, coeff)
        # v_blocks: list of (block_idx, coeff)
        # w_blocks: list of (block_idx, coeff)
        nonlocal current_rank
        
        for r in range(7):
            global_r = current_rank + r
            
            # Process U
            for local_i in range(2):
                for local_j in range(2):
                    # U_2 index: local_i * 2 + local_j
                    u_val = U_2[local_i * 2 + local_j][r]
                    if u_val != 0:
                        for b_idx, coeff in u_blocks:
                            # Map local i,j to global i,j
                            # Block 00: i, j
                            # Block 01: i, j+2
                            # Block 10: i+2, j
                            # Block 11: i+2, j+2
                            if b_idx == 0:
                                gi, gj = local_i, local_j
                            elif b_idx == 1:
                                gi, gj = local_i, local_j + 2
                            elif b_idx == 2:
                                gi, gj = local_i + 2, local_j
                            elif b_idx == 3:
                                gi, gj = local_i + 2, local_j + 2
                            else:
                                continue
                            U[gi * m + gj][global_r] += u_val * coeff
                            
            # Process V
            for local_j in range(2):
                for local_k in range(2):
                    # V_2 index: local_j * 2 + local_k
                    v_val = V_2[local_j * 2 + local_k][r]
                    if v_val != 0:
                        for b_idx, coeff in v_blocks:
                            # Block 00: j, k
                            # Block 01: j, k+2
                            # Block 10: j+2, k
                            # Block 11: j+2, k+2
                            if b_idx == 0:
                                gj, gk = local_j, local_k
                            elif b_idx == 1:
                                gj, gk = local_j, local_k + 2
                            elif b_idx == 2:
                                gj, gk = local_j + 2, local_k
                            elif b_idx == 3:
                                gj, gk = local_j + 2, local_k + 2
                            else:
                                continue
                            V[gj * p + gk][global_r] += v_val * coeff
                            
            # Process W
            for local_k in range(2):
                for local_i in range(2):
                    # W_2 index: local_k * 2 + local_i
                    w_val = W_2[local_k * 2 + local_i][r]
                    if w_val != 0:
                        for b_idx, coeff in w_blocks:
                            # Block 00: k, i
                            # Block 01: k, i+2
                            # Block 10: k+2, i
                            # Block 11: k+2, i+2
                            if b_idx == 0:
                                gk, gi = local_k, local_i
                            elif b_idx == 1:
                                gk, gi = local_k, local_i + 2
                            elif b_idx == 2:
                                gk, gi = local_k + 2, local_i
                            elif b_idx == 3:
                                gk, gi = local_k + 2, local_i + 2
                            else:
                                continue
                            W[gk * n + gi][global_r] += w_val * coeff
                            
        current_rank += 7

    # Term 0
    add_strassen_term(0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    # Term 1
    add_strassen_term(1, [(2, 1), (3, 1)], [(0, 1)], [(2, 1)])
    # Term 2
    add_strassen_term(2, [(0, 1)], [(1, 1), (3, -1)], [(1, 1)])
    # Term 3
    add_strassen_term(3, [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
    # Term 4
    add_strassen_term(4, [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (3, 1)])
    # Term 5
    add_strassen_term(5, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    # Term 6
    add_strassen_term(6, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)])
    
    # --- Part 2: Schoolbook for 4x4x1 (k=4) ---
    # i in 0..3, j in 0..3, k=4
    # U[i*m+j, r] = 1
    # V[j*p+4, r] = 1
    # W[4*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            r = rank_444 + i * m + j
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
