import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def strassen_2x2():
    """Return U, V, W for 2x2 x 2x2 matrix multiplication, rank 7."""
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # S1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += 1, C11 += 1
    
    # S2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C10 += 1, C11 -= 1
    
    # S3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01 += 1, C11 += 1
    
    # S4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += 1, C10 += 1
    
    # S5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00 -= 1, C01 += 1
    
    # S6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += 1
    
    # S7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = -1  # C00 += 1, C11 -= 1
    
    return U, V, W

def build_4x4x4x4():
    """Build rank-49 decomposition for 4x4 x 4x4."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = strassen_2x2()
    
    # Block-level Strassen combinations
    # A blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    
    # For each M_t, define the A_comb and B_comb as linear combinations
    # A_comb is a 2x2 matrix, each entry is a linear combo of A entries
    # B_comb is a 2x2 matrix, each entry is a linear combo of B entries
    
    # M1: A_comb = A11+A22, B_comb = B11+B22
    # M2: A_comb = A21+A22, B_comb = B11
    # M3: A_comb = A11, B_comb = B12-B22
    # M4: A_comb = A22, B_comb = B21-B11
    # M5: A_comb = A11+A12, B_comb = B22
    # M6: A_comb = A21-A11, B_comb = B11+B12
    # M7: A_comb = A12-A22, B_comb = B21+B22
    
    # A entry indices in the 4x4 matrix (row-major):
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
    # A[3,0]=12, A[3,1]=13, A[3,2]=14, A[3,3]=15
    
    # A11: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # A12: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # A21: A[2,0]=8, A[2,1]=9, A[3,0]=12, A[3,1]=13
    # A22: A[2,2]=10, A[2,3]=11, A[3,2]=14, A[3,3]=15
    
    # B entry indices in the 4x4 matrix (row-major):
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    
    # B11: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # B12: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # B21: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # B22: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    
    # C entry indices (row-major for 4x4 output):
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3
    # C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
    # C[2,0]=8, C[2,1]=9, C[2,2]=10, C[2,3]=11
    # C[3,0]=12, C[3,1]=13, C[3,2]=14, C[3,3]=15
    
    # C11: C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    # C12: C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7
    # C21: C[2,0]=8, C[2,1]=9, C[3,0]=12, C[3,1]=13
    # C22: C[2,2]=10, C[2,3]=11, C[3,2]=14, C[3,3]=15
    
    # Define A_comb for each M_t: dict from 2x2 local index to list of (A_global_index, coeff)
    # 2x2 local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    A_comb = {}
    B_comb = {}
    
    # M1: A11+A22
    A_comb[0] = {0: [(0,1),(10,1)], 1: [(1,1),(11,1)], 2: [(4,1),(14,1)], 3: [(5,1),(15,1)]}
    B_comb[0] = {0: [(0,1),(10,1)], 1: [(1,1),(11,1)], 2: [(4,1),(14,1)], 3: [(5,1),(15,1)]}
    
    # M2: A21+A22, B11
    A_comb[1] = {0: [(8,1),(10,1)], 1: [(9,1),(11,1)], 2: [(12,1),(14,1)], 3: [(13,1),(15,1)]}
    B_comb[1] = {0: [(0,1)], 1: [(1,1)], 2: [(4,1)], 3: [(5,1)]}
    
    # M3: A11, B12-B22
    A_comb[2] = {0: [(0,1)], 1: [(1,1)], 2: [(4,1)], 3: [(5,1)]}
    B_comb[2] = {0: [(2,1),(10,-1)], 1: [(3,1),(11,-1)], 2: [(6,1),(14,-1)], 3: [(7,1),(15,-1)]}
    
    # M4: A22, B21-B11
    A_comb[3] = {0: [(10,1)], 1: [(11,1)], 2: [(14,1)], 3: [(15,1)]}
    B_comb[3] = {0: [(8,1),(0,-1)], 1: [(9,1),(1,-1)], 2: [(12,1),(4,-1)], 3: [(13,1),(5,-1)]}
    
    # M5: A11+A12, B22
    A_comb[4] = {0: [(0,1),(2,1)], 1: [(1,1),(3,1)], 2: [(4,1),(6,1)], 3: [(5,1),(7,1)]}
    B_comb[4] = {0: [(10,1)], 1: [(11,1)], 2: [(14,1)], 3: [(15,1)]}
    
    # M6: A21-A11, B11+B12
    A_comb[5] = {0: [(8,1),(0,-1)], 1: [(9,1),(1,-1)], 2: [(12,1),(4,-1)], 3: [(13,1),(5,-1)]}
    B_comb[5] = {0: [(0,1),(2,1)], 1: [(1,1),(3,1)], 2: [(4,1),(6,1)], 3: [(5,1),(7,1)]}
    
    # M7: A12-A22, B21+B22
    A_comb[6] = {0: [(2,1),(10,-1)], 1: [(3,1),(11,-1)], 2: [(6,1),(14,-1)], 3: [(7,1),(15,-1)]}
    B_comb[6] = {0: [(8,1),(10,1)], 1: [(9,1),(11,1)], 2: [(12,1),(14,1)], 3: [(13,1),(15,1)]}
    
    # C block mapping for each M_t
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each M_t, the 2x2 result maps to a specific C block with a sign
    C_block_map = {}
    # M1 -> C11, sign +1
    C_block_map[0] = [(0, 1), (1, 1), (4, 1), (5, 1)]  # C[0,0],C[0,1],C[1,0],C[1,1]
    # M2 -> C21, sign +1
    C_block_map[1] = [(8, 1), (9, 1), (12, 1), (13, 1)]
    # M3 -> C12, sign +1
    C_block_map[2] = [(2, 1), (3, 1), (6, 1), (7, 1)]
    # M4 -> C11, sign +1 AND C21, sign +1
    # Wait, M4 appears in both C11 and C21!
    # C11 = M1 + M4 - M5 + M7
    # C21 = M2 + M4
    # So M4 contributes to both C11 and C21.
    
    # This means I can't just assign one C block per M_t. I need to handle the fact that
    # some M_t contribute to multiple C blocks.
    
    # Let me reconsider. The Strassen decomposition computes:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_t is a 2x2 matrix. The W matrix for the full decomposition needs to
    # distribute the 2x2 result of each M_t to the appropriate C blocks.
    
    # For M1: contributes +1 to C11 (top-left 2x2) and +1 to C22 (bottom-right 2x2)
    #   Wait, no. Let me re-read.
    #   C11 = M1 + M4 - M5 + M7  (M1 is a 2x2 matrix, added to C11)
    #   C22 = M1 - M2 + M3 + M6  (M1 is also added to C22)
    #   So M1's 2x2 result is added to BOTH C11 and C22.
    
    # This complicates things. The W matrix for M1 needs to have entries in both
    # the C11 block and the C22 block.
    
    # Let me define the W contributions more carefully.
    
    # For each M_t, the 2x2 result R = [[r00, r01], [r10, r11]] contributes:
    # - To C11: R (if M_t is in C11's formula)
    # - To C12: R (if M_t is in C12's formula)
    # - To C21: R (if M_t is in C21's formula)
    # - To C22: R (if M_t is in C22's formula)
    
    # With signs.
    
    # M1: C11 (+1), C22 (+1)
    # M2: C21 (+1), C22 (-1)
    # M3: C12 (+1), C22 (+1)
    # M4: C11 (+1), C21 (+1)
    # M5: C11 (-1), C12 (+1)
    # M6: C22 (+1)
    # M7: C11 (+1)
    
    # C block global indices:
    # C11: C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    # C12: C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7
    # C21: C[2,0]=8, C[2,1]=9, C[3,0]=12, C[3,1]=13
    # C22: C[2,2]=10, C[2,3]=11, C[3,2]=14, C[3,3]=15
    
    C11_idx = [0, 1, 4, 5]
    C12_idx = [2, 3, 6, 7]
    C21_idx = [8, 9, 12, 13]
    C22_idx = [10, 11, 14, 15]
    
    # For each M_t, list of (C_block, sign)
    M_contributions = {
        0: [(C11_idx, 1), (C22_idx, 1)],   # M1
        1: [(C21_idx, 1), (C22_idx, -1)],  # M2
        2: [(C12_idx, 1), (C22_idx, 1)],   # M3
        3: [(C11_idx, 1), (C21_idx, 1)],   # M4
        4: [(C11_idx, -1), (C12_idx, 1)],  # M5
        5: [(C22_idx, 1)],                  # M6
        6: [(C11_idx, 1)],                  # M7
    }
    
    # Now build the full U, V, W
    col = 0
    for t in range(7):  # M1..M7
        for s in range(7):  # Strassen sub-terms
            # U: for each of the 4 A entries in the 2x2 local block
            for a_local in range(4):
                u_coeff = U2[a_local, s]
                if u_coeff != 0:
                    for a_global, a_sign in A_comb[t][a_local]:
                        U[a_global, col] += u_coeff * a_sign
            
            # V: for each of the 4 B entries in the 2x2 local block
            for b_local in range(4):
                v_coeff = V2[b_local, s]
                if v_coeff != 0:
                    for b_global, b_sign in B_comb[t][b_local]:
                        V[b_global, col] += v_coeff * b_sign
            
            # W: for each of the 4 C entries in the 2x2 local result
            for c_local in range(4):
                w_coeff = W2[c_local, s]
                if w_coeff != 0:
                    for c_block, c_sign in M_contributions[t]:
                        W[c_block[c_local], col] += w_coeff * c_sign
            
            col += 1
    
    return U, V, W

def build_4x4x4x1():
    """Build rank-16 schoolbook decomposition for 4x4 x 4x1."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = 4, 4, 9
    
    # Build 4x4 x 4x4 decomposition (rank 49)
    U1, V1, W1 = build_4x4x4x4()
    
    # Build second 4x4 x 4x4 decomposition (rank 49) - same as first but for B2
    # Actually, B2 is columns 4:8 of B, so it's a different set of B entries
    # I need to be more careful here.
    
    # Let me reconsider the structure.
    # A is 4x4, B is 4x9.
    # B = [B1 | B2 | B3] where B1 is 4x4 (cols 0-3), B2 is 4x4 (cols 4-7), B3 is 4x1 (col 8)
    # C = A*B1 + A*B2 + A*B3
    # C is 4x9.
    
    # For A*B1: A is 4x4, B1 is 4x4. This is a 4x4 x 4x4 multiplication.
    #   U has shape (4*4, 49), V has shape (4*4, 49), W has shape (4*4, 49)
    #   But in the full problem, V should have shape (4*9, rank) and W should have shape (9*4, rank)
    #   So for B1, the V entries correspond to B[j, k] where k=0..3, which are V[j*9 + k, :]
    #   And W entries correspond to C[i, k] where k=0..3, which are W[k*4 + i, :]
    
    # For A*B2: A is 4x4, B2 is 4x4.
    #   V entries correspond to B[j, k] where k=4..7, which are V[j*9 + k, :]
    #   W entries correspond to C[i, k] where k=4..7, which are W[k*4 + i, :]
    
    # For A*B3: A is 4x4, B3 is 4x1.
    #   V entries correspond to B[j, k] where k=8, which are V[j*9 + 8, :]
    #   W entries correspond to C[i, k] where k=8, which are W[8*4 + i, :]
    
    # So I need to build the full U, V, W with the correct indexing.
    
    rank_total = 49 + 49 + 16
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((36, rank_total))
    W_full = np.zeros((36, rank_total))
    
    # Part 1: A*B1 (rank 49)
    # U1: (16, 49), V1: (16, 49), W1: (16, 49)
    # V1[j, k] corresponds to B[j, k] for k=0..3 → V_full[j*9 + k, 0:49]
    # W1[k, i] corresponds to C[i, k] for k=0..3 → W_full[k*4 + i, 0:49]
    # Wait, W1 is indexed as W[k*n+i] in the 4x4x4x4 subproblem, where n=4.
    # In the subproblem, W has shape (p*n, rank) = (4*4, 49) = (16, 49).
    # W_sub[k*4 + i, c] where k=0..3, i=0..3.
    # In the full problem, W_full[k*9 + i, c] where k=0..3, i=0..3.
    
    U_full[:, 0:49] = U1
    
    for j in range(4):
        for k in range(4):
            V_full[j*9 + k, 0:49] = V1[j*4 + k, :]
    
    for k in range(4):
        for i in range(4):
            W_full[k*9 + i, 0:49] = W1[k*4 + i, :]
    
    # Part 2: A*B2 (rank 49)
    # Same structure as Part 1, but B2 is cols 4-7 of B.
    # V2[j, k] for k=0..3 corresponds to B[j, k+4] → V_full[j*9 + (k+4), 49:98]
    # W2[k, i] for k=0..3 corresponds to C[i, k+4] → W_full[(k+4)*9 + i, 49:98]
    
    U_full[:, 49:98] = U1  # Same A
    
    for j in range(4):
        for k in range(4):
            V_full[j*9 + (k+4), 49:98] = V1[j*4 + k, :]
    
    for k in range(4):
        for i in range(4):
            W_full[(k+4)*9 + i, 49:98] = W1[k*4 + i, :]
    
    # Part 3: A*B3 (rank 16)
    U3, V3, W3 = build_4x4x4x1()
    # U3: (16, 16), V3: (4*1, 16) = (4, 16), W3: (1*4, 16) = (4, 16)
    # V3[j, 0] corresponds to B[j, 8] → V_full[j*9 + 8, 98:114]
    # W3[0, i] corresponds to C[i, 8] → W_full[8*9 + i, 98:114]
    
    U_full[:, 98:114] = U3
    
    for j in range(4):
        V_full[j*9 + 8, 98:114] = V3[j, :]
    
    for i in range(4):
        W_full[8*9 + i, 98:114] = W3[0, i]
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Verify
    if time.time() < deadline - 10:
        # Quick verification
        U_np = np.array(U_list)
        V_np = np.array(V_list)
        W_np = np.array(W_list)
        
        # Check a few random entries
        import random
        random.seed(42)
        for _ in range(100):
            i = random.randint(0, 3)
            j = random.randint(0, 3)
            k = random.randint(0, 8)
            val = sum(U_np[i*4+j, c] * V_np[j*9+k, c] * W_np[k*4+i, c] for c in range(rank_total))
            if val != 1:
                raise ValueError(f"Verification failed at ({i},{j},{k}): got {val}, expected 1")
    
    return {"U": U_list, "V": V_list, "W": W_list}
