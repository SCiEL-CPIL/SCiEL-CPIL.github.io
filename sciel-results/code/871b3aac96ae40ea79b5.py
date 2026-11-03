import time
import sys
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Target entries for validation
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.append((i * m + j, j * p + k, k * n + i))
    
    def check_decomposition(U, V, W, rank):
        for (a, b, c) in target_entries:
            val = 0.0
            for s in range(rank):
                val += U[a, s] * V[b, s] * W[c, s]
            if abs(val - 1) > 1e-6:
                return False
        return True
    
    # Laderman's algorithm for 3x3x3 (rank 23)
    def laderman_3x3x3(i_off, j_off, k_off):
        # Laderman's 23-term algorithm for 3x3 matrix multiplication
        # A[i_off:i_off+3, j_off:j_off+3] * B[j_off:j_off+3, k_off:k_off+3] = C[i_off:i_off+3, k_off:k_off+3]
        # Using 0-indexed local coordinates (0,1,2)
        
        # M1 = A[0,0]*B[0,0]
        add_component(
            {(i_off+0, j_off+0): 1},
            {(j_off+0, k_off+0): 1},
            {(i_off+0, k_off+0): 1}
        )
        
        # M2 = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        add_component(
            {(i_off+0, j_off+0): 1, (i_off+0, j_off+1): 1},
            {(j_off+0, k_off+1): 1, (j_off+1, k_off+1): 1},
            {(i_off+0, k_off+1): 1}
        )
        
        # M3 = A[0,0]*B[1,0] + A[1,0]*B[0,0]
        add_component(
            {(i_off+0, j_off+0): 1, (i_off+1, j_off+0): 1},
            {(j_off+1, k_off+0): 1, (j_off+0, k_off+0): 1},
            {(i_off+1, k_off+0): 1}
        )
        
        # M4 = A[1,1]*B[1,1]
        add_component(
            {(i_off+1, j_off+1): 1},
            {(j_off+1, k_off+1): 1},
            {(i_off+1, k_off+1): 1}
        )
        
        # M5 = A[0,1]*B[1,0] + A[1,1]*B[0,0]
        add_component(
            {(i_off+0, j_off+1): 1, (i_off+1, j_off+1): 1},
            {(j_off+1, k_off+0): 1, (j_off+0, k_off+0): 1},
            {(i_off+1, k_off+0): 1, (i_off+0, k_off+0): -1}
        )
        
        # M6 = A[0,2]*B[2,0] + A[2,2]*B[0,0]
        add_component(
            {(i_off+0, j_off+2): 1, (i_off+2, j_off+2): 1},
            {(j_off+2, k_off+0): 1, (j_off+0, k_off+0): 1},
            {(i_off+2, k_off+0): 1, (i_off+0, k_off+0): -1}
        )
        
        # M7 = A[1,2]*B[2,1] + A[2,2]*B[1,1]
        add_component(
            {(i_off+1, j_off+2): 1, (i_off+2, j_off+2): 1},
            {(j_off+2, k_off+1): 1, (j_off+1, k_off+1): 1},
            {(i_off+2, k_off+1): 1, (i_off+1, k_off+1): -1}
        )
        
        # M8 = A[2,0]*B[0,2] + A[0,0]*B[2,2]
        add_component(
            {(i_off+2, j_off+0): 1, (i_off+0, j_off+0): 1},
            {(j_off+0, k_off+2): 1, (j_off+2, k_off+2): 1},
            {(i_off+0, k_off+2): 1, (i_off+2, k_off+2): -1}
        )
        
        # M9 = A[2,1]*B[1,2] + A[1,1]*B[2,2]
        add_component(
            {(i_off+2, j_off+1): 1, (i_off+1, j_off+1): 1},
            {(j_off+1, k_off+2): 1, (j_off+2, k_off+2): 1},
            {(i_off+1, k_off+2): 1, (i_off+2, k_off+2): -1}
        )
        
        # M10 = A[0,0]*B[2,1] + A[0,1]*B[2,2] + A[0,2]*B[1,1]
        add_component(
            {(i_off+0, j_off+0): 1, (i_off+0, j_off+1): 1, (i_off+0, j_off+2): 1},
            {(j_off+2, k_off+1): 1, (j_off+2, k_off+2): 1, (j_off+1, k_off+1): 1},
            {(i_off+0, k_off+2): 1}
        )
        
        # M11 = A[1,0]*B[2,0] + A[0,0]*B[2,1] + A[0,1]*B[2,2]
        add_component(
            {(i_off+1, j_off+0): 1, (i_off+0, j_off+0): 1, (i_off+0, j_off+1): 1},
            {(j_off+2, k_off+0): 1, (j_off+2, k_off+1): 1, (j_off+2, k_off+2): 1},
            {(i_off+1, k_off+2): 1}
        )
        
        # M12 = A[2,0]*B[1,0] + A[1,0]*B[2,1] + A[0,0]*B[2,2]
        add_component(
            {(i_off+2, j_off+0): 1, (i_off+1, j_off+0): 1, (i_off+0, j_off+0): 1},
            {(j_off+1, k_off+0): 1, (j_off+2, k_off+1): 1, (j_off+2, k_off+2): 1},
            {(i_off+2, k_off+2): 1}
        )
        
        # M13 = A[0,1]*B[0,2] + A[0,2]*B[0,1] + A[0,0]*B[1,2]
        add_component(
            {(i_off+0, j_off+1): 1, (i_off+0, j_off+2): 1, (i_off+0, j_off+0): 1},
            {(j_off+0, k_off+2): 1, (j_off+0, k_off+1): 1, (j_off+1, k_off+2): 1},
            {(i_off+0, k_off+2): 1, (i_off+0, k_off+1): -1}
        )
        
        # M14 = A[1,2]*B[0,2] + A[0,2]*B[1,2] + A[0,1]*B[1,1]
        add_component(
            {(i_off+1, j_off+2): 1, (i_off+0, j_off+2): 1, (i_off+0, j_off+1): 1},
            {(j_off+0, k_off+2): 1, (j_off+1, k_off+2): 1, (j_off+1, k_off+1): 1},
            {(i_off+1, k_off+2): 1, (i_off+0, k_off+2): -1}
        )
        
        # M15 = A[2,0]*B[0,1] + A[0,0]*B[2,1] + A[1,0]*B[2,2]
        add_component(
            {(i_off+2, j_off+0): 1, (i_off+0, j_off+0): 1, (i_off+1, j_off+0): 1},
            {(j_off+0, k_off+1): 1, (j_off+2, k_off+1): 1, (j_off+2, k_off+2): 1},
            {(i_off+2, k_off+1): 1}
        )
        
        # M16 = A[0,2]*B[0,0] + A[0,1]*B[0,1] + A[0,0]*B[0,2]
        add_component(
            {(i_off+0, j_off+2): 1, (i_off+0, j_off+1): 1, (i_off+0, j_off+0): 1},
            {(j_off+2, k_off+0): 1, (j_off+1, k_off+1): 1, (j_off+0, k_off+2): 1},
            {(i_off+0, k_off+0): 1, (i_off+0, k_off+1): 1}
        )
        
        # M17 = A[1,0]*B[0,0] + A[0,0]*B[1,0] + A[1,1]*B[1,1]
        add_component(
            {(i_off+1, j_off+0): 1, (i_off+0, j_off+0): 1, (i_off+1, j_off+1): 1},
            {(j_off+0, k_off+0): 1, (j_off+1, k_off+0): 1, (j_off+1, k_off+1): 1},
            {(i_off+1, k_off+0): 1}
        )
        
        # M18 = A[2,1]*B[0,0] + A[1,1]*B[1,0] + A[2,2]*B[2,0]
        add_component(
            {(i_off+2, j_off+1): 1, (i_off+1, j_off+1): 1, (i_off+2, j_off+2): 1},
            {(j_off+0, k_off+0): 1, (j_off+1, k_off+0): 1, (j_off+2, k_off+0): 1},
            {(i_off+2, k_off+0): 1}
        )
        
        # M19 = A[0,1]*B[1,0] + A[1,1]*B[0,0] + A[0,2]*B[2,0]
        add_component(
            {(i_off+0, j_off+1): 1, (i_off+1, j_off+1): 1, (i_off+0, j_off+2): 1},
            {(j_off+1, k_off+0): 1, (j_off+0, k_off+0): 1, (j_off+2, k_off+0): 1},
            {(i_off+0, k_off+0): 1}
        )
        
        # M20 = A[2,0]*B[0,1] + A[1,0]*B[1,1] + A[0,0]*B[2,2]
        add_component(
            {(i_off+2, j_off+0): 1, (i_off+1, j_off+0): 1, (i_off+0, j_off+0): 1},
            {(j_off+0, k_off+1): 1, (j_off+1, k_off+1): 1, (j_off+2, k_off+2): 1},
            {(i_off+2, k_off+1): 1}
        )
        
        # M21 = A[0,2]*B[1,0] + A[1,2]*B[0,0] + A[2,2]*B[2,1]
        add_component(
            {(i_off+0, j_off+2): 1, (i_off+1, j_off+2): 1, (i_off+2, j_off+2): 1},
            {(j_off+1, k_off+0): 1, (j_off+0, k_off+0): 1, (j_off+2, k_off+1): 1},
            {(i_off+2, k_off+1): 1}
        )
        
        # M22 = A[1,0]*B[1,2] + A[0,0]*B[2,2] + A[1,1]*B[0,2]
        add_component(
            {(i_off+1, j_off+0): 1, (i_off+0, j_off+0): 1, (i_off+1, j_off+1): 1},
            {(j_off+1, k_off+2): 1, (j_off+2, k_off+2): 1, (j_off+0, k_off+2): 1},
            {(i_off+1, k_off+2): 1}
        )
        
        # M23 = A[2,1]*B[0,2] + A[1,1]*B[1,2] + A[0,1]*B[2,2]
        add_component(
            {(i_off+2, j_off+1): 1, (i_off+1, j_off+1): 1, (i_off+0, j_off+1): 1},
            {(j_off+0, k_off+2): 1, (j_off+1, k_off+2): 1, (j_off+2, k_off+2): 1},
            {(i_off+2, k_off+2): 1}
        )
    
    def strassen_2x2x2(i_off, j_off, k_off):
        # M1 = (A11 + A22) * (B11 + B22)
        add_component(
            {(i_off, j_off): 1, (i_off+1, j_off+1): 1},
            {(j_off, k_off): 1, (j_off+1, k_off+1): 1},
            {(i_off, k_off): 1, (i_off+1, k_off+1): 1}
        )
        
        # M2 = (A21 + A22) * B11
        add_component(
            {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1},
            {(j_off, k_off): 1},
            {(i_off+1, k_off): 1}
        )
        
        # M3 = A11 * (B12 - B22)
        add_component(
            {(i_off, j_off): 1},
            {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1},
            {(i_off, k_off+1): 1}
        )
        
        # M4 = A22 * (B21 - B11)
        add_component(
            {(i_off+1, j_off+1): 1},
            {(j_off+1, k_off): 1, (j_off, k_off): -1},
            {(i_off+1, k_off): 1}
        )
        
        # M5 = (A11 + A12) * B22
        add_component(
            {(i_off, j_off): 1, (i_off, j_off+1): 1},
            {(j_off+1, k_off+1): 1},
            {(i_off, k_off+1): -1, (i_off+1, k_off+1): 1}
        )
        
        # M6 = (A21 - A11) * (B11 + B12)
        add_component(
            {(i_off+1, j_off): 1, (i_off, j_off): -1},
            {(j_off, k_off): 1, (j_off, k_off+1): 1},
            {(i_off+1, k_off+1): 1}
        )
        
        # M7 = (A12 - A22) * (B21 + B22)
        add_component(
            {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1},
            {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1},
            {(i_off, k_off+1): 1, (i_off+1, k_off+1): -1}
        )
    
    def schoolbook_2x2x1(i_off, j_off, k_off):
        add_component({(i_off, j_off): 1}, {(j_off, k_off): 1}, {(i_off, k_off): 1})
        add_component({(i_off, j_off+1): 1}, {(j_off+1, k_off): 1}, {(i_off, k_off): 1})
        add_component({(i_off+1, j_off): 1}, {(j_off, k_off): 1}, {(i_off+1, k_off): 1})
        add_component({(i_off+1, j_off+1): 1}, {(j_off+1, k_off): 1}, {(i_off+1, k_off): 1})
    
    def schoolbook_1x2x2(i_off, j_off, k_off):
        add_component({(i_off, j_off): 1}, {(j_off, k_off): 1}, {(i_off, k_off): 1})
        add_component({(i_off, j_off): 1}, {(j_off, k_off+1): 1}, {(i_off, k_off+1): 1})
        add_component({(i_off, j_off+1): 1}, {(j_off+1, k_off): 1}, {(i_off, k_off): 1})
        add_component({(i_off, j_off+1): 1}, {(j_off+1, k_off+1): 1}, {(i_off, k_off+1): 1})
    
    def schoolbook_1x2x1(i_off, j_off, k_off):
        add_component({(i_off, j_off): 1}, {(j_off, k_off): 1}, {(i_off, k_off): 1})
        add_component({(i_off, j_off+1): 1}, {(j_off+1, k_off): 1}, {(i_off, k_off): 1})
    
    def outer_product_3x1x7(i_off, j_off, k_off):
        for di in range(3):
            for dk in range(7):
                add_component(
                    {(i_off+di, j_off): 1},
                    {(j_off, k_off+dk): 1},
                    {(i_off+di, k_off+dk): 1}
                )
    
    # Try different decompositions
    # Strategy 1: Use Laderman for 3x3x3 blocks
    # Split m=5 into 3+2, p=7 into 3+4 or 4+3
    
    # Decomposition: (3x3 x 3x3) + (3x3 x 3x2) + (3x2 x 5x3) + (3x2 x 5x2)
    # This is getting complex. Let me try a simpler approach.
    
    # Approach: Use Laderman for the first 3x3x3 block, then handle the rest
    # A is 3x5, B is 5x7, C is 3x7
    
    # Split: A = [A1 A2] where A1 is 3x3, A2 is 3x2
    #         B = [B1; B2] where B1 is 3x7, B2 is 2x7
    # C = A1*B1 + A2*B2
    
    # Use Laderman for A1*B1 (3x3 x 3x7): need to split B1 into 3x3 + 3x4
    # Use Strassen/schoolbook for the rest
    
    # Let's try: Laderman for 3x3x3 (A[0:3,0:3] * B[0:3,0:3])
    # Then handle remaining terms
    
    # Actually, let me try the same structure as before but with Laderman
    # where applicable
    
    # For now, let's use the existing 99-rank solution as a baseline
    # and try to improve it
    
    # Use the same decomposition as the current program
    max_rank = 105
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    current_rank = 0
    
    def add_component(A_coeffs, B_coeffs, C_coeffs):
        nonlocal current_rank
        if current_rank >= max_rank:
            return
        s = current_rank
        for (i, j), coeff in A_coeffs.items():
            U[i * m + j, s] += coeff
        for (j, k), coeff in B_coeffs.items():
            V[j * p + k, s] += coeff
        for (i, k), coeff in C_coeffs.items():
            W[k * n + i, s] += coeff
        current_rank += 1
    
    # Use the same decomposition as the current program (99 rank)
    # Term 1: J in {0, 1}
    strassen_2x2x2(0, 0, 0)
    strassen_2x2x2(0, 0, 2)
    strassen_2x2x2(0, 0, 4)
    schoolbook_2x2x1(0, 0, 6)
    schoolbook_1x2x2(2, 0, 0)
    schoolbook_1x2x2(2, 0, 2)
    schoolbook_1x2x2(2, 0, 4)
    schoolbook_1x2x1(2, 0, 6)
    
    # Term 2: J in {2, 3}
    strassen_2x2x2(0, 2, 0)
    strassen_2x2x2(0, 2, 2)
    strassen_2x2x2(0, 2, 4)
    schoolbook_2x2x1(0, 2, 6)
    schoolbook_1x2x2(2, 2, 0)
    schoolbook_1x2x2(2, 2, 2)
    schoolbook_1x2x2(2, 2, 4)
    schoolbook_1x2x1(2, 2, 6)
    
    # Term 3: J in {4}
    outer_product_3x1x7(0, 4, 0)
    
    if current_rank == 99:
        if check_decomposition(U, V, W, 99):
            best = {"U": U[:, :99].tolist(), "V": V[:, :99].tolist(), "W": W[:, :99].tolist()}
            best_rank = 99
    
    return best
# EVOLVE-BLOCK-END
