import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Rank 56 Strassen decomposition
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen for 2x2 * 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Coefficients for A (U), B (V), C (W) for each of the 7 Strassen products
    # A indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # B indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # C indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'A': {0: 1, 3: 1},
            'B': {0: 1, 3: 1},
            'C': {0: 1, 3: 1} # Wait, C contribution is different
        }
    ]
    # It's easier to just hardcode the linear combinations for U, V, W columns
    
    # We have 4 blocks of B (columns 0-1, 2-3, 4-5, 6-7)
    # For each block, we compute A_L B_block and A_R B_block
    # A_L is cols 0-1 of A, A_R is cols 2-3 of A
    # B_block is 2x2
    
    # Let's iterate over the 4 column blocks of B
    col_offset = 0
    for b_block in range(4):
        # B block columns: 2*b_block, 2*b_block+1
        # B block rows: 0,1 for top half, 2,3 for bottom half?
        # No, B is 4x8.
        # We split B into top 2 rows and bottom 2 rows.
        # B_top is 2x8, B_bot is 2x8.
        # We split B_top into 4 2x2 blocks.
        # B_bot into 4 2x2 blocks.
        
        # A_L is 2x2 (cols 0,1 of A)
        # A_R is 2x2 (cols 2,3 of A)
        
        # We need to compute C_block = A_L B_top_block + A_R B_bot_block
        # C_block is 2x2.
        
        # Let's do Strassen for A_L B_top_block (7 mults)
        # And Strassen for A_R B_bot_block (7 mults)
        
        # Indices for A_L in U:
        # A_L[0,0] -> U[0, col]
        # A_L[0,1] -> U[1, col]
        # A_L[1,0] -> U[2, col]
        # A_L[1,1] -> U[3, col]
        
        # Indices for A_R in U:
        # A_R[0,0] -> U[2, col] (A[0,2])
        # A_R[0,1] -> U[3, col] (A[0,3])
        # A_R[1,0] -> U[4, col] (A[1,2])
        # A_R[1,1] -> U[5, col] (A[1,3])
        
        # Indices for B_top_block in V:
        # B_top_block is rows 0,1 of B, cols 2*b_block, 2*b_block+1
        # V index = row * 8 + col
        # B[0, 2*b_block] -> V[0, col]
        # B[0, 2*b_block+1] -> V[1, col]
        # B[1, 2*b_block] -> V[8, col]
        # B[1, 2*b_block+1] -> V[9, col]
        
        # Indices for B_bot_block in V:
        # B_bot_block is rows 2,3 of B, cols 2*b_block, 2*b_block+1
        # B[2, 2*b_block] -> V[16, col]
        # B[2, 2*b_block+1] -> V[17, col]
        # B[3, 2*b_block] -> V[24, col]
        # B[3, 2*b_block+1] -> V[25, col]
        
        # Indices for C_block in W:
        # C_block is 2x2, corresponding to C rows 0,1 and cols 2*b_block, 2*b_block+1
        # W index = col * 2 + row (Trace convention W[k*n+i, r] -> W[col*n+row])
        # C[0, 2*b_block] -> W[2*b_block*2 + 0, col] = W[4*b_block, col]
        # C[0, 2*b_block+1] -> W[2*b_block*2 + 1, col] = W[4*b_block+1, col]
        # C[1, 2*b_block] -> W[2*b_block*2 + 2, col] = W[4*b_block+2, col]
        # C[1, 2*b_block+1] -> W[2*b_block*2 + 3, col] = W[4*b_block+3, col]
        
        # Helper to add Strassen terms
        def add_strassen(u_base, v_base, w_base, col_start):
            # u_base: list of 4 indices in U
            # v_base: list of 4 indices in V
            # w_base: list of 4 indices in W
            
            # M1: (A00+A11)(B00+B11)
            U[u_base[0], col_start] += 1
            U[u_base[3], col_start] += 1
            V[v_base[0], col_start] += 1
            V[v_base[3], col_start] += 1
            W[w_base[0], col_start] += 1 # C00
            W[w_base[3], col_start] += 1 # C11
            
            # M2: (A10+A11)B00
            U[u_base[2], col_start+1] += 1
            U[u_base[3], col_start+1] += 1
            V[v_base[0], col_start+1] += 1
            W[w_base[2], col_start+1] += 1 # C10
            W[w_base[3], col_start+1] += -1 # C11
            
            # M3: A00(B01-B11)
            U[u_base[0], col_start+2] += 1
            V[v_base[1], col_start+2] += 1
            V[v_base[3], col_start+2] += -1
            W[w_base[1], col_start+2] += 1 # C01
            W[w_base[3], col_start+2] += 1 # C11
            
            # M4: A11(B10-B00)
            U[u_base[3], col_start+3] += 1
            V[v_base[2], col_start+3] += 1
            V[v_base[0], col_start+3] += -1
            W[w_base[2], col_start+3] += 1 # C10
            
            # M5: (A00+A01)B11
            U[u_base[0], col_start+4] += 1
            U[u_base[1], col_start+4] += 1
            V[v_base[3], col_start+4] += 1
            W[w_base[0], col_start+4] += -1 # C00
            W[w_base[1], col_start+4] += 1 # C01
            
            # M6: (A10-A00)(B00+B01)
            U[u_base[2], col_start+5] += 1
            U[u_base[0], col_start+5] += -1
            V[v_base[0], col_start+5] += 1
            V[v_base[1], col_start+5] += 1
            W[w_base[3], col_start+5] += 1 # C11
            
            # M7: (A01-A11)(B10+B11)
            U[u_base[1], col_start+6] += 1
            U[u_base[3], col_start+6] += -1
            V[v_base[2], col_start+6] += 1
            V[v_base[3], col_start+6] += 1
            W[w_base[0], col_start+6] += 1 # C00
            W[w_base[2], col_start+6] += 1 # C10

        # Define bases for A_L and A_R
        # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
        u_L = [0, 1, 2, 3]
        # A_R: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2,3,4,5
        u_R = [2, 3, 4, 5]
        
        # Define bases for B_top and B_bot
        # B_top: B[0, 2b], B[0, 2b+1], B[1, 2b], B[1, 2b+1]
        # V indices: 0, 1, 8, 9 (for b=0)
        v_T = [0 + 2*b_block, 1 + 2*b_block, 8 + 2*b_block, 9 + 2*b_block]
        
        # B_bot: B[2, 2b], B[2, 2b+1], B[3, 2b], B[3, 2b+1]
        # V indices: 16, 17, 24, 25 (for b=0)
        v_B = [16 + 2*b_block, 17 + 2*b_block, 24 + 2*b_block, 25 + 2*b_block]
        
        # Define bases for C
        # C: C[0, 2b], C[0, 2b+1], C[1, 2b], C[1, 2b+1]
        # W indices: 4b, 4b+1, 4b+2, 4b+3
        w_C = [4*b_block, 4*b_block+1, 4*b_block+2, 4*b_block+3]
        
        # Add Strassen for A_L B_T
        add_strassen(u_L, v_T, w_C, col_offset)
        col_offset += 7
        
        # Add Strassen for A_R B_B
        add_strassen(u_R, v_B, w_C, col_offset)
        col_offset += 7
        
    # Verify
    valid = True
    if time.time() < deadline:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    
    if not valid:
        U, V, W, rank = schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
