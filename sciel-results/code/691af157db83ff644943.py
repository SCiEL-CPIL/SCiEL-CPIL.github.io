import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Target: 2x3 * 3x7
    # We use a block decomposition:
    # A is 2x3. Split into A1 (2x2, cols 0-1) and A2 (2x1, col 2).
    # B is 3x7. Split into B1 (2x7, rows 0-1) and B2 (1x7, row 2).
    # C = A1*B1 + A2*B2.
    
    # A1*B1 is 2x2 * 2x7.
    # Split B1 (2x7) into B1_1 (2x2), B1_2 (2x2), B1_3 (2x2), B1_4 (2x1).
    # A1*B1_1: 2x2*2x2 (Rank 7, Strassen)
    # A1*B1_2: 2x2*2x2 (Rank 7, Strassen)
    # A1*B1_3: 2x2*2x2 (Rank 7, Strassen)
    # A1*B1_4: 2x2*2x1 (Rank 2)
    
    # A2*B2 is 2x1 * 1x7 (Rank 1)
    
    # Total Rank = 7 + 7 + 7 + 2 + 1 = 24.
    
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    # u_idx, v_idx, w_idx are the indices in U, V, W.
    # col is the rank index.
    # val is the value.
    def add_term(u_idx, v_idx, w_idx, col, val):
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val

    # Strassen 2x2x2 decomposition
    # Local indices:
    # A: i in {0,1}, j in {0,1} -> idx i*2+j
    # B: j in {0,1}, k in {0,1} -> idx j*2+k
    # C: k in {0,1}, i in {0,1} -> idx k*2+i
    # 
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-f)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    def add_strassen_block(offset_a, offset_b, offset_c, start_col):
        # offset_a: row/col offset in A (2x2)
        # offset_b: row/col offset in B (2x2)
        # offset_c: row/col offset in C (2x2)
        
        # P1: (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11 (for output mapping)
        # Wait, the tensor form is sum U[i,j] V[j,k] W[k,i].
        # For Strassen, we define intermediate products.
        # The standard tensor decomposition for Strassen uses 7 terms.
        # The coefficients for U, V, W are:
        
        # Term 1: P1
        # U[0,0]=1, U[1,1]=1
        # V[0,0]=1, V[1,1]=1
        # W[0,0]=1, W[1,1]=1 (Wait, W maps k,i. C00 needs W[0,0], C11 needs W[1,1]?)
        # Actually, the output C is formed by summing these terms.
        # The decomposition must satisfy T[i,j,k] = sum U[i,j,l] V[j,k,l] W[k,i,l].
        # For Strassen, the "W" part distributes the result to C.
        
        # Let's use the known coefficients for Strassen tensor rank 7.
        # The 7 terms are:
        # l=0: U: [[1,0],[0,1]], V: [[1,0],[0,1]], W: [[1,0],[0,1]] (P1) -> C00 += 1, C11 += 1
        # l=1: U: [[0,0],[1,1]], V: [[1,0],[0,0]], W: [[0,1],[0,0]] (P2) -> C10 += 1
        # l=2: U: [[1,0],[0,0]], V: [[0,1],[0,-1]], W: [[0,1],[0,0]] (P3) -> C01 += 1
        # l=3: U: [[0,0],[0,1]], V: [[0,-1],[0,1]], W: [[1,0],[0,1]] (P4) -> C00 += 1, C11 += 1
        # l=4: U: [[1,1],[0,0]], V: [[0,0],[0,1]], W: [[-1,0],[0,0]] (P5) -> C00 -= 1
        # l=5: U: [[1,-1],[0,0]], V: [[1,1],[0,0]], W: [[0,0],[0,1]] (P6) -> C11 += 1
        # l=6: U: [[0,1],[0,-1]], V: [[0,0],[1,1]], W: [[1,0],[0,1]] (P7) -> C00 += 1, C11 += 1
        
        # Let's verify P1: U[0,0]=1, U[1,1]=1. V[0,0]=1, V[1,1]=1.
        # T[0,0,0] += 1*1*1 = 1.
        # T[0,1,0] += 0.
        # T[1,0,0] += 0.
        # T[1,1,0] += 1*1*1 = 1.
        # This adds 1 to C00 and C11?
        # W[0,0]=1 (k=0, i=0). W[1,1]=1 (k=1, i=1).
        # So T[0,0,0] gets W[0,0]=1.
        # T[1,1,0] gets W[1,1]=1.
        # This matches P1 contributing to C00 and C11.
        
        # We will hardcode the 7 terms for Strassen.
        
        # Term 0 (P1)
        add_term(offset_a + 0, offset_b + 0, offset_c + 0, start_col + 0, 1)
        add_term(offset_a + 3, offset_b + 2, offset_c + 3, start_col + 0, 1) # A[1,1], B[1,1], C[1,1]
        
        # Term 1 (P2)
        add_term(offset_a + 2, offset_b + 0, offset_c + 2, start_col + 1, 1) # A[1,0], B[0,0], C[1,0]
        add_term(offset_a + 3, offset_b + 0, offset_c + 2, start_col + 1, 1) # A[1,1], B[0,0], C[1,0]
        
        # Term 2 (P3)
        add_term(offset_a + 0, offset_b + 1, offset_c + 1, start_col + 2, 1) # A[0,0], B[0,1], C[0,1]
        add_term(offset_a + 0, offset_b + 2, offset_c + 1, start_col + 2, -1) # A[0,0], B[1,1], C[0,1]
        
        # Term 3 (P4)
        add_term(offset_a + 3, offset_b + 1, offset_c + 0, start_col + 3, -1) # A[1,1], B[0,1], C[0,0]
        add_term(offset_a + 3, offset_b + 2, offset_c + 3, start_col + 3, 1) # A[1,1], B[1,1], C[1,1]
        
        # Term 4 (P5)
        add_term(offset_a + 0, offset_b + 2, offset_c + 0, start_col + 4, 1) # A[0,0], B[1,1], C[0,0]
        add_term(offset_a + 1, offset_b + 2, offset_c + 0, start_col + 4, 1) # A[0,1], B[1,1], C[0,0]
        
        # Term 5 (P6)
        add_term(offset_a + 0, offset_b + 0, offset_c + 3, start_col + 5, 1) # A[0,0], B[0,0], C[1,1]
        add_term(offset_a + 0, offset_b + 1, offset_c + 3, start_col + 5, 1) # A[0,0], B[0,1], C[1,1]
        add_term(offset_a + 1, offset_b + 0, offset_c + 3, start_col + 5, -1) # A[0,1], B[0,0], C[1,1]
        add_term(offset_a + 1, offset_b + 1, offset_c + 3, start_col + 5, -1) # A[0,1], B[0,1], C[1,1]
        
        # Term 6 (P7)
        add_term(offset_a + 1, offset_b + 2, offset_c + 0, start_col + 6, 1) # A[0,1], B[1,1], C[0,0]
        add_term(offset_a + 1, offset_b + 2, offset_c + 3, start_col + 6, 1) # A[0,1], B[1,1], C[1,1]
        add_term(offset_a + 3, offset_b + 2, offset_c + 0, start_col + 6, -1) # A[1,1], B[1,1], C[0,0]
        add_term(offset_a + 3, offset_b + 2, offset_c + 3, start_col + 6, -1) # A[1,1], B[1,1], C[1,1]
        
        # Note: The offsets for U, V, W need to be mapped correctly to the global indices.
        # U index: i*m + j. For 2x2 block at row_off, col_off in A (which is 2x3), 
        # the local i,j map to global i, j+col_off.
        # V index: j*p + k. For 2x2 block at row_off, col_off in B (which is 3x7),
        # the local j,k map to global j+row_off, k+col_off.
        # W index: k*n + i. For 2x2 block at row_off, col_off in C (which is 2x7),
        # the local k,i map to global k+col_off, i.
        
        # My add_term function above assumed local indices 0..3 for U, V, W.
        # I need to pass the global offsets.
        
    # The logic above is getting complicated with offsets.
    # I will implement a cleaner version.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
