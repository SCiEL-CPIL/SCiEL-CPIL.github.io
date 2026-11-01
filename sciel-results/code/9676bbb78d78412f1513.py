"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition that splits the p columns into pairs (Strassen-like rank 7 per pair)
    # and handles the last single column with schoolbook (rank 4).
    # 13 = 6 * 2 + 1.
    # Rank = 6 * 7 + 4 = 46.
    
    rank = 46
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_offset = 0
    current_rank = 0
    
    # Strassen decomposition for 2x2x2 (m=2, p_block=2)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U indices: a:0, b:1, c:2, d:3
    # V indices for block: e:0, f:1, g:2, h:3 (local)
    # W indices for block: C00:0, C01:1, C10:2, C11:3 (local)
    
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    def add_strassen_block(start_col, start_rank):
        # start_col: 0-based index of first column in B
        # start_rank: 0-based index in rank dimension
        
        # P1
        U[0, start_rank] += 1; U[3, start_rank] += 1
        V[0, start_rank] += 1; V[1, start_rank] += 1 # e+h? No, B is 2x2. e=B[0,0], h=B[1,1]?
        # Wait, B local indices:
        # Row 0 of B: B[0, start_col], B[0, start_col+1]
        # Row 1 of B: B[1, start_col], B[1, start_col+1]
        # Let e = B[0, start_col], f = B[0, start_col+1]
        # Let g = B[1, start_col], h = B[1, start_col+1]
        # P1 = (a+d)(e+h)
        # V index for e: 0*m + start_col
        # V index for h: 1*m + (start_col+1)
        
        V[0*m + start_col, start_rank] = 1
        V[1*m + (start_col+1), start_rank] = 1
        # W: C00 and C11 get P1?
        # C00 = P1 - P5 + P3 + P7
        # C11 = P1 - P2 + P4 + P6
        W[0*n + 0, start_rank] = 1 # C00 (k=0, i=0)
        W[1*n + 1, start_rank] = 1 # C11 (k=1, i=1)
        
        # P2 = (c+d)e
        U[2, start_rank+1] += 1; U[3, start_rank+1] += 1
        V[0*m + start_col, start_rank+1] = 1
        # C21 = P2 + P3? No, C21 is C[1,0]. C00 is C[0,0].
        # My W indexing: W[k*n + i].
        # C[0,0] -> k=0, i=0 -> idx 0.
        # C[0,1] -> k=0, i=1 -> idx 1.
        # C[1,0] -> k=1, i=0 -> idx 2.
        # C[1,1] -> k=1, i=1 -> idx 3.
        # Strassen:
        # C[0,0] = P1 - P5 + P3 + P7
        # C[0,1] = P5 + P4
        # C[1,0] = P2 + P3
        # C[1,1] = P1 - P2 + P4 + P6
        
        W[2*n + 0, start_rank+1] = 1 # C[1,0] gets P2
        W[3*n + 1, start_rank+1] = -1 # C[1,1] gets -P2
        
        # P3 = a(f-h)
        U[0, start_rank+2] += 1
        V[0*m + (start_col+1), start_rank+2] = 1
        V[1*m + start_col, start_rank+2] = -1
        W[0*n + 0, start_rank+2] = 1 # C[0,0] gets P3
        W[2*n + 0, start_rank+2] = 1 # C[1,0] gets P3
        
        # P4 = d(g-e)
        U[3, start_rank+3] += 1
        V[1*m + start_col, start_rank+3] = 1
        V[0*m + start_col, start_rank+3] = -1
        W[0*n + 1, start_rank+3] = 1 # C[0,1] gets P4
        W[3*n + 1, start_rank+3] = 1 # C[1,1] gets P4
        
        # P5 = (a+b)h
        U[0, start_rank+4] += 1; U[1, start_rank+4] += 1
        V[1*m + (start_col+1), start_rank+4] = 1
        W[0*n + 0, start_rank+4] = -1 # C[0,0] gets -P5
        W[0*n + 1, start_rank+4] = 1  # C[0,1] gets P5
        
        # P6 = (c-a)(e+f)
        U[2, start_rank+5] += 1; U[0, start_rank+5] -= 1
        V[0*m + start_col, start_rank+5] = 1
        V[0*m + (start_col+1), start_rank+5] = 1
        W[3*n + 1, start_rank+5] = 1 # C[1,1] gets P6
        
        # P7 = (b-d)(g+h)
        U[1, start_rank+6] += 1; U[3, start_rank+6] -= 1
        V[1*m + start_col, start_rank+6] = 1
        V[1*m + (start_col+1), start_rank+6] = 1
        W[0*n + 0, start_rank+6] = 1 # C[0,0] gets P7
        
        return start_rank + 7

    # Process 6 pairs
    for i in range(6):
        start_col = i * 2
        current_rank = add_strassen_block(start_col, current_rank)
        
    # Process last single column (p=1)
    # Schoolbook for 2x2x1:
    # C[0,0] = a*e + b*g
    # C[1,0] = c*e + d*g
    # Rank 4.
    last_col = 12
    
    # ae
    U[0, current_rank] = 1
    V[0*m + last_col, current_rank] = 1
    W[0*n + 0, current_rank] = 1
    
    # bg
    U[1, current_rank+1] = 1
    V[1*m + last_col, current_rank+1] = 1
    W[0*n + 0, current_rank+1] = 1
    
    # ce
    U[2, current_rank+2] = 1
    V[0*m + last_col, current_rank+2] = 1
    W[1*n + 0, current_rank+2] = 1
    
    # dg
    U[3, current_rank+3] = 1
    V[1*m + last_col, current_rank+3] = 1
    W[1*n + 0, current_rank+3] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
