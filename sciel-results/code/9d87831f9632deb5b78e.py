"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 5]
_START_TIME = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    total_rank = 18  # 7 (Strassen for cols 0-1) + 7 (Strassen for cols 2-3) + 4 (standard for col 4)
    
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return _build_rank18(n, m, p)
    
    return _build_rank18(n, m, p)

def _build_rank18(n, m, p):
    total_rank = 18
    
    # U: (n*m, rank) = (4, 18)
    # V: (m*p, rank) = (10, 18)
    # W: (p*n, rank) = (10, 18)
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2 multiplication: rank 7
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C[0,0] = M1+M4-M5+M7, C[0,1] = M3+M5
    # C[1,0] = M2+M4, C[1,1] = M1-M2+M3+M6
    
    # For a 2x2 block of B starting at column offset `col_off`:
    # B[j, col_off+0] and B[j, col_off+1]
    # The 7 Strassen terms with their U, V, W coefficients:
    
    def add_strassen_block(col_off, start_col):
        """Add 7 Strassen rank-1 terms for B columns [col_off, col_off+1]."""
        # Each term r has:
        #   U coefficients over A[i,j] (i in {0,1}, j in {0,1}) -> row i*m+j = i*2+j
        #   V coefficients over B[j,k] (j in {0,1}, k in {col_off, col_off+1}) -> row j*p+k
        #   W coefficients over C[i,k] (i in {0,1}, k in {col_off, col_off+1}) -> row k*n+i
        
        # Term 1: M1 = (a+d)(e+h)
        #   A part: a=1, d=1 -> U[0,0]=1, U[1,1]=1
        #   B part: e=B[0,col_off], h=B[1,col_off+1] -> V[0, col_off]=1, V[1, col_off+1]=1
        #   C part: M1 contributes to C[0,col_off] with +1, C[1,col_off+1] with +1
        #           W[col_off*2+0] = +1, W[(col_off+1)*2+1] = +1
        c = start_col
        U[0][c] += 1.0   # a = A[0,0]
        U[3][c] += 1.0   # d = A[1,1]
        V[col_off][c] += 1.0       # B[0, col_off]
        V[1 * p + (col_off + 1)][c] += 1.0  # B[1, col_off+1]
        W[col_off * n + 0][c] += 1.0    # C[0, col_off]
        W[(col_off + 1) * n + 1][c] += 1.0  # C[1, col_off+1]
        
        # Term 2: M2 = (c+d)e
        #   A part: c=1, d=1 -> U[2,0]=1, U[3,0]=1
        #   B part: e=B[0,col_off] -> V[0, col_off]=1
        #   C part: M2 contributes to C[1,col_off] with +1, C[1,col_off+1] with -1
        c = start_col + 1
        U[2][c] += 1.0   # c = A[1,0]
        U[3][c] += 1.0   # d = A[1,1]
        V[col_off][c] += 1.0       # B[0, col_off]
        W[col_off * n + 1][c] += 1.0    # C[1, col_off]
        W[(col_off + 1) * n + 1][c] += -1.0  # C[1, col_off+1]
        
        # Term 3: M3 = a(f-h)
        #   A part: a=1 -> U[0,0]=1
        #   B part: f=B[0,col_off+1], h=B[1,col_off+1] -> V[0,col_off+1]=1, V[1,col_off+1]=-1
        #   C part: M3 contributes to C[0,col_off+1] with +1, C[1,col_off+1] with +1
        c = start_col + 2
        U[0][c] += 1.0   # a = A[0,0]
        V[0 * p + (col_off + 1)][c] += 1.0  # B[0, col_off+1]
        V[1 * p + (col_off + 1)][c] += -1.0 # B[1, col_off+1]
        W[(col_off + 1) * n + 0][c] += 1.0  # C[0, col_off+1]
        W[(col_off + 1) * n + 1][c] += 1.0  # C[1, col_off+1]
        
        # Term 4: M4 = d(g-e)
        #   A part: d=1 -> U[3,0]=1
        #   B part: g=B[1,col_off], e=B[0,col_off] -> V[1,col_off]=1, V[0,col_off]=-1
        #   C part: M4 contributes to C[0,col_off] with +1, C[1,col_off] with +1
        c = start_col + 3
        U[3][c] += 1.0   # d = A[1,1]
        V[1 * p + col_off][c] += 1.0   # B[1, col_off]
        V[0 * p + col_off][c] += -1.0  # B[0, col_off]
        W[col_off * n + 0][c] += 1.0   # C[0, col_off]
        W[col_off * n + 1][c] += 1.0   # C[1, col_off]
        
        # Term 5: M5 = (a+b)h
        #   A part: a=1, b=1 -> U[0,0]=1, U[1,0]=1
        #   B part: h=B[1,col_off+1] -> V[1,col_off+1]=1
        #   C part: M5 contributes to C[0,col_off] with -1, C[0,col_off+1] with +1
        c = start_col + 4
        U[0][c] += 1.0   # a = A[0,0]
        U[1][c] += 1.0   # b = A[0,1]
        V[1 * p + (col_off + 1)][c] += 1.0  # B[1, col_off+1]
        W[col_off * n + 0][c] += -1.0   # C[0, col_off]
        W[(col_off + 1) * n + 0][c] += 1.0  # C[0, col_off+1]
        
        # Term 6: M6 = (c-a)(e+f)
        #   A part: c=1, a=-1 -> U[2,0]=1, U[0,0]=-1
        #   B part: e=B[0,col_off], f=B[0,col_off+1] -> V[0,col_off]=1, V[0,col_off+1]=1
        #   C part: M6 contributes to C[1,col_off+1] with +1
        c = start_col + 5
        U[2][c] += 1.0    # c = A[1,0]
        U[0][c] += -1.0   # a = A[0,0]
        V[0 * p + col_off][c] += 1.0      # B[0, col_off]
        V[0 * p + (col_off + 1)][c] += 1.0  # B[0, col_off+1]
        W[(col_off + 1) * n + 1][c] += 1.0  # C[1, col_off+1]
        
        # Term 7: M7 = (b-d)(g+h)
        #   A part: b=1, d=-1 -> U[1,0]=1, U[3,0]=-1
        #   B part: g=B[1,col_off], h=B[1,col_off+1] -> V[1,col_off]=1, V[1,col_off+1]=1
        #   C part: M7 contributes to C[0,col_off] with +1
        c = start_col + 6
        U[1][c] += 1.0    # b = A[0,1]
        U[3][c] += -1.0   # d = A[1,1]
        V[1 * p + col_off][c] += 1.0      # B[1, col_off]
        V[1 * p + (col_off + 1)][c] += 1.0  # B[1, col_off+1]
        W[col_off * n + 0][c] += 1.0   # C[0, col_off]
    
    # Block 1: columns 0,1
    add_strassen_block(0, 0)
    # Block 2: columns 2,3
    add_strassen_block(2, 7)
    
    # Block 3: column 4 (2x1 standard multiplication)
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    # 4 rank-1 terms:
    
    # Term 8: A[0,0] * B[0,4] -> C[0,4]
    c = 14
    U[0][c] = 1.0
    V[0 * p + 4][c] = 1.0
    W[4 * n + 0][c] = 1.0
    
    # Term 9: A[0,1] * B[1,4] -> C[0,4]
    c = 15
    U[1][c] = 1.0
    V[1 * p + 4][c] = 1.0
    W[4 * n + 0][c] = 1.0
    
    # Term 10: A[1,0] * B[0,4] -> C[1,4]
    c = 16
    U[2][c] = 1.0
    V[0 * p + 4][c] = 1.0
    W[4 * n + 1][c] = 1.0
    
    # Term 11: A[1,1] * B[1,4] -> C[1,4]
    c = 17
    U[3][c] = 1.0
    V[1 * p + 4][c] = 1.0
    W[4 * n + 1][c] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
