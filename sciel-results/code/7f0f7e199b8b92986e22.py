"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 43 decomposition
    # 6 blocks of 2x2x2 using Strassen (rank 7 each) -> 42
    # 1 block of 2x1x6 using rank 1 -> 1
    # Total rank 43
    
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U coeffs for A (rows 0,1; cols 0,1)
    # M1: a=1, d=1
    # M2: c=1, d=1
    # M3: a=1
    # M4: d=1
    # M5: a=1, b=1
    # M6: c=1, a=-1
    # M7: b=1, d=-1
    
    U_strassen = [
        [(1, 0), (1, 1)], # M1: A00, A11
        [(1, 1), (1, 0)], # M2: A10, A11 (c, d) -> A[1,0], A[1,1]
        [(1, 0)],         # M3: A00
        [(1, 1)],         # M4: A11
        [(1, 0), (0, 1)], # M5: A00, A01
        [(1, 1), (-1, 0)],# M6: A10, A00
        [(0, 1), (-1, 1)] # M7: A01, A11
    ]
    
    # V coeffs for B (rows 0,1; cols 0,1)
    # M1: e=1, h=1
    # M2: e=1
    # M3: f=1, h=-1
    # M4: g=1, e=-1
    # M5: h=1
    # M6: e=1, f=1
    # M7: g=1, h=1
    
    V_strassen = [
        [(0, 0), (1, 1)], # M1: B00, B11
        [(0, 0)],         # M2: B00
        [(0, 1), (-1, 1)],# M3: B01, B11
        [(1, 0), (-1, 0)],# M4: B10, B00
        [(1, 1)],         # M5: B11
        [(0, 0), (0, 1)], # M6: B00, B01
        [(1, 0), (1, 1)]  # M7: B10, B11
    ]
    
    # W coeffs for C (rows 0,1; cols 0,1)
    # Index in W is k*n + i.
    # C00 (i=0, k=0) -> W[0*2+0, r] = W[0, r]
    # C01 (i=0, k=1) -> W[1*2+0, r] = W[2, r]
    # C10 (i=1, k=0) -> W[0*2+1, r] = W[1, r]
    # C11 (i=1, k=1) -> W[1*2+1, r] = W[3, r]
    
    # M1: C00=1, C11=1
    # M2: C10=1, C11=-1
    # M3: C01=1, C11=1
    # M4: C00=1, C10=1
    # M5: C00=-1, C01=1
    # M6: C11=1
    # M7: C00=1
    
    W_strassen = [
        [(0, 1), (3, 1)], # M1: C00, C11
        [(1, 1), (3, -1)],# M2: C10, C11
        [(2, 1), (3, 1)], # M3: C01, C11
        [(0, 1), (1, 1)], # M4: C00, C10
        [(0, -1), (2, 1)],# M5: C00, C01
        [(3, 1)],         # M6: C11
        [(0, 1)]          # M7: C00
    ]
    
    current_rank = 0
    
    # Define the 6 Strassen blocks
    # A cols 0,1; B rows 0,1; B cols 0,1 -> C rows 0,1; C cols 0,1
    # A cols 0,1; B rows 0,1; B cols 2,3 -> C rows 0,1; C cols 2,3
    # A cols 0,1; B rows 0,1; B cols 4,5 -> C rows 0,1; C cols 4,5
    # A cols 2,3; B rows 2,3; B cols 0,1 -> C rows 0,1; C cols 0,1
    # A cols 2,3; B rows 2,3; B cols 2,3 -> C rows 0,1; C cols 2,3
    # A cols 2,3; B rows 2,3; B cols 4,5 -> C rows 0,1; C cols 4,5
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (2, 2, 0),
        (2, 2, 2),
        (2, 2, 4)
    ]
    
    for j_a, j_b, k_b in blocks:
        for r in range(7):
            col = current_rank + r
            # Fill U
            for (i, j), val in U_strassen[r]:
                U[i * m + (j_a + j)][col] = val
            # Fill V
            for (i, j), val in V_strassen[r]:
                V[(j_b + i) * p + (k_b + j)][col] = val
            # Fill W
            for (i, j), val in W_strassen[r]:
                # i is col index k, j is row index i?
                # In W_strassen, I used (k, val) where k is col index of C.
                # My W_strassen entries were (k_idx, val).
                # Let's re-verify W_strassen indices.
                # C00 (k=0, i=0) -> W[0*2+0] = W[0]
                # C01 (k=1, i=0) -> W[1*2+0] = W[2]
                # C10 (k=0, i=1) -> W[0*2+1] = W[1]
                # C11 (k=1, i=1) -> W[1*2+1] = W[3]
                
                # My W_strassen list:
                # M1: (0, 1) -> W[0], (3, 1) -> W[3]
                # This matches k=0,i=0 and k=1,i=1.
                # So the first element is k (col), second is val.
                # Wait, I stored (k, val).
                # So k_idx = element[0].
                # i_idx is implicit?
                # No, W index is k*n + i.
                # I need to know i as well.
                # Let's rewrite W_strassen to store (k, i, val).
                
                pass # Logic needs fixing in the list definition above
                
    # Let's redefine W_strassen with (k, i, val)
    W_strassen = [
        [(0, 0, 1), (1, 1, 1)], # M1: C00, C11
        [(0, 1, 1), (1, 1, -1)],# M2: C10, C11
        [(1, 0, 1), (1, 1, 1)], # M3: C01, C11
        [(0, 0, 1), (0, 1, 1)], # M4: C00, C10
        [(0, 0, -1), (1, 0, 1)],# M5: C00, C01
        [(1, 1, 1)],            # M6: C11
        [(0, 0, 1)]             # M7: C00
    ]
    
    current_rank = 0
    for j_a, j_b, k_b in blocks:
        for r in range(7):
            col = current_rank + r
            # Fill U
            for (i, j), val in U_strassen[r]:
                U[i * m + (j_a + j)][col] = val
            # Fill V
            for (i, j), val in V_strassen[r]:
                V[(j_b + i) * p + (k_b + j)][col] = val
            # Fill W
            for (k, i, val) in W_strassen[r]:
                W[(k_b + k) * n + (j_a // 2 * 2 + i)] # i is local row, global row is j_a//2*2 + i?
                # No, global row i is just i (0 or 1) because n=2 and we don't split n.
                # The blocks are for A cols and B rows/cols.
                # A rows are always 0,1.
                # So global i is local i.
                W[(k_b + k) * n + i][col] = val
            current_rank += 1 # This is wrong, rank increases by 7 per block
            
    # Fix loop
    current_rank = 0
    for j_a, j_b, k_b in blocks:
        for r in range(7):
            col = current_rank + r
            for (i, j), val in U_strassen[r]:
                U[i * m + (j_a + j)][col] = val
            for (i, j), val in V_strassen[r]:
                V[(j_b + i) * p + (k_b + j)][col] = val
            for (k, i, val) in W_strassen[r]:
                W[(k_b + k) * n + i][col] = val
        current_rank += 7
        
    # Last block: 2x1x6
    # A col 4, B row 4, B cols 0-5
    # Rank 1
    col = current_rank
    # U: A[0,4]=1, A[1,4]=1
    U[0 * m + 4][col] = 1
    U[1 * m + 4][col] = 1
    # V: B[4,0..5]=1
    for k in range(6):
        V[4 * p + k][col] = 1
    # W: C[0..5, 0..1]=1
    for k in range(6):
        for i in range(2):
            W[k * n + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
