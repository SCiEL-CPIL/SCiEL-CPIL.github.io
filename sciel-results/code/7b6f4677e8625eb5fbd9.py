import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition: 5 Strassen blocks (rank 7 each)
    rank = 35
    
    # U shape (n*m, rank) = (4, 35)
    # V shape (m*p, rank) = (20, 35)
    # W shape (p*n, rank) = (20, 35)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # A = [[a, b], [c, d]] -> indices 0,1,2,3 in U
    # B = [[e, f], [g, h]] -> indices 0,1,2,3 in V (local)
    # C = [[C00, C01], [C10, C11]] -> indices 0,1,2,3 in W (local)
    
    # The 7 Strassen products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Contributions to C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the factors for each term
    # U factors (a, b, c, d)
    # V factors (e, f, g, h)
    # W factors (C00, C01, C10, C11)
    
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for k in range(5):
        for u_vec, v_vec, w_vec in terms:
            # U: A indices are global 0..3
            for i in range(4):
                U[i][col] = u_vec[i]
            
            # V: B indices for block k
            # B[0, 2k] -> 2k
            # B[0, 2k+1] -> 2k+1
            # B[1, 2k] -> 10 + 2k
            # B[1, 2k+1] -> 10 + 2k + 1
            V[2*k][col] = v_vec[0]
            V[2*k+1][col] = v_vec[1]
            V[10+2*k][col] = v_vec[2]
            V[10+2*k+1][col] = v_vec[3]
            
            # W: C indices for block k
            # C[0, 2k] -> 2k * 2 + 0 = 4k
            # C[0, 2k+1] -> 2k * 2 + 1 = 4k+1
            # C[1, 2k] -> (2k+1) * 2 + 0 = 4k+2
            # C[1, 2k+1] -> (2k+1) * 2 + 1 = 4k+3
            # Note: W index is k*n + i. Here k is column index in 2x10?
            # The problem states W[k*n+i, r].
            # k is the column index of C (0..9). i is row index (0..1).
            # So W[2k + i, r].
            # Wait, the previous code used 4k, 4k+1, 4k+2, 4k+3.
            # Let's verify:
            # Block k corresponds to columns 2k and 2k+1.
            # C[0, 2k] -> index 2k * 2 + 0 = 4k. Correct.
            # C[0, 2k+1] -> index (2k+1) * 2 + 0 = 4k+2.
            # Wait.
            # W index = col * n + row.
            # C[0, 2k] is row 0, col 2k. Index = 2k * 2 + 0 = 4k.
            # C[0, 2k+1] is row 0, col 2k+1. Index = (2k+1) * 2 + 0 = 4k+2.
            # C[1, 2k] is row 1, col 2k. Index = 2k * 2 + 1 = 4k+1.
            # C[1, 2k+1] is row 1, col 2k+1. Index = (2k+1) * 2 + 1 = 4k+3.
            
            # My previous mapping in the code was:
            # W[4*t][col] = 1 (C00)
            # W[4*t+1][col] = 1 (C01?? No, C10)
            # Let's check the previous code's W mapping.
            # Previous code:
            # W[4*t][col] -> C[0, 2t]
            # W[4*t+1][col] -> C[1, 2t] ??
            # Let's re-read the previous code comments.
            # "Local indices for C (W) part of block t:
            #   k=0, i=0 -> global 2*2t + 0 = 4t
            #   k=0, i=1 -> global 2*2t + 1 = 4t+1
            #   k=1, i=0 -> global 2*(2t+1) + 0 = 4t+2
            #   k=1, i=1 -> global 2*(2t+1) + 1 = 4t+3"
            
            # So:
            # 4t: C[0, 2t]
            # 4t+1: C[1, 2t]
            # 4t+2: C[0, 2t+1]
            # 4t+3: C[1, 2t+1]
            
            # My w_vec is [C00, C01, C10, C11] in local block coords.
            # Local C00 -> Global C[0, 2k] -> 4k
            # Local C01 -> Global C[0, 2k+1] -> 4k+2
            # Local C10 -> Global C[1, 2k] -> 4k+1
            # Local C11 -> Global C[1, 2k+1] -> 4k+3
            
            W[4*k][col] = w_vec[0]
            W[4*k+2][col] = w_vec[1]
            W[4*k+1][col] = w_vec[2]
            W[4*k+3][col] = w_vec[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
