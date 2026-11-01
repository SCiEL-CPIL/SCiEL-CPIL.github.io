"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # Rank 36 construction
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col_idx
        U[u_idx][col_idx] += u_val
        V[v_idx][col_idx] += v_val
        W[w_idx][col_idx] += w_val
        col_idx += 1

    # Block 1: Left part (Cols 0,1)
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # This is a 2x4 x 4x2 multiplication.
    # Schoolbook for 2x4 x 4x2 uses 8 mults.
    # Indices in U (n*m = 8): A row 0 is 0,1; A row 1 is 2,3? No, A is 2x4.
    # U index: i*m + j. i in 0..1, j in 0..3.
    # V index: j*p + k. j in 0..3, k in 0..4.
    # W index: k*n + i. k in 0..4, i in 0..1.
    
    # Term 1: A[0,0] * B[0,0] -> C[0,0]
    add_term(0, 0, 0, 1, 1, 1) # A00*B00 -> C00
    # Term 2: A[0,1] * B[1,0] -> C[0,0]
    add_term(1, 4, 0, 1, 1, 1) # A01*B10 -> C00
    # Term 3: A[1,0] * B[0,0] -> C[1,0]
    add_term(4, 0, 1, 1, 1, 1) # A10*B00 -> C10
    # Term 4: A[1,1] * B[1,0] -> C[1,0]
    add_term(5, 4, 1, 1, 1, 1) # A11*B10 -> C10
    
    # Term 5: A[0,2] * B[2,0] -> C[0,0]
    add_term(2, 8, 0, 1, 1, 1) # A02*B20 -> C00
    # Term 6: A[0,3] * B[3,0] -> C[0,0]
    add_term(3, 12, 0, 1, 1, 1) # A03*B30 -> C00
    # Term 7: A[1,2] * B[2,0] -> C[1,0]
    add_term(6, 8, 1, 1, 1, 1) # A12*B20 -> C10
    # Term 8: A[1,3] * B[3,0] -> C[1,0]
    add_term(7, 12, 1, 1, 1, 1) # A13*B30 -> C10
    
    # Col 1 part of Left Block
    # Term 9: A[0,0] * B[0,1] -> C[0,1]
    add_term(0, 1, 1+2, 1, 1, 1) # A00*B01 -> C01 (W index k=1, i=0 -> 1*2+0=2)
    # Term 10: A[0,1] * B[1,1] -> C[0,1]
    add_term(1, 5, 1+2, 1, 1, 1) # A01*B11 -> C01
    # Term 11: A[1,0] * B[0,1] -> C[1,1]
    add_term(4, 1, 1+2+2, 1, 1, 1) # A10*B01 -> C11 (W index k=1, i=1 -> 1*2+1=3)
    # Term 12: A[1,1] * B[1,1] -> C[1,1]
    add_term(5, 5, 1+2+2, 1, 1, 1) # A11*B11 -> C11
    
    # Term 13: A[0,2] * B[2,1] -> C[0,1]
    add_term(2, 9, 1+2, 1, 1, 1) # A02*B21 -> C01
    # Term 14: A[0,3] * B[3,1] -> C[0,1]
    add_term(3, 13, 1+2, 1, 1, 1) # A03*B31 -> C01
    # Term 15: A[1,2] * B[2,1] -> C[1,1]
    add_term(6, 9, 1+2+2, 1, 1, 1) # A12*B21 -> C11
    # Term 16: A[1,3] * B[3,1] -> C[1,1]
    add_term(7, 13, 1+2+2, 1, 1, 1) # A13*B31 -> C11
    
    # Block 2: Right part (Cols 2,3)
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    
    # Col 2
    # Term 17: A[0,0] * B[0,2] -> C[0,2]
    add_term(0, 2, 2+2, 1, 1, 1) # W k=2, i=0 -> 2*2+0=4
    # Term 18: A[0,1] * B[1,2] -> C[0,2]
    add_term(1, 6, 2+2, 1, 1, 1)
    # Term 19: A[1,0] * B[0,2] -> C[1,2]
    add_term(4, 2, 2+2+2, 1, 1, 1) # W k=2, i=1 -> 5
    # Term 20: A[1,1] * B[1,2] -> C[1,2]
    add_term(5, 6, 2+2+2, 1, 1, 1)
    
    # Term 21: A[0,2] * B[2,2] -> C[0,2]
    add_term(2, 10, 2+2, 1, 1, 1)
    # Term 22: A[0,3] * B[3,2] -> C[0,2]
    add_term(3, 14, 2+2, 1, 1, 1)
    # Term 23: A[1,2] * B[2,2] -> C[1,2]
    add_term(6, 10, 2+2+2, 1, 1, 1)
    # Term 24: A[1,3] * B[3,2] -> C[1,2]
    add_term(7, 14, 2+2+2, 1, 1, 1)
    
    # Col 3
    # Term 25: A[0,0] * B[0,3] -> C[0,3]
    add_term(0, 3, 3+2, 1, 1, 1) # W k=3, i=0 -> 6
    # Term 26: A[0,1] * B[1,3] -> C[0,3]
    add_term(1, 7, 3+2, 1, 1, 1)
    # Term 27: A[1,0] * B[0,3] -> C[1,3]
    add_term(4, 3, 3+2+2, 1, 1, 1) # W k=3, i=1 -> 7
    # Term 28: A[1,1] * B[1,3] -> C[1,3]
    add_term(5, 7, 3+2+2, 1, 1, 1)
    
    # Term 29: A[0,2] * B[2,3] -> C[0,3]
    add_term(2, 11, 3+2, 1, 1, 1)
    # Term 30: A[0,3] * B[3,3] -> C[0,3]
    add_term(3, 15, 3+2, 1, 1, 1)
    # Term 31: A[1,2] * B[2,3] -> C[1,3]
    add_term(6, 11, 3+2+2, 1, 1, 1)
    # Term 32: A[1,3] * B[3,3] -> C[1,3]
    add_term(7, 15, 3+2+2, 1, 1, 1)
    
    # Block 3: Last column (Col 4)
    # C[:, 4] = A[:, 0:2] * B[0:2, 4] + A[:, 2:4] * B[2:4, 4]
    
    # Term 33: A[0,0] * B[0,4] -> C[0,4]
    add_term(0, 4, 4+2, 1, 1, 1) # W k=4, i=0 -> 8
    # Term 34: A[0,1] * B[1,4] -> C[0,4]
    add_term(1, 5+4, 4+2, 1, 1, 1) # B[1,4] is V index 1*5+4=9
    # Term 35: A[1,0] * B[0,4] -> C[1,4]
    add_term(4, 4, 4+2+2, 1, 1, 1) # W k=4, i=1 -> 9
    # Term 36: A[1,1] * B[1,4] -> C[1,4]
    add_term(5, 5+4, 4+2+2, 1, 1, 1)
    
    # Wait, I missed A[:, 2:4] for the last column?
    # Yes, I need 4 more terms for the right half of A.
    # This would make rank 40 again.
    
    # The logic that "Last column is 4 mults" was flawed because it ignored the A2 part.
    # So schoolbook is 40.
    
    # I will return the schoolbook rank 40 decomposition as it is guaranteed correct.
    # If I can't find a verified <40 construction quickly, I must not risk an invalid one.
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][c] = 1
                V_s[j * p + k][c] = 1
                W_s[k * n + i][c] = 1
                c += 1
                
    if time.time() > deadline:
         return {"U": U_s, "V": V_s, "W": W_s}
         
    return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
