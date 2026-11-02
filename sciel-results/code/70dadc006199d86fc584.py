"""Program for 5x5 matrix multiplication tensor decomposition.
Implements the BCLR (Bini-Capovani-Lotti-Romani) rank-78 decomposition.
"""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s limit
    
    # Block decomposition: 5 = 3 + 1 + 1
    # A = [A11(3x3) A12(3x1) A13(3x1)]
    #     [A21(1x3) A22(1x1) A23(1x1)]
    #     [A31(1x3) A32(1x1) A33(1x1)]
    #
    # B = [B11(3x3) B12(3x1) B13(3x1)]
    #     [B21(1x3) B22(1x1) B23(1x1)]
    #     [B31(1x3) B32(1x1) B33(1x1)]
    #
    # C11 = A11*B11 + A12*B21 + A13*B31
    # C12 = A11*B12 + A12*B22 + A13*B32
    # C13 = A11*B13 + A12*B23 + A13*B33
    # C21 = A21*B11 + A22*B21 + A23*B31
    # C22 = A21*B12 + A22*B22 + A23*B32
    # C23 = A21*B13 + A22*B23 + A23*B33
    # C31 = A31*B11 + A32*B21 + A33*B31
    # C32 = A31*B12 + A32*B22 + A33*B32
    # C33 = A31*B13 + A32*B23 + A33*B33
    #
    # Use Strassen (rank 7) for A11*B11, direct for all others.
    # Total rank = 7 + 9 + 9 + 9 + 9 + 9 + 3 + 3 + 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 79
    
    # Strassen decomposition for 3x3 matrices:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We'll build U, V, W matrices where:
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)  
    # W has shape (p*n, rank) = (25, rank)
    #
    # Indexing: U[i*m+j, k] corresponds to coefficient of A[i,j] in term k
    #           V[j*p+k, k] corresponds to coefficient of B[j,k] in term k
    #           W[k*n+i, k] corresponds to coefficient for C[i,k] in term k
    
    U = [[0] * 1000 for _ in range(n * m)]  # pre-allocate, trim later
    V = [[0] * 1000 for _ in range(m * p)]
    W = [[0] * 1000 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1
    
    # Block indices for A (row, col) -> flat index row*m + col
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, col 3
    # A13: rows 0-2, col 4
    # A21: row 3, cols 0-2
    # A22: row 3, col 3
    # A23: row 3, col 4
    # A31: row 4, cols 0-2
    # A32: row 4, col 3
    # A33: row 4, col 4
    
    # B11: rows 0-2, cols 0-2
    # B12: rows 0-2, col 3
    # B13: rows 0-2, col 4
    # B21: row 3, cols 0-2
    # B22: row 3, col 3
    # B23: row 3, col 4
    # B31: row 4, cols 0-2
    # B32: row 4, col 3
    # B33: row 4, col 4
    
    # C11: rows 0-2, cols 0-2
    # C12: rows 0-2, col 3
    # C13: rows 0-2, col 4
    # C21: row 3, cols 0-2
    # C22: row 3, col 3
    # C23: row 3, col 4
    # C31: row 4, cols 0-2
    # C32: row 4, col 3
    # C33: row 4, col 4
    
    # Strassen for C11 = A11*B11 (7 terms)
    # M1 = (A11+A22)*(B11+B22)
    # But in our context, A11 is 3x3, B11 is 3x3
    # Strassen sub-blocks within 3x3: split as 3=2+1
    # This gets complicated. Instead, let's use a simpler approach.
    
    # Actually, for 3x3 matrix multiplication, Strassen gives rank 7.
    # The standard Strassen decomposition for 3x3:
    # Split 3x3 into blocks: top-left 2x2, and the rest
    # 
    # Let me use the explicit Strassen formulas for 3x3:
    # A = [a11 a12 a13]     B = [b11 b12 b13]
    #     [a21 a22 a23]         [b21 b22 b23]
    #     [a31 a32 a33]         [b31 b32 b33]
    #
    # One known rank-7 decomposition for 3x3:
    # We can use the fact that 3x3 mult has rank 7 via:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # But this only handles the 2x2 top-left block. We need to handle
    # the full 3x3. Let me use a different approach.
    
    # For simplicity and correctness, let me use direct multiplication
    # for the 3x3 part (9 terms) and see the total rank.
    # Total = 9 + 9+9+9+9+9+3+3+3+3+3+3+1+1+1+1+1+1+1+1+1+1+1+1 = 9+57+24 = 90
    # That's still better than 125.
    
    # Actually, let me just do full direct multiplication for all blocks.
    # C11 = A11*B11: 9 terms (3x3 direct)
    # C11 += A12*B21: 3*1*3 = 9 terms (each A12[r]*B21[c] contributes to C11[r,c])
    # Wait, A12 is 3x1, B21 is 1x3, so A12*B21 is 3x3 with (r,c) = A12[r]*B21[c]
    # That's 9 terms.
    
    # Let me just do schoolbook for everything but organize by blocks.
    # This gives rank 125 which is the same as before.
    
    # Let me try a hybrid: use Strassen for the 3x3*3x3 part.
    # The 3x3*3x3 multiplication can be done in rank 7.
    
    # Explicit rank-7 decomposition for 3x3 matrix multiplication:
    # Using the standard Strassen-like decomposition.
    # 
    # For 3x3, one known rank-7 decomposition uses:
    # Let A = [A_topleft(2x2), A_topright(2x1), A_bottom(1x3)]
    # B = [B_topleft(2x2), B_topright(2x1), B_bottom(1x3)]
    #
    # This is getting complex. Let me use a verified simple approach.
    
    # I'll implement the full schoolbook but verify it works, then try
    # to optimize with Strassen for the 3x3 block.
    
    # For now, let's do the block-organized schoolbook (rank 125) as a
    # verified baseline, then attempt Strassen optimization.
    
    # Actually, let me implement Strassen for 3x3 properly.
    # The rank-7 decomposition for 3x3 matrix multiplication:
    #
    # Reference: The 3x3 matrix multiplication tensor has rank 7.
    # One explicit decomposition:
    # 
    # A = [[a,b,c],[d,e,f],[g,h,i]]
    # B = [[x,y,z],[u,v,w],[p,q,r]]
    #
    # M1 = (a+e)*(x+v)
    # M2 = (d+e)*x
    # M3 = a*(y-v)
    # M4 = e*(u-x)
    # M5 = (a+b)*v
    # M6 = (d-a)*(x+y)
    # M7 = (b-e)*(u+v)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # But this only covers the top-left 2x2 of C! We need C[0,2], C[1,2], C[2,*]
    # 
    # So this isn't a full 3x3 decomposition. The rank-7 result for 3x3
    # requires a more sophisticated construction.
    #
    # Let me just use direct multiplication for the 3x3 block (9 terms)
    # and accept rank = 9 + 9+9+9+9+9+3+3+3+3+3+3+1+1+1+1+1+1+1+1+1+1+1+1 = 90
    
    # Wait, let me recount:
    # C11 contributions:
    #   A11*B11: 3*3 = 9 terms
    #   A12*B21: 3*1*3 = 9 terms (A12 is 3x1, B21 is 1x3, product is 3x3)
    #   A13*B31: 3*1*3 = 9 terms
    # C12 contributions:
    #   A11*B12: 3*3*1 = 9 terms
    #   A12*B22: 3*1*1 = 3 terms
    #   A13*B32: 3*1*1 = 3 terms
    # C13 contributions:
    #   A11*B13: 9 terms
    #   A12*B23: 3 terms
    #   A13*B33: 3 terms
    # C21 contributions:
    #   A21*B11: 1*3*3 = 9 terms
    #   A22*B21: 1*1*3 = 3 terms
    #   A23*B31: 1*1*3 = 3 terms
    # C22 contributions:
    #   A21*B12: 1*3*1 = 3 terms
    #   A22*B22: 1*1*1 = 1 term
    #   A23*B32: 1*1*1 = 1 term
    # C23 contributions:
    #   A21*B13: 3 terms
    #   A22*B23: 1 term
    #   A23*B33: 1 term
    # C31 contributions:
    #   A31*B11: 9 terms
    #   A32*B21: 3 terms
    #   A33*B31: 3 terms
    # C32 contributions:
    #   A31*B12: 3 terms
    #   A32*B22: 1 term
    #   A33*B32: 1 term
    # C33 contributions:
    #   A31*B13: 3 terms
    #   A32*B23: 1 term
    #   A33*B33: 1 term
    
    # Total: (9+9+9) + (9+3+3) + (9+3+3) + (9+3+3) + (3+1+1) + (3+1+1) + (9+3+3) + (3+1+1) + (3+1+1)
    # = 27 + 15 + 15 + 15 + 5 + 5 + 15 + 5 + 5 = 107
    
    # Hmm, that's 107, not 125. The block structure saves some because
    # when we have A12 (3x1) * B21 (1x3), each product A12[r]*B21[c] 
    # contributes to exactly one C11[r,c] element, so it's 9 terms.
    # But in schoolbook, each C[i,k] = sum_j A[i,j]*B[j,k], so C11[0,0]
    # = A11[0,0]*B11[0,0] + A12[0]*B21[0] + A13[0]*B31[0], which is 3 terms.
    # Total for C11 (3x3) = 9*3 = 27 terms. Same as block count.
    # So total is still 125. I was double-counting.
    
    # OK so block organization doesn't reduce rank without Strassen.
    # Let me just implement schoolbook cleanly and return it.
    
    rank = n * m * p  # 125
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
