import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    # A is n x m = 2x2, B is m x p = 2x10, C is n x p = 2x10
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We use the Strassen algorithm over the ring R = F^5.
    # View B as a 2x2 matrix over R, where B[0,0] = B[:, 0:5], B[0,1] = B[:, 5:10], etc.
    # A is a 2x2 matrix over F.
    # The product C = A * B is a 2x2 matrix over R.
    # Strassen computes this with 7 multiplications in R.
    # Each multiplication in R (scalar * vector) decomposes into 5 scalar multiplications (one per component).
    # Total rank = 7 * 5 = 35.
    
    # Indexing conventions:
    # A[i,j] is at index i*m + j = i*2 + j in the U matrix (size n*m = 4).
    # B[j,k] is at index j*p + k = j*10 + k in the V matrix (size m*p = 20).
    # C[i,k] is at index k*n + i = k*2 + i in the W matrix (size p*n = 20).
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen terms definitions:
    # A_coeffs[s]: list of (A_index, coeff)
    # B_coeffs[s]: list of (B_row, B_block, coeff) where B_block is 0 (left) or 1 (right)
    # C_contrib[s]: (coeff for C[0,0], coeff for C[0,1], coeff for C[1,0], coeff for C[1,1])
    # C[0,0] is C[0, 0:5], C[0,1] is C[0, 5:10], C[1,0] is C[1, 0:5], C[1,1] is C[1, 5:10]
    
    A_coeffs = [
        [(0, 1), (3, 1)],    # M1: A00 + A11
        [(2, 1), (3, 1)],    # M2: A10 + A11
        [(0, 1)],             # M3: A00
        [(3, 1)],             # M4: A11
        [(0, 1), (1, 1)],    # M5: A00 + A01
        [(2, 1), (0, -1)],   # M6: A10 - A00
        [(1, 1), (3, -1)],   # M7: A01 - A11
    ]
    
    B_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1: B[0,0] + B[1,1]
        [(0, 0, 1)],               # M2: B[0,0]
        [(0, 1, 1), (1, 1, -1)],  # M3: B[0,1] - B[1,1]
        [(1, 0, 1), (0, 0, -1)],  # M4: B[1,0] - B[0,0]
        [(1, 1, 1)],               # M5: B[1,1]
        [(0, 0, 1), (0, 1, 1)],   # M6: B[0,0] + B[0,1]
        [(1, 0, 1), (1, 1, 1)],   # M7: B[1,0] + B[1,1]
    ]
    
    C_contrib = [
        (1, 0, 0, 1),    # M1 -> C00 += M1, C11 += M1
        (0, 0, 1, -1),   # M2 -> C10 += M2, C11 -= M2
        (0, 1, 0, 1),    # M3 -> C01 += M3, C11 += M3
        (1, 0, 1, 0),    # M4 -> C00 += M4, C10 += M4
        (-1, 1, 0, 0),   # M5 -> C00 -= M5, C01 += M5
        (0, 0, 0, 1),    # M6 -> C11 += M6
        (1, 0, 0, 0),    # M7 -> C00 += M7
    ]
    
    # Verify Strassen formula:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Let's check the contributions:
    # M1: C00+=1, C11+=1. Correct.
    # M2: C10+=1, C11-=-1. Correct.
    # M3: C01+=1, C11+=1. Correct.
    # M4: C00+=1, C10+=1. Correct.
    # M5: C00-=-1, C01+=1. Correct.
    # M6: C11+=1. Correct.
    # M7: C00+=1. Correct.
    
    col = 0
    for s in range(7):
        a_list = A_coeffs[s]
        b_list = B_coeffs[s]
        c00_c, c01_c, c10_c, c11_c = C_contrib[s]
        
        for t in range(5):
            # U part: coefficients for A
            for a_idx, a_c in a_list:
                U[a_idx][col] = a_c
            
            # V part: coefficients for B
            for b_row, b_block, b_c in b_list:
                b_col = b_block * 5 + t  # actual column in B (0-4 for block 0, 5-9 for block 1)
                V[b_row * p + b_col][col] = b_c
            
            # W part: coefficients for C
            # C[0,0] block is C[0, 0:5], indices 0, 2, 4, 6, 8
            # C[0,1] block is C[0, 5:10], indices 10, 12, 14, 16, 18
            # C[1,0] block is C[1, 0:5], indices 1, 3, 5, 7, 9
            # C[1,1] block is C[1, 5:10], indices 11, 13, 15, 17, 19
            
            if c00_c != 0:
                W[t * n + 0][col] = c00_c  # C[0, t], index = t*2 + 0
            if c01_c != 0:
                W[(5 + t) * n + 0][col] = c01_c  # C[0, 5+t], index = (5+t)*2 + 0
            if c10_c != 0:
                W[t * n + 1][col] = c10_c  # C[1, t], index = t*2 + 1
            if c11_c != 0:
                W[(5 + t) * n + 1][col] = c11_c  # C[1, 5+t], index = (5+t)*2 + 1
            
            col += 1
    
    # Optional: Verify the decomposition
    # U is 4x35, V is 20x35, W is 20x35
    # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # should equal delta_{i,j} * (C[i,k] coefficient) ... wait.
    # The tensor for matrix multiplication (n,m,p) is T[i, j, k] = 1 if i*m+j == k? No.
    # The tensor T is defined such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So T[i, j, k] is 1 if the indices match, else 0.
    # Specifically, for (2,2,10):
    # T[i, j, k] is non-zero only if the multiplication A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i, j, k] = 1 if the j in A[i,j] matches the j in B[j,k].
    # In our flattened indices:
    # U index u = i*m + j
    # V index v = j*p + k
    # W index w = k*n + i
    # The decomposition is sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w]
    # T[u,v,w] = 1 if u's j == v's j and v's k == w's k and u's i == w's i.
    # Let's check a few entries.
    
    # This verification is not strictly necessary for the submission but good for debugging.
    # We trust the Strassen construction.
    
    return {"U": U, "V": V, "W": W}
