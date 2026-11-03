import time

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M0: (A00+A11)(B00+B11) -> C00+=1, C01+=0, C10+=0, C11+=1
    # M1: (A10+A11)B00       -> C00+=0, C01+=0, C10+=1, C11+=1
    # M2: A00(B01-B11)       -> C00+=0, C01+=1, C10+=0, C11+=-1
    # M3: A11(B10-B00)       -> C00+=-1, C01+=0, C10+=1, C11+=0
    # M4: (A00+A01)B11       -> C00+=0, C01+=1, C10+=0, C11+=0
    # M5: (A10-A00)(B00+B01) -> C00+=1, C01+=0, C10+=0, C11+=0
    # M6: (A01-A11)(B10+B11) -> C00+=-1, C01+=0, C10+=0, C11+=1
    
    # Local A coeffs (2x2), Local B coeffs (2x2), W coeffs (2x2 for C)
    # Order: A00, A01, A10, A11
    strassen_terms = [
        # M0
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
        # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]),
        # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1])
    ]
    
    col = 0
    # 4 groups of k (pairs)
    for g in range(4):
        k_start = g * 2
        # 2 parts of j (pairs)
        for s in range(2):
            j_start = s * 2
            # 7 Strassen terms
            for t in range(7):
                a_coeffs, b_coeffs, w_coeffs = strassen_terms[t]
                
                # Fill U
                # a_coeffs are for local A[0,0], A[0,1], A[1,0], A[1,1]
                # Global A[i, j]
                # i=0, j=j_start+0 -> idx 0*m + j_start
                # i=0, j=j_start+1 -> idx 0*m + j_start+1
                # i=1, j=j_start+0 -> idx 1*m + j_start
                # i=1, j=j_start+1 -> idx 1*m + j_start+1
                
                U[0 * m + j_start][col] = a_coeffs[0]
                U[0 * m + j_start + 1][col] = a_coeffs[1]
                U[1 * m + j_start][col] = a_coeffs[2]
                U[1 * m + j_start + 1][col] = a_coeffs[3]
                
                # Fill V
                # b_coeffs are for local B[0,0], B[0,1], B[1,0], B[1,1]
                # Global B[j, k]
                # j=j_start+0, k=k_start+0 -> idx (j_start)*p + k_start
                # j=j_start+0, k=k_start+1 -> idx (j_start)*p + k_start+1
                # j=j_start+1, k=k_start+0 -> idx (j_start+1)*p + k_start
                # j=j_start+1, k=k_start+1 -> idx (j_start+1)*p + k_start+1
                
                V[j_start * p + k_start][col] = b_coeffs[0]
                V[j_start * p + k_start + 1][col] = b_coeffs[1]
                V[(j_start + 1) * p + k_start][col] = b_coeffs[2]
                V[(j_start + 1) * p + k_start + 1][col] = b_coeffs[3]
                
                # Fill W
                # w_coeffs are for local C[0,0], C[0,1], C[1,0], C[1,1]
                # Global C[i, k] stored as W[k*n + i]
                # i=0, k=k_start+0 -> idx (k_start)*n + 0
                # i=0, k=k_start+1 -> idx (k_start+1)*n + 0
                # i=1, k=k_start+0 -> idx (k_start)*n + 1
                # i=1, k=k_start+1 -> idx (k_start+1)*n + 1
                
                W[k_start * n + 0][col] = w_coeffs[0]
                W[k_start * n + 1][col] = w_coeffs[1]
                W[k_start * n + 0 + 1][col] = w_coeffs[2] # Wait, index is k*n+i.
                # k=k_start, i=1 -> k_start*n + 1
                # k=k_start+1, i=1 -> (k_start+1)*n + 1
                
                # Correction:
                # C[0,0] -> i=0, k=k_start -> W[k_start*n + 0]
                # C[0,1] -> i=0, k=k_start+1 -> W[(k_start+1)*n + 0]
                # C[1,0] -> i=1, k=k_start -> W[k_start*n + 1]
                # C[1,1] -> i=1, k=k_start+1 -> W[(k_start+1)*n + 1]
                
                W[k_start * n + 0][col] = w_coeffs[0]
                W[(k_start + 1) * n + 0][col] = w_coeffs[1]
                W[k_start * n + 1][col] = w_coeffs[2]
                W[(k_start + 1) * n + 1][col] = w_coeffs[3]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
