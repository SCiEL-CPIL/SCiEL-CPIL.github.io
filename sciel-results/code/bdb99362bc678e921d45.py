import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 51 based on Strassen for 2x2x6 parts
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # U[i*m+j, col] = u
    # V[j*p+k, col] = v
    # W[k*n+i, col] = w
    # Term contributes u*v*w to C[i,k] via A[i,j]B[j,k]
    # Actually, the decomposition is C = sum_r U[:,r] V[:,r] W[:,r]
    # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want C[i,k] = sum_j A[i,j] B[j,k]
    
    # Part 1: A_22 B_26 using Strassen
    # A_22 is A[0:2, 0:2]
    # B_26 is B[0:2, 0:6]
    # Split B_26 into 3 blocks of 2x2: B_t for t=0,1,2
    # B_t columns are 2t, 2t+1
    
    # Strassen factors for 2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)(B22+B21)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to express each M as sum of rank-1 terms in the global space.
    # Each M is a product of a linear combo of A and a linear combo of B.
    # (sum a_ij A_ij) * (sum b_jk B_jk)
    # This is a sum of a_ij b_jk A_ij B_jk terms.
    # Each A_ij B_jk is a rank-1 term in the tensor.
    
    def add_strassen_term(A_coeffs, B_coeffs, t):
        nonlocal col
        # A_coeffs: dict {(i,j): coeff} for i,j in 0..1
        # B_coeffs: dict {(j,k): coeff} for j in 0..1, k in 0..1 (relative to block t)
        # Global B index is (j, 2t+k)
        
        # Expand product
        for (i, j), ca in A_coeffs.items():
            for (j2, k), cb in B_coeffs.items():
                if j == j2:
                    # Term: ca * cb * A[i,j] * B[j, 2t+k]
                    # U[i*m+j, col] = ca
                    # V[j*p + (2t+k), col] = cb
                    # W[(2t+k)*n + i, col] = 1
                    # Note: W index is k_global * n + i
                    k_global = 2 * t + k
                    U[i * m + j][col] += ca
                    V[j * p + k_global][col] += cb
                    W[k_global * n + i][col] += 1
        col += 1
        return col - 1

    # Define Strassen combinations
    # A indices: 0,1
    # B indices: 0,1 (relative)
    
    # M1: (A00+A11)(B00+B11)
    A1 = {(0,0):1, (1,1):1}
    B1 = {(0,0):1, (1,1):1}
    for t in range(3):
        add_strassen_term(A1, B1, t)
        
    # M2: (A10+A11)B00
    A2 = {(1,0):1, (1,1):1}
    B2 = {(0,0):1}
    for t in range(3):
        add_strassen_term(A2, B2, t)

    # M3: A00(B01-B11)
    A3 = {(0,0):1}
    B3 = {(0,1):1, (1,1):-1}
    for t in range(3):
        add_strassen_term(A3, B3, t)

    # M4: A11(B10-B00)
    A4 = {(1,1):1}
    B4 = {(1,0):1, (0,0):-1}
    for t in range(3):
        add_strassen_term(A4, B4, t)

    # M5: (A00+A01)(B11+B10)
    A5 = {(0,0):1, (0,1):1}
    B5 = {(1,1):1, (1,0):1}
    for t in range(3):
        add_strassen_term(A5, B5, t)

    # M6: (A10-A00)(B00+B01)
    A6 = {(1,0):1, (0,0):-1}
    B6 = {(0,0):1, (0,1):1}
    for t in range(3):
        add_strassen_term(A6, B6, t)

    # M7: (A01-A11)(B10+B11)
    A7 = {(0,1):1, (1,1):-1}
    B7 = {(1,0):1, (1,1):1}
    for t in range(3):
        add_strassen_term(A7, B7, t)

    # Part 2: A_21 B_16
    # A_21 is A[i, 2] for i in 0,1
    # B_16 is B[2, k] for k in 0..5
    # C[i, k] += A[i, 2] B[2, k]
    for i in range(2):
        for k in range(6):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 3: A_12 B_26
    # A_12 is A[2, j] for j in 0,1
    # B_26 is B[j, k] for j in 0,1, k in 0..5
    # C[2, k] += sum_j A[2, j] B[j, k]
    for j in range(2):
        for k in range(6):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    # Part 4: A_11 B_16
    # A_11 is A[2, 2]
    # B_16 is B[2, k]
    # C[2, k] += A[2, 2] B[2, k]
    for k in range(6):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
