"""Matrix multiplication 2,2,16 rank decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # Schoolbook rank 64.
    # Strassen rank for 8 blocks of 2x2: 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen linear forms for A (2x2)
    # M1: a+d
    # M2: c+d
    # M3: a
    # M4: d
    # M5: a+b
    # M6: c-a
    # M7: b-d
    
    # Strassen linear forms for B (2x2) and C (2x2) reconstruction
    # We define coefficients for A, B, C for each of the 7 products.
    # Indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h (local to block)
    # C: 0=C00, 1=C01, 2=C10, 3=C11 (local to block)
    
    # Product definitions: (A_coeffs, B_coeffs, C_coeffs)
    # C is the output of the product term in the sum.
    # Actually, we need W such that sum_r sum_l U[:,r,l] V[:,r,l] W[:,r,l] reconstructs C.
    # So W contains the coefficients of C_{ij} in the linear combination of products.
    
    strassen_forms = [
        # l=0, M1=(a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C00 += 1, C11 += 1
        # l=1, M2=(c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # C10 += 1, C11 += -1
        # l=2, M3=a(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # C01 += 1, C11 += 1
        # l=3, M4=d(g-e)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]),  # C00 += 1, C10 += 1
        # l=4, M5=(a+b)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # C00 += -1, C01 += 1
        # l=5, M6=(c-a)(e+f)
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # C11 += 1
        # l=6, M7=(b-d)(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # C00 += 1
    ]
    
    col = 0
    num_blocks = p // 2 # 8 blocks
    
    for r in range(num_blocks):
        for l in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_forms[l]
            
            # Fill U (A part)
            # A is 2x2, flattened row-major: a(0), b(1), c(2), d(3)
            U[0, col] = a_coeffs[0]
            U[1, col] = a_coeffs[1]
            U[2, col] = a_coeffs[2]
            U[3, col] = a_coeffs[3]
            
            # Fill V (B part)
            # B is 2x16. Block r uses cols 2r and 2r+1.
            # B indices: j*16 + k.
            # Local B: e(0,2r), f(1,2r), g(0,2r+1), h(1,2r+1)
            # Wait, standard 2x2 matrix B_local = [[e, f], [g, h]]
            # e = B[0, 2r] -> index 0*16 + 2r
            # f = B[1, 2r] -> index 1*16 + 2r
            # g = B[0, 2r+1] -> index 0*16 + 2r+1
            # h = B[1, 2r+1] -> index 1*16 + 2r+1
            
            V[0 * 16 + 2 * r, col] = b_coeffs[0]
            V[1 * 16 + 2 * r, col] = b_coeffs[1]
            V[0 * 16 + (2 * r + 1), col] = b_coeffs[2]
            V[1 * 16 + (2 * r + 1), col] = b_coeffs[3]
            
            # Fill W (C part)
            # C is 2x16. Block r uses cols 2r and 2r+1.
            # W index: k*n + i = k*2 + i.
            # Local C: C00(0,2r), C01(0,2r+1), C10(1,2r), C11(1,2r+1)
            # c_coeffs: [C00, C01, C10, C11]
            
            # C00 at i=0, k=2r -> idx 2r*2 + 0
            W[2 * r * 2 + 0, col] = c_coeffs[0]
            # C01 at i=0, k=2r+1 -> idx (2r+1)*2 + 0
            W[(2 * r + 1) * 2 + 0, col] = c_coeffs[1]
            # C10 at i=1, k=2r -> idx 2r*2 + 1
            W[2 * r * 2 + 1, col] = c_coeffs[2]
            # C11 at i=1, k=2r+1 -> idx (2r+1)*2 + 1
            W[(2 * r + 1) * 2 + 1, col] = c_coeffs[3]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
