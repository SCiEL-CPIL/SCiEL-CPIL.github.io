import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A is at rows a_i, cols a_j
    # B is at rows b_j, cols b_k
    # C is at rows c_k, cols c_i
    # We need to map local 2x2 indices to global indices
    def add_strassen_block(a_i0, a_j0, b_j0, b_k0, c_k0, c_i0):
        nonlocal col
        # Local indices for A: (0,0)->(a_i0, a_j0), (0,1)->(a_i0, a_j0+1), (1,0)->(a_i0+1, a_j0), (1,1)->(a_i0+1, a_j0+1)
        # Local indices for B: (0,0)->(b_j0, b_k0), ...
        # Local indices for C: (0,0)->(c_k0, c_i0), ...
        
        # Strassen intermediates M1..M7
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11.
        U[a_i0*4 + a_j0][col] = 1
        U[(a_i0+1)*4 + (a_j0+1)][col] = 1
        V[b_j0*5 + b_k0][col] = 1
        V[(b_j0+1)*5 + (b_k0+1)][col] = 1
        # W: C00, C11
        W[b_k0*2 + c_i0][col] = 1
        W[(b_k0+1)*2 + (c_i0+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a_i0*4 + a_j0][col] = 1 # A10? No, A10 is (1,0). A11 is (1,1).
        # Wait, Strassen M2 = (c+d)e. c=A10, d=A11, e=B00.
        U[(a_i0+1)*4 + a_j0][col] = 1
        U[(a_i0+1)*4 + (a_j0+1)][col] = 1
        V[b_j0*5 + b_k0][col] = 1
        W[b_k0*2 + (c_i0+1)][col] = 1 # C21
        col += 1
        
        # M3 = A00(B01-B11)
        U[a_i0*4 + a_j0][col] = 1
        V[b_j0*5 + (b_k0+1)][col] = 1
        V[(b_j0+1)*5 + (b_k0+1)][col] = -1
        W[b_k0*2 + c_i0][col] = 1 # C11? No C12.
        # C12 is (0,1) in C. Index c_k0, c_i0+1.
        W[b_k0*2 + (c_i0+1)][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(a_i0+1)*4 + (a_j0+1)][col] = 1
        V[(b_j0+1)*5 + b_k0][col] = 1
        V[b_j0*5 + b_k0][col] = -1
        W[(b_k0+1)*2 + c_i0][col] = 1 # C21? No C21 is (1,0).
        # C21 index: k=1, i=0.
        W[(b_k0+1)*2 + c_i0][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a_i0*4 + a_j0][col] = 1
        U[a_i0*4 + (a_j0+1)][col] = 1
        V[(b_j0+1)*5 + (b_k0+1)][col] = 1
        W[(b_k0+1)*2 + (c_i0+1)][col] = 1 # C22
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(a_i0+1)*4 + a_j0][col] = 1
        U[a_i0*4 + a_j0][col] = -1
        V[b_j0*5 + b_k0][col] = 1
        V[b_j0*5 + (b_k0+1)][col] = 1
        W[(b_k0+1)*2 + (c_i0+1)][col] = 1 # C22
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_i0*4 + (a_j0+1)][col] = 1
        U[(a_i0+1)*4 + (a_j0+1)][col] = -1
        V[(b_j0+1)*5 + b_k0][col] = 1
        V[(b_j0+1)*5 + (b_k0+1)][col] = 1
        W[b_k0*2 + c_i0][col] = 1 # C11
        col += 1
        
        # Now add the combinations to W
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We already added 1s to W for the "main" terms.
        # We need to add the remaining coefficients.
        
        # Let's restart this helper to be cleaner.
        pass

    # The above manual addition is error-prone. 
    # Better to define the 7 rank-1 tensors for Strassen and sum them.
    
    def add_strassen(a_i0, a_j0, b_j0, b_k0, c_k0, c_i0):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00, C11
        # M2: (A10+A11)B00 -> C10
        # M3: A00(B01-B11) -> C01
        # M4: A11(B10-B00) -> C10
        # M5: (A00+A01)B11 -> C11
        # M6: (A10-A00)(B00+B01) -> C11
        # M7: (A01-A11)(B10+B11) -> C00
        
        # 1. M1
        U[a_i0*4 + a_j0][col] += 1
        U[(a_i0+1)*4 + (a_j0+1)][col] += 1
        V[b_j0*5 + b_k0][col] += 1
        V[(b_j0+1)*5 + (b_k0+1)][col] += 1
        W[b_k0*2 + c_i0][col] += 1
        W[(b_k0+1)*2 + (c_i0+1)][col] += 1
        col += 1
        
        # 2. M2
        U[(a_i0+1)*4 + a_j0][col] += 1
        U[(a_i0+1)*4 + (a_j0+1)][col] += 1
        V[b_j0*5 + b_k0][col] += 1
        W[(b_k0+1)*2 + c_i0][col] += 1
        col += 1
        
        # 3. M3
        U[a_i0*4 + a_j0][col] += 1
        V[b_j0*5 + (b_k0+1)][col] += 1
        V[(b_j0+1)*5 + (b_k0+1)][col] += -1
        W[b_k0*2 + (c_i0+1)][col] += 1
        col += 1
        
        # 4. M4
        U[(a_i0+1)*4 + (a_j0+1)][col] += 1
        V[(b_j0+1)*5 + b_k0][col] += 1
        V[b_j0*5 + b_k0][col] += -1
        W[(b_k0+1)*2 + c_i0][col] += 1
        col += 1
        
        # 5. M5
        U[a_i0*4 + a_j0][col] += 1
        U[a_i0*4 + (a_j0+1)][col] += 1
        V[(b_j0+1)*5 + (b_k0+1)][col] += 1
        W[(b_k0+1)*2 + (c_i0+1)][col] += 1
        col += 1
        
        # 6. M6
        U[(a_i0+1)*4 + a_j0][col] += 1
        U[a_i0*4 + a_j0][col] += -1
        V[b_j0*5 + b_k0][col] += 1
        V[b_j0*5 + (b_k0+1)][col] += 1
        W[(b_k0+1)*2 + (c_i0+1)][col] += 1
        col += 1
        
        # 7. M7
        U[a_i0*4 + (a_j0+1)][col] += 1
        U[(a_i0+1)*4 + (a_j0+1)][col] += -1
        V[(b_j0+1)*5 + b_k0][col] += 1
        V[(b_j0+1)*5 + (b_k0+1)][col] += 1
        W[b_k0*2 + c_i0][col] += 1
        col += 1
        
        # Now apply the linear combinations for C
        # C00 (k=0, i=0) = M1 + M4 - M5 + M7
        # M1 is at col-7, M4 at col-4, M5 at col-3, M7 at col-1
        W[b_k0*2 + c_i0][col-7] += 0 # Already 1
        W[b_k0*2 + c_i0][col-4] += 1 # M4
        W[b_k0*2 + c_i0][col-3] += -1 # M5
        W[b_k0*2 + c_i0][col-1] += 1 # M7
        
        # C01 (k=0, i=1) = M3 + M5
        # M3 at col-5, M5 at col-3
        W[b_k0*2 + (c_i0+1)][col-5] += 0 # Already 1
        W[b_k0*2 + (c_i0+1)][col-3] += 1 # M5
        
        # C10 (k=1, i=0) = M2 + M4
        # M2 at col-6, M4 at col-4
        W[(b_k0+1)*2 + c_i0][col-6] += 0 # Already 1
        W[(b_k0+1)*2 + c_i0][col-4] += 1 # M4
        
        # C11 (k=1, i=1) = M1 - M2 + M3 + M6
        # M1 at col-7, M2 at col-6, M3 at col-5, M6 at col-2
        W[(b_k0+1)*2 + (c_i0+1)][col-7] += 0 # Already 1
        W[(b_k0+1)*2 + (c_i0+1)][col-6] += -1 # M2
        W[(b_k0+1)*2 + (c_i0+1)][col-5] += 1 # M3
        W[(b_k0+1)*2 + (c_i0+1)][col-2] += 1 # M6

    def add_standard_221(a_i0, a_j0, b_j0, b_k0, c_k0, c_i0):
        nonlocal col
        # C00 = A00 B00
        U[a_i0*4 + a_j0][col] += 1
        V[b_j0*5 + b_k0][col] += 1
        W[b_k0*2 + c_i0][col] += 1
        col += 1
        # C01 = A01 B00
        U[a_i0*4 + (a_j0+1)][col] += 1
        V[b_j0*5 + b_k0][col] += 1
        W[b_k0*2 + (c_i0+1)][col] += 1
        col += 1
        # C10 = A10 B00
        U[(a_i0+1)*4 + a_j0][col] += 1
        V[b_j0*5 + b_k0][col] += 1
        W[(b_k0+1)*2 + c_i0][col] += 1
        col += 1
        # C11 = A11 B00
        U[(a_i0+1)*4 + (a_j0+1)][col] += 1
        V[b_j0*5 + b_k0][col] += 1
        W[(b_k0+1)*2 + (c_i0+1)][col] += 1
        col += 1

    # 1. C[:, 0:4]
    # A[:, 0:2] * B[0:2, 0:4]
    add_strassen(0, 0, 0, 0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 2, 2, 0) # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    
    # A[:, 2:4] * B[2:4, 0:4]
    add_strassen(0, 2, 2, 0, 0, 0) # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 2, 2, 2, 2, 0) # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]

    # 2. C[:, 4]
    # A[:, 0:2] * B[0:2, 4]
    add_standard_221(0, 0, 0, 4, 4, 0)
    
    # A[:, 2:4] * B[2:4, 4]
    add_standard_221(0, 2, 2, 4, 4, 0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
