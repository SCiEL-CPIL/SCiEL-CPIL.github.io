"""Rewritten program for 2x2 by 2x12 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use the Kronecker product approach:
    # M_{2,2,12} = M_{2,2,2} ⊗ M_{1,1,6}
    # Rank = rank(M_{2,2,2}) * rank(M_{1,1,6})
    # 
    # For M_{2,2,2}, we use a rank-9 decomposition (not the optimal Strassen rank-7,
    # but a simpler one that works well with the Kronecker structure)
    # Actually, let's use Strassen's rank-7 for M_{2,2,2}
    # And rank-6 for M_{1,1,6} (trivial: each output is one input times a coefficient)
    #
    # Wait, M_{1,1,6} is just scaling: y_i = x_i * c_i
    # This has rank 6 (each term is one product)
    #
    # So total rank = 7 * 6 = 42
    #
    # Let's build this using the Kronecker product structure.
    #
    # M_{2,2,2} has inputs A (2x2), B (2x2), outputs C (2x2)
    # M_{1,1,6} has inputs x (1x6), scalar s, outputs y (1x6)
    #
    # The Kronecker product M_{2,2,2} ⊗ M_{1,1,6} gives:
    # - Input A becomes 2x2 (from M_{2,2,2}) ⊗ 1x1 = 2x2, but we need 2x2 for the first factor
    # - Actually, the full tensor is:
    #   A is 2x2 (n=2, m=2 for the first factor)
    #   B is 2x12 (m=2, p=12 for the second factor)
    #   C is 2x12 (p=12, n=2 for the output)
    #
    # Let me think about this differently.
    #
    # The tensor M_{2,2,12} can be viewed as:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # where i in {0,1}, j in {0,1}, k in {0,...,11}
    #
    # Using the Kronecker product M_{2,2,2} ⊗ M_{1,1,6}:
    # Let k = 2*u + v where u in {0,...,5}, v in {0,1}
    # Then B_{j,k} = B_{j, 2u+v}
    #
    # The Kronecker product means:
    # For each block u (0..5), we have a 2x2 subproblem
    # C_{i, 2u+v} = sum_j A_{i,j} * B_{j, 2u+v}
    #
    # This is exactly 6 independent 2x2-by-2x1 multiplications, but we can
    # use the full 2x2-by-2x2 structure by treating each pair of columns as a 2x2 block.
    #
    # So for each u in 0..5:
    #   C_{i, 2u+v} = sum_j A_{i,j} * B_{j, 2u+v} for v in {0,1}
    #
    # This is a 2x2 matrix A times a 2x2 matrix B_u = [B_{0,2u}, B_{0,2u+1}; B_{1,2u}, B_{1,2u+1}]
    # giving C_u = [C_{0,2u}, C_{0,2u+1}; C_{1,2u}, C_{1,2u+1}]
    #
    # So we have 6 independent 2x2x2 matrix multiplications.
    # Each has rank 7 (Strassen), so total rank = 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # n*m = 4
    V = [[0] * rank for _ in range(m * p)]  # m*p = 24
    W = [[0] * rank for _ in range(p * n)]  # p*n = 24
    
    # Strassen's 7 forms for 2x2x2:
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen formulas:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B10)
    # M6 = (A01 - A11) * (B01 + B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # U forms (for A):
    U_forms = [
        [1, 0, 0, 1],   # M0: A00 + A11
        [0, 0, 1, 1],   # M1: A10 + A11
        [1, 0, 0, 0],   # M2: A00
        [0, 0, 0, 1],   # M3: A11
        [1, 1, 0, 0],   # M4: A00 + A01
        [-1, 0, 1, 0],  # M5: A10 - A00
        [0, 1, 0, -1]   # M6: A01 - A11
    ]
    
    # V forms (for B):
    V_forms = [
        [1, 0, 0, 1],   # M0: B00 + B11
        [1, 0, 0, 0],   # M1: B00
        [0, 1, 0, -1],  # M2: B01 - B11
        [0, 0, 1, -1],  # M3: B10 - B00
        [0, 0, 0, 1],   # M4: B11
        [1, 0, 1, 0],   # M5: B00 + B10
        [0, 1, 0, 1]    # M6: B01 + B11
    ]
    
    # W forms (for C):
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    W_forms = [
        [1, 0, 0, 0],   # M0 contributes to C00
        [0, 0, 0, -1],  # M1 contributes to C11
        [0, 1, 0, 1],   # M2 contributes to C01 and C11
        [1, 0, 1, 0],   # M3 contributes to C00 and C10
        [-1, 1, 0, 0],  # M4 contributes to C00 and C01
        [0, 0, 0, 1],   # M5 contributes to C11
        [1, 0, 0, 0]    # M6 contributes to C00
    ]
    
    # For each block u (0..5) and each Strassen term t (0..6):
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U: A indices 0..3
            for i in range(4):
                U[i][c] = U_forms[t][i]
            
            # Fill V: B indices for block u
            # B_{0, 2u} is at index 0*12 + 2u = 2u
            # B_{0, 2u+1} is at index 0*12 + 2u+1 = 2u+1
            # B_{1, 2u} is at index 1*12 + 2u = 12 + 2u
            # B_{1, 2u+1} is at index 1*12 + 2u+1 = 12 + 2u+1
            
            B_indices = [2*u, 2*u+1, 12+2*u, 12+2*u+1]
            for i in range(4):
                V[B_indices[i]][c] += V_forms[t][i]
            
            # Fill W: C indices for block u
            # C_{0, 2u} is at index (2u)*2 + 0 = 4u
            # C_{0, 2u+1} is at index (2u+1)*2 + 0 = 4u+2
            # C_{1, 2u} is at index (2u)*2 + 1 = 4u+1
            # C_{1, 2u+1} is at index (2u+1)*2 + 1 = 4u+3
            
            C_indices = [4*u, 4*u+2, 4*u+1, 4*u+3]
            for i in range(4):
                W[C_indices[i]][c] += W_forms[t][i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
