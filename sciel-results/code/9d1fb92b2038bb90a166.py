"""Improved program for matrix multiplication 2x4 by 4x5 tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # We construct a rank-39 decomposition using Strassen on a 2x2x2 sub-block
    # and schoolbook for the remaining terms.
    #
    # The matrix multiplication tensor M_{2,4,5} has entries indexed by:
    #   U: (i, j) where i in [0,2), j in [0,4)  -> 8 rows
    #   V: (j, k) where j in [0,4), k in [0,5)  -> 20 rows
    #   W: (k, i) where k in [0,5), i in [0,2)  -> 10 rows
    #
    # The tensor is: sum over i,j,k of e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    #
    # Strategy: Apply Strassen to the sub-tensor involving k in {0,1}, j in {0,1}.
    # This sub-tensor is M_{2,2,2} which has rank 7 (Strassen).
    # The remaining terms are handled by schoolbook.
    #
    # Terms breakdown:
    # 1. Strassen block: k in {0,1}, j in {0,1} -> 7 terms (instead of 8 schoolbook)
    # 2. Schoolbook: k in {0,1}, j in {2,3,4} -> 2*2*3 = 12 terms
    # 3. Schoolbook: k in {2,3,4}, j in {0,1,2,3,4} -> 2*3*5 = 30 terms
    # Total: 7 + 12 + 30 = 49... wait, that's more than 40.
    #
    # Let me recount. The full schoolbook is n*m*p = 2*4*5 = 40 terms.
    # Strassen replaces 8 terms (k in {0,1}, j in {0,1}, i in {0,1}) with 7 terms.
    # So total = 40 - 8 + 7 = 39.
    #
    # Let me enumerate all 40 schoolbook terms and replace the appropriate 8 with 7 Strassen terms.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 8 x 39
    V = [[0] * rank for _ in range(m * p)]  # 20 x 39
    W = [[0] * rank for _ in range(p * n)]  # 10 x 39
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # If this is in the Strassen block (k in {0,1}, j in {0,1}), skip
                if k in (0, 1) and j in (0, 1):
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now add the 7 Strassen terms for the 2x2x2 sub-block (k in {0,1}, j in {0,1})
    # Strassen's algorithm for 2x2 matrix multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B11 + B10)
    #
    # In our case, the sub-block involves:
    # A indices: (i, j) where i in {0,1}, j in {0,1}
    # B indices: (j, k) where j in {0,1}, k in {0,1}
    # C indices: (k, i) where k in {0,1}, i in {0,1}
    #
    # The 8 schoolbook terms for this sub-block are:
    # (i=0,j=0,k=0), (i=0,j=0,k=1), (i=0,j=1,k=0), (i=0,j=1,k=1)
    # (i=1,j=0,k=0), (i=1,j=0,k=1), (i=1,j=1,k=0), (i=1,j=1,k=1)
    #
    # Strassen terms:
    # M1 = (A_{0,0} + A_{1,1}) * (B_{0,0} + B_{1,1})
    #   U: A_{0,0} and A_{1,1} get +1
    #   V: B_{0,0} and B_{1,1} get +1
    #   W: C_{0,0} gets +1, C_{0,1} gets -1, C_{1,0} gets -1, C_{1,1} gets +1
    #
    # Wait, I need to be more careful. The Strassen algorithm computes:
    # C_{0,0} = M1 - M4 + M6 - M7
    # C_{0,1} = M3 + M5
    # C_{1,0} = M2 + M4
    # C_{1,1} = M1 - M3 + M5 - M6
    #
    # So the W matrix entries for each Strassen term are:
    # M1: C_{0,0}=+1, C_{0,1}=0, C_{1,0}=0, C_{1,1}=+1  -> wait no
    #
    # Let me redo this. In standard Strassen:
    # C_{0,0} = M1 - M4 + M6 - M7
    # C_{0,1} = M3 + M5
    # C_{1,0} = M2 + M4
    # C_{1,1} = M1 - M3 + M5 - M6
    #
    # So:
    # M1 contributes to C_{0,0} with +1 and C_{1,1} with +1
    # M2 contributes to C_{1,0} with +1
    # M3 contributes to C_{0,1} with +1 and C_{1,1} with -1
    # M4 contributes to C_{0,0} with -1 and C_{1,0} with +1
    # M5 contributes to C_{0,1} with +1 and C_{1,1} with +1
    # M6 contributes to C_{0,0} with +1 and C_{1,1} with -1
    # M7 contributes to C_{0,0} with -1
    
    # U indices for A: (i,j) -> i*m + j, so:
    # A_{0,0} = index 0, A_{0,1} = index 1, A_{1,0} = index 4, A_{1,1} = index 5
    # V indices for B: (j,k) -> j*p + k, so:
    # B_{0,0} = index 0, B_{0,1} = index 1, B_{1,0} = index 5, B_{1,1} = index 6
    # W indices for C: (k,i) -> k*n + i, so:
    # C_{0,0} = index 0, C_{0,1} = index 1, C_{1,0} = index 2, C_{1,1} = index 3
    
    # M1 = (A_{0,0} + A_{1,1}) * (B_{0,0} + B_{1,1})
    U[0][col] = 1; U[5][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A_{1,0} + A_{1,1}) * B_{0,0}
    U[4][col] = 1; U[5][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # M3 = A_{0,0} * (B_{0,1} - B_{1,1})
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M4 = A_{1,1} * (B_{1,0} - B_{0,0})
    U[5][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # M5 = (A_{0,0} + A_{0,1}) * B_{1,1}
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M6 = (A_{1,0} - A_{0,0}) * (B_{0,0} + B_{1,0})
    U[4][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = -1
    col += 1
    
    # M7 = (A_{0,1} - A_{1,1}) * (B_{1,1} + B_{1,0})
    U[1][col] = 1; U[5][col] = -1
    V[6][col] = 1; V[5][col] = 1
    W[0][col] = -1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
