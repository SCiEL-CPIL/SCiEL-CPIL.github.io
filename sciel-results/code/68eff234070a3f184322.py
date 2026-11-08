import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Construct a rank-16 decomposition for 2x3 * 3x3 matrix multiplication.
    
    Uses:
    - Strassen (rank 7) for the 2x2 * 2x2 sub-block
    - 9 additional terms for the remaining 3 terms involving the 3rd row/column
    
    Total: 7 + 9 = 16
    """
    n, m, p = 2, 3, 3
    R = 16
    
    U = np.zeros((n * m, R))  # 6 x 16
    V = np.zeros((m * p, R))  # 9 x 16
    W = np.zeros((p * n, R))  # 6 x 16
    
    # Index mapping:
    # A[i,j] where i in {0,1}, j in {0,1,2} -> U row index = i*3 + j
    #   A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B[j,k] where j in {0,1,2}, k in {0,1,2} -> V row index = j*3 + k
    #   B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
    # C[k,i] where k in {0,1,2}, i in {0,1} -> W row index = k*2 + i (trace convention)
    #   C00=0, C10=1, C01=2, C11=3, C02=4, C12=5
    
    # Part 1: Strassen decomposition for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # This handles contributions to C00, C01, C10, C11 from A00,A01,A10,A11 and B00,B01,B10,B11
    
    # M1: (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M2: (A10+A11)*B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C10, -C11
    
    # M3: A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # M4: A11*(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C00, C10
    
    # M5: (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # -C00, C01
    
    # M6: (A10-A00)*(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11
    
    # M7: (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1  # C00
    
    # Part 2: Additional terms for the 3rd row/column
    # We need to account for:
    # - C00, C01, C10, C11 contributions involving A02,A12 with B20,B21
    # - C02, C12 contributions involving A00,A01,A02,A10,A11,A12 with B02,B12,B22
    
    # M8: A02*B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # M9: A02*B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # M10: A12*B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # M11: A12*B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # M12: A00*B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # M13: A01*B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # M14: A02*B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # M15: A10*B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # M16: A11*B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # We're missing A12*B22 -> C12! Let me recount...
    # Actually, I need to check if we can merge any terms.
    
    # Let me verify the full decomposition by checking all 18 output entries:
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C11 = A10*B01 + A11*B11 + A12*B21
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    
    # From Strassen (M1-M7), we get the 2x2*2x2 part correctly.
    # From M8-M16, we get the remaining terms.
    # But we're missing A12*B22 for C12.
    
    # This gives us 16 terms but we're missing one term. Let me reconsider.
    
    # Actually, let me verify that Strassen handles the 2x2*2x2 correctly and
    # then count what's left.
    
    # The 2x2*2x2 product gives us 4 terms:
    # A00*B00, A01*B10, A10*B00, A11*B10 (for C00, C10)
    # A00*B01, A01*B11, A10*B01, A11*B11 (for C01, C11)
    
    # The remaining terms are:
    # For C00: A02*B20
    # For C10: A12*B20
    # For C01: A02*B21
    # For C11: A12*B21
    # For C02: A00*B02, A01*B12, A02*B22
    # For C12: A10*B02, A11*B12, A12*B22
    
    # That's 4 + 3 + 3 = 10 additional terms, total 7 + 10 = 17.
    
    # To get to 16, we need to find a way to combine one of these terms.
    
    # Let me try a different approach: use a rank-16 decomposition directly.
    
    # Actually, let me just return the rank-17 solution which is known to work,
    # and see if we can improve it later.
    
    # For now, let me fix the missing term:
    # We need A12*B22 -> C12, which would be term 16.
    # But we already used indices 7-15 for M8-M16.
    
    # Let me restructure: use 7 Strassen + 9 other terms = 16 total.
    
    # I need to be more careful. Let me just implement the known rank-17 solution
    # and verify it works.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Let me rewrite this more carefully with a verified rank-17 solution


def construct():
    """
    Rank-17 exact decomposition for 2x3 * 3x3 matrix multiplication.
    """
    n, m, p = 2, 3, 3
    R = 17
    
    U = np.zeros((n * m, R))  # 6 x 17
    V = np.zeros((m * p, R))  # 9 x 17
    W = np.zeros((p * n, R))  # 6 x 17
    
    # Index mapping:
    # A[i,j] -> U row = i*3 + j: A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B[j,k] -> V row = j*3 + k: B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
    # C[k,i] -> W row = k*2 + i: C00=0, C10=1, C01=2, C11=3, C02=4, C12=5
    
    # Strassen (7 terms) for the 2x2 * 2x2 sub-block
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Schoolbook terms (10 terms) for the remaining products
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
