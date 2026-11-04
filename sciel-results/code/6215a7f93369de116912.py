"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 150s limit
    
    # We'll use a structured Kronecker product approach.
    # Decompose 4x4x6 as a combination of 2x2x2 (Strassen, rank 7) 
    # and handle the rectangular 6 dimension.
    #
    # Approach: Use the fact that matmul(4,4,6) can be seen as
    # 4 independent 2x2x3 multiplications combined, or use
    # a block structure.
    #
    # Let's use a direct structured decomposition of rank 42:
    # Partition i into 2 groups of 2, j into 2 groups of 2.
    # For each (i_group, j_group) pair, we have a 2x2x6 subproblem.
    # We can solve each 2x2x6 with rank 6 (schoolbook for the 2x2 part 
    # would be 4, times 3 for the 6/2... let me think).
    #
    # Actually, let's use a simpler verified approach:
    # Rank 49 decomposition via recursive Strassen on 4x4x4 part,
    # extended to 6 columns.
    #
    # For 4x4x4, Strassen recursively gives rank 7^2 = 49.
    # For 4x4x6, we can use rank 49 * (6/4) ... no.
    #
    # Let me just build a valid decomposition directly.
    # The tensor T[i,j,k] = 1 for all valid (i,j,k).
    # We need U[16,r], V[24,r], W[24,r] such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Use a Kronecker product structure:
    # 4 = 2*2, 4 = 2*2, 6 = 2*3
    # T_4x4x6 = T_2x2x2 ⊗ T_2x2x3 (approximately, in block sense)
    # 
    # For 2x2x2, Strassen rank 7:
    # For 2x2x3, schoolbook rank 12.
    # Kronecker product rank = 7 * 12 = 84. Still better than 96.
    
    # Let me construct the 2x2x2 Strassen decomposition:
    # Indices: i,j in {0,1}, k in {0,1}
    # U_str: (2*2, 7) = (4,7), V_str: (2*2, 7) = (4,7), W_str: (2*2, 7) = (4,7)
    
    U_str = np.zeros((4, 7))
    V_str = np.zeros((4, 7))
    W_str = np.zeros((4, 7))
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # c=0: M1 = (A00+A11)(B00+B11)
    # U: i=0,j=0 -> 1; i=1,j=1 -> 1
    # V: j=0,k=0 -> 1; j=1,k=1 -> 1
    # W: k=0,i=0 -> 1; k=1,i=1 -> 1
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[3, 0] = 1
    # But W should give: W[0*2+0]*W... wait, W[k*n+i] for 2x2x2 is W[k*2+i]
    # M1 contributes to C00 and C11 with coefficients.
    # This is getting complex. Let me use a known correct Strassen decomposition.
    
    # Standard Strassen 2x2x2 decomposition (rank 7):
    # Each term c has U, V, W vectors.
    # The decomposition satisfies: for each (i,j,k), sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = delta(i,j,k valid)
    # Actually the tensor is T[i,j,k] = 1 for all i,j,k (it's the matmul tensor).
    
    # Let me just use schoolbook for 2x2x2 (rank 4) to be safe, then optimize.
    # 2x2x2 schoolbook: rank 4
    U_sb = np.zeros((4, 4))
    V_sb = np.zeros((4, 4))
    W_sb = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c = i*4 + j*2 + k
                U_sb[i*2+j, c] = 1
                V_sb[j*2+k, c] = 1
                W_sb[k*2+i, c] = 1
    
    # 2x2x3 schoolbook: rank 12
    U_223 = np.zeros((4, 12))
    V_223 = np.zeros((6, 12))
    W_223 = np.zeros((6, 12))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                c = i*6 + j*3 + k
                U_223[i*2+j, c] = 1
                V_223[j*3+k, c] = 1
                W_223[k*2+i, c] = 1
    
    # Now the Kronecker product for 4x4x6:
    # 4 = 2*2 in i, 4 = 2*2 in j, 6 = 2*3 in k
    # The tensor T_4x4x6[i,j,k] where i=i1*2+i0, j=j1*2+j0, k=k1*3+k0
    # This is NOT a simple Kronecker product because 6 = 2*3 mixes the dimensions.
    
    # Let me use a different factorization: 4x4x6 = (2x2x2) x (2x2x3) in a block sense.
    # i = i_high*2 + i_low, j = j_high*2 + j_low, k = k_high*3 + k_low
    # where i_high,j_high in {0,1}, i_low,j_low in {0,1}, k_high in {0,1}, k_low in {0,1,2}
    
    # The matmul tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_{j_high,j_low} A[i_high,i_low, j_high,j_low] * B[j_high,j_low, k_high,k_low]
    
    # This can be decomposed as: for each (i_high, j_high, k_high), we have a 2x2x3 subproblem
    # with A_block[i_low, j_low] and B_block[j_low, k_low].
    
    # There are 2*2*2 = 8 such subproblems, each of type 2x2x3 with rank 12.
    # Total rank = 8 * 12 = 96. Same as schoolbook!
    
    # That's not helpful. Let me think differently.
    
    # Alternative: use Strassen on the 2x2 blocks in i,j dimensions.
    # A is 4x4, B is 4x6.
    # Partition A into 2x2 blocks: A = [[A00, A01], [A10, A11]] each 2x2
    # Partition B into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]] each 2x3
    # C = A*B is 4x6: C = [[C00, C01, C02], [C10, C11, C12]] each 2x3
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Using Strassen on the 2x2 block structure (treating each 2x3 as a "vector" of 3):
    # M1 = (A00+A11) * (B00+B10)  -- but this is 2x2 * 2x3 = 2x3, rank 4 per term
    # This is getting complicated.
    
    # Let me just implement a valid decomposition with rank < 96 using a different method.
    # I'll use a random optimization approach to find a low-rank decomposition.
    
    # Actually, let me just return a valid schoolbook decomposition for now,
    # but with a slightly optimized structure.
    
    # For now, let's use rank 72: decompose 4x4x6 as 3 copies of 4x4x2.
    # 4x4x2 can be done with rank 28 (using Strassen-like methods).
    # 3 * 28 = 84. Still not great.
    
    # Let me just use the schoolbook (rank 96) as a baseline and try to optimize.
    # But first, let me make sure I return something valid.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i*m*p + j*p + k
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

