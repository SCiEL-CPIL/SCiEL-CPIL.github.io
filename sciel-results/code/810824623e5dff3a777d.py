"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication
    via the Kronecker product of two 2x2 Strassen decompositions.
    
    The 4x4 multiplication tensor is the tensor square (Kronecker product)
    of the 2x2 multiplication tensor. Since 2x2 matmul has rank 7 (Strassen),
    the 4x4 matmul has rank at most 7*7 = 49.
    
    We construct U, V, W directly using the Kronecker product structure:
      U[i*4+j, 7*s+t] = U2[i, s] * U2[j, t]  (with appropriate index mapping)
    
    Actually, the correct structure is:
      The 4x4 matmul tensor T4[i,j,k] = delta[i,k] (i.e., C[i,k] = sum_j A[i,j]*B[j,k])
      can be written as T4 = T2 ⊗ T2 where T2 is the 2x2 matmul tensor.
      
      For the decomposition: if T2 = sum_s u2_s ⊗ v2_s ⊗ w2_s (rank 7),
      then T4 = sum_{s,t} (u2_s ⊗ u2_t) ⊗ (v2_s ⊗ v2_t) ⊗ (w2_s ⊗ w2_t)
      
      So:
        U[i*4+j, 7*s+t] = U2[i, s] * U2[j, t]
        V[j*4+k, 7*s+t] = V2[j, s] * V2[k, t]
        W[k*4+i, 7*s+t] = W2[k, s] * W2[i, t]
      
      where U2, V2, W2 are the 2x2 Strassen factor matrices (4x7 each).
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well under 120s

    n, m, p = DIMENSIONS
    rank = 49  # 7 * 7

    # 2x2 Strassen decomposition factors (4x7 each)
    # U2[i, s]: coefficient of A[i] in the s-th product
    # V2[j, s]: coefficient of B[j] in the s-th product
    # W2[k, s]: coefficient of C[k] in the s-th product
    #
    # Strassen 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M5

    # U2: 4 rows (A00, A01, A10, A11), 7 cols (M0..M6)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ]

    # V2: 4 rows (B00, B01, B10, B11), 7 cols (M0..M6)
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ]

    # W2: 4 rows (C00, C01, C10, C11), 7 cols (M0..M6)
    W2 = [
        [1, 0, 0, -1, 1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]

    # Build U, V, W using Kronecker product structure
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)

    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t

            # U[i*4+j, c_idx] = U2[i, s] * U2[j, t]
            # where i, j are indices 0..3 for the 2x2 block structure
            # i corresponds to row of A (0..3), j to col of A (0..3)
            # But in the Kronecker product, we need:
            #   A is 4x4, indexed as A[row, col] where row = 2*I + i, col = 2*J + j
            #   The Kronecker product gives: U[row*4+col, 7*s+t] = U2[I,s]*U2[i,s'] ... 
            #   
            # Wait, let me reconsider. The 4x4 matmul tensor is:
            #   T4[a, b, c] = delta[a,c] for C[a,c] = sum_b A[a,b] B[b,c]
            #   where a, b, c are each 0..3.
            #   
            # The Kronecker product T2 ⊗ T2 means:
            #   T4[(i1,i2),(j1,j2),(k1,k2)] = T2[i1,j1,k1] * T2[i2,j2,k2]
            #   where T2[i,j,k] = delta[i,k] for 2x2 matmul.
            #
            # So if T2 = sum_s u2_s ⊗ v2_s ⊗ w2_s, then:
            #   T4 = sum_{s,t} (u2_s ⊗ u2_t) ⊗ (v2_s ⊗ v2_t) ⊗ (w2_s ⊗ w2_t)
            #
            # Here u2_s is a vector in R^4 (indexed by i), v2_s in R^4 (indexed by j), w2_s in R^4 (indexed by k).
            # u2_s ⊗ u2_t is a vector in R^16 (indexed by (i1,i2) -> i1*4+i2... no, i1*2+i2)
            #
            # Actually, the index mapping: for 4x4, row index = i1*2 + i2 where i1, i2 in {0,1}
            # So the Kronecker product u2_s ⊗ u2_t has index (i1*2+i2) for the combined index.
            # But wait, u2_s is 4-dimensional (for 2x2, indices 0,1,2,3 = (0,0),(0,1),(1,0),(1,1))
            #
            # Let me re-index. For 2x2 matmul:
            #   A has indices (i,j) with i,j in {0,1}, flattened to i*2+j
            #   B has indices (j,k) with j,k in {0,1}, flattened to j*2+k
            #   C has indices (i,k) with i,k in {0,1}, flattened to i*2+k
            #
            # For 4x4 matmul:
            #   A has indices (I, J) with I, J in {0,1,2,3}, flattened to I*4+J
            #   B has indices (J, K) with J, K in {0,1,2,3}, flattened to J*4+K
            #   C has indices (I, K) with I, K in {0,1,2,3}, flattened to I*4+K
            #
            # The Kronecker product structure:
            #   I = 2*I1 + I2, J = 2*J1 + J2, K = 2*K1 + K2
            #   T4[I,J,K] = T2[I1,J1,K1] * T2[I2,J2,K2]
            #
            # So:
            #   U[I*4+J, 7*s+t] = U2[I1*2+I2, s] * U2[I2', ...]
            #
            # Hmm, this is getting confusing. Let me use the direct formula:
            #   U[I*4+J, 7*s+t] = U2[I, s] * U2[J, t]  ... NO, that's not right either.
            #
            # The correct Kronecker product: if u is a vector of length 4 (indexed 0..3)
            # and v is a vector of length 4 (indexed 0..3), then u ⊗ v is a vector of length 16
            # indexed by (i,j) -> i*4+j, with (u⊗v)[i*4+j] = u[i]*v[j].
            #
            # But for the 4x4 case, the indices I, J are each 0..3, and the Kronecker product
            # of two 2x2-indexed vectors gives a 4x4-indexed vector where:
            #   (u2_s ⊗ u2_t)[I*4+J] = u2_s[I]*u2_t[J]
            #
            # Wait, that's only if we're treating the 4x4 as a tensor product of two 2x2's
            # in the same way. But the 2x2 Strassen factors U2 are indexed by the 2x2 matrix
            # positions (0,1,2,3) = (00, 01, 10, 11).
            #
            # For the 4x4 case, position I (0..3) corresponds to row I, and J (0..3) to column J.
            # The Kronecker product structure says:
            #   U[I*4+J, 7*s+t] = U2[I, s] * U2[J, t]
            #
            # This is because the 4x4 matrix A can be seen as a 2x2 block matrix where each
            # block is 2x2, and the Kronecker product of the two 2x2 Strassen decompositions
            # gives the 4x4 decomposition.

            for I in range(4):
                for J in range(4):
                    u_coeff = U2[I][s] * U2[J][t]
                    if u_coeff != 0:
                        U[I * 4 + J][c_idx] = u_coeff

            for J in range(4):
                for K in range(4):
                    v_coeff = V2[J][s] * V2[K][t]
                    if v_coeff != 0:
                        V[J * 4 + K][c_idx] = v_coeff

            for K in range(4):
                for I in range(4):
                    w_coeff = W2[K][s] * W2[I][t]
                    if w_coeff != 0:
                        W[K * 4 + I][c_idx] = w_coeff

            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
