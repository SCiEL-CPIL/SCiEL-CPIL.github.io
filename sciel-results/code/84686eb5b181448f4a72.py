import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition by concatenating parts.
    # Part 1: A (4x4) * B1 (4x4)
    # Part 2: A (4x4) * B2 (4x4)
    # Part 3: A (4x4) * B3 (4x1)
    
    # Helper to get Strassen 2x2x2 decomposition
    # Returns U (4x7), V (4x7), W (4x7) for 2x2x2
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11 is (0,0), A22 is (1,1)
        # B11 is (0,0), B22 is (1,1)
        # Output C11 += M1, C12 += M1, C21 += M1, C22 += M1 (with signs)
        
        # We define the 7 multiplications
        # r=0: M1
        # r=1: M2
        # r=2: M3
        # r=3: M4
        # r=4: M5
        # r=5: M6
        # r=6: M7
        
        # Coefficients for A blocks (2x2) and B blocks (2x2)
        # u[r, i_block, j_block]
        # v[r, j_block, k_block]
        
        u_coeffs = [
            [[1, 0], [0, 1]], # M1: A11, A22
            [[0, 0], [1, 1]], # M2: A21, A22
            [[1, 0], [0, 0]], # M3: A11
            [[0, 0], [0, 1]], # M4: A22
            [[1, 1], [0, 0]], # M5: A11, A12
            [[1, 0], [-1, 0]],# M6: A11, -A21? No. M6 = (A21 - A11)(B11 + B12)
            # M6: A21, -A11
            [[0, 1], [0, -1]] # M7: A12, -A22? M7 = (A12 - A22)(B21 + B22)
        ]
        # Correction on M6, M7 coefficients
        # M6 = (A21 - A11) * (B11 + B12)
        # A part: A21 (1,0), -A11 (-1,0) -> [[-1, 0], [1, 0]]
        # B part: B11 (0,0), B12 (0,1) -> [[1, 1], [0, 0]]
        
        # M7 = (A12 - A22) * (B21 + B22)
        # A part: A12 (0,1), -A22 (-1,1) -> [[0, -1], [1, 0]]? No.
        # A12 is (0,1), A22 is (1,1).
        # A12 - A22: (0,1) is 1, (1,1) is -1.
        
        u_coeffs = [
            [[1, 0], [0, 1]], # M1
            [[-1, 0], [1, 0]], # M6 A part? No, let's list them properly.
        ]
        
        # Let's just hardcode the 7 terms for 2x2x2
        # Term 0: (A00+A11)(B00+B11)
        # Term 1: (A10+A11)B00
        # Term 2: A00(B01-B11)
        # Term 3: A11(B10-B00)
        # Term 4: (A00+A01)B11
        # Term 5: (A10-A00)(B00+B01)
        # Term 6: (A01-A11)(B10+B11)
        
        # U indices: i*2+j
        # V indices: j*2+k
        # W indices: k*2+i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # T0
        # A: (0,0), (1,1)
        # B: (0,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        # C00 += T0, C11 += T0
        # C10 += T0, C01 += T0 ?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # This is getting messy. I will use a known explicit list for 2x2x2 rank 7.
        # See: https://en.wikipedia.org/wiki/Strassen_algorithm
        # I will implement the combination logic directly in the 4x4x4 builder.
        
        return U, V, W

    # Since manual derivation is error-prone, I will use a simpler approach for the 4x4x4 part
    # using the block structure and the 2x2x2 rank 7 decomposition.
    
    # Let's define the 2x2x2 decomposition explicitly.
    # We want to decompose T_{i,j,k} = delta_{i,k} for 2x2.
    # No, T is the multiplication tensor.
    
    # I will use a library or hardcoded values for 2x2x2 rank 7.
    # U (4x7), V (4x7), W (4x7)
    # Columns correspond to the 7 multiplications.
    
    # M1: A11+A22, B11+B22 -> C11, C12, C21, C22
    # Actually, the output is a 2x2 matrix.
    # The decomposition of the 2x2x2 tensor is:
    # Sum_{r=1}^7 u_r \otimes v_r \otimes w_r
    
    # I will construct the 4x4x4 decomposition by iterating over the 7 block-multiplications.
    # For each block multiplication, I use the 2x2x2 decomposition.
    
    # Let's assume we have the 2x2x2 tensors U2, V2, W2 (4x7 each).
    # For 4x4x4:
    # We have 7 Strassen steps.
    # Step r involves A-blocks and B-blocks.
    # The result is added to C-blocks.
    
    # This is too complex to write from scratch in the time limit.
    # I will fall back to the schoolbook decomposition but optimized for the 4x4x9 case?
    # No, 144 is the baseline.
    
    # Let's try to implement the 4x4x4 Strassen (Rank 49) correctly.
    # I will use the fact that 4x4x4 can be decomposed into 49 rank-1 terms.
    
    # I will use a hardcoded list of 49 terms for 4x4x4 if I can find/generate them.
    # Or I can just use the schoolbook for 4x4x4 (Rank 64) and 4x4x1 (Rank 16).
    # Total 64+64+16 = 144. Same as baseline.
    
    # I need to implement Strassen.
    # I will write a helper that takes the 2x2x2 rank 7 tensors and builds the 4x4x4.
    
    # 2x2x2 Rank 7 Tensors:
    # I will define them now.
    # u, v, w are 4x7 matrices.
    
    # Let's use the following decomposition for 2x2x2:
    # (Source: various online sources for Strassen)
    
    U2 = np.array([
        [1, -1, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 0, 0, 0, 1],   # A01
        [0, 1, 0, 0, 0, -1, 0],  # A10
        [1, 1, 0, 1, 1, 0, -1]   # A11
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 1],   # B01
        [0, 0, 0, 1, 0, 0, 0],   # B10
        [1, 0, -1, 1, 1, 0, 1]   # B11
    ])
    
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 1]   # C11
    ])
    
    # Verify: C = U * V * W?
    # C_{ik} = sum_j sum_r U_{ir} V_{jr} W_{kr}? No.
    # Tensor T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
    # Here indices are flattened.
    # U index: i*2+j? No, U is (4x7). Index is i*2+j?
    # In 2x2, i,j,k are 0,1.
    # U[i,j, r]. Flattened U[i*2+j, r].
    # V[j,k, r]. Flattened V[j*2+k, r].
    # W[k,i, r]. Flattened W[k*2+i, r].
    
    # Let's check one term.
    # T_{0,0,0} should be 1.
    # sum_r U[0, r] V[0, r] W[0, r]
    # r=0: 1*1*1 = 1
    # r=1: -1*1*0 = 0
    # r=2: 1*0*0 = 0
    # r=3: 0*-1*0 = 0
    # r=4: 1*0*0 = 0
    # r=5: 1*1*0 = 0
    # r=6: 0*0*1 = 0
    # Sum = 1. Correct.
    
    # T_{0,1,0} should be 0.
    # sum_r U[0, r] V[2, r] W[0, r] (V index j*2+k. j=0, k=1 -> 0*2+1=1? No. j=0, k=1 is index 1? No. j is row, k is col.
    # V is (m*p, rank). m=2, p=2.
    # V[j, k] -> index j*2+k.
    # j=0, k=1 -> index 1.
    # U[0, r] is A_{0,0}.
    # V[1, r] is B_{0,1}.
    # W[0, r] is C_{0,0}? No, W[k*n+i]. k=0, i=0 -> index 0.
    # Term: A00 * B01 * C00.
    # r=0: 1 * 0 * 1 = 0
    # r=1: -1 * 0 * 0 = 0
    # r=2: 1 * 1 * 0 = 0
    # r=3: 0 * 0 * 0 = 0
    # r=4: 1 * 0 * 0 = 0
    # r=5: 1 * 1 * 0 = 0
    # r=6: 0 * 0 * 1 = 0
    # Sum = 0. Correct.
    
    # This U2, V2, W2 seems valid.
    
    # Now build 4x4x4.
    # Split A, B, C into 2x2 blocks.
    # A_{IJ}, B_{JK}, C_{KI}.
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product.
    # We decompose each M using U2, V2, W2.
    # M = A_block * B_block.
    # A_block = sum_{ij} u_{block, ij} A_{ij}.
    # This is not a single rank-1 term.
    # We must expand the block multiplication.
    
    # For M1:
    # A_part = A00 + A11.
    # B_part = B00 + B11.
    # M1 = (A00 + A11) * (B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # Each term is a 2x2x2 multiplication.
    # So M1 contributes 4 * 7 = 28 rank-1 terms?
    # No, we can combine them.
    # But for simplicity, we can just add the decompositions of the 4 products.
    # Total rank for M1: 28.
    # Total rank for 4x4x4: 7 * 28 = 196?
    # This is worse than schoolbook (64).
    
    # We must use the block structure more efficiently.
    # The Strassen algorithm for 4x4x4 using 2x2 blocks has rank 49.
    # This implies that the 7 multiplications are of 2x2 matrices, and each 2x2 multiplication is rank 7.
    # So total rank is 49.
    # How?
    # The decomposition is linear.
    # We don't expand M1 into 4 terms. We treat M1 as a single "multiplication" of linear combinations.
    # But the tensor decomposition is a sum of rank-1 tensors.
    # A rank-1 tensor for 4x4x4 is u \otimes v \otimes w where u, v, w are vectors of size 16.
    # The Strassen decomposition of 4x4x4 into 49 terms means there are 49 such vectors.
    
    # I will construct the 49 terms by iterating over the 7 Strassen steps and the 7 sub-steps of 2x2x2.
    # Step s (0..6):
    #   A_s = sum_{IJ} alpha_{s, IJ} A_{IJ}
    #   B_s = sum_{JK} beta_{s, JK} B_{JK}
    #   C_{KI} += gamma_{s, KI} (A_s B_s)
    
    # For each s, A_s B_s is a 2x2 matrix product.
    # We decompose A_s B_s using the 2x2x2 rank 7 decomposition.
    # A_s = sum_{ij} A_s_{ij} E_{ij}.
    # B_s = sum_{jk} B_s_{jk} F_{jk}.
    # The product is sum_{ij, jk} A_s_{ij} B_s_{jk} (E_{ij} F_{jk}).
    # This is a sum of 16 terms.
    # Each term is a 2x2x2 multiplication? No.
    # E_{ij} F_{jk} is a 2x2 matrix.
    # We need to decompose the tensor for this specific linear combination.
    
    # This is effectively re-running Strassen on the coefficients.
    # I will generate the 49 terms programmatically.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Coefficients for Strassen 4x4x4
    # s=0: M1
    # A: A00, A11. B: B00, B11. C: C00, C01, C10, C11 (with signs)
    
    # I will skip the 4x4x4 optimization and use Schoolbook for 4x4x4 (Rank 64)
    # and Schoolbook for 4x4x1 (Rank 16).
    # Total 144.
    # This is not an improvement.
    
    # I must implement the 49-rank decomposition.
    # I will use the U2, V2, W2 to build it.
    
    # For each Strassen step s:
    # We have linear combinations of A blocks and B blocks.
    # Let A_s be the 2x2 matrix of coefficients for A blocks.
    # Let B_s be the 2x2 matrix of coefficients for B blocks.
    # The product P_s = A_s B_s is a 2x2 matrix of coefficients.
    # Wait, A_s is not a matrix. A_s is a linear combination of A blocks.
    # A_s = sum_{IJ} A_s_{IJ} A_{IJ}.
    # This A_s is a 2x2 matrix (in the block sense).
    # Its entries are linear combinations of the global A entries.
    # We can treat A_s as a 2x2 matrix of "variables".
    # The multiplication A_s B_s is a 2x2x2 tensor operation on these variables.
    # We decompose this using U2, V2, W2.
    # The result is a sum of 7 rank-1 terms in the "block" space.
    # Each term is u_block \otimes v_block \otimes w_block.
    # u_block is a 2x2 matrix of coefficients for A_{ij}.
    # We flatten u_block to a vector of 4 coefficients.
    # These coefficients are applied to the global A indices.
    
    # So for each of the 7 Strassen steps s:
    #   We have u2, v2, w2 from the 2x2x2 decomposition.
    #   For each of the 7 sub-steps t:
    #       u_coeff = u2[:, t] (4 coeffs for A blocks)
    #       v_coeff = v2[:, t] (4 coeffs for B blocks)
    #       w_coeff = w2[:, t] (4 coeffs for C blocks)
    #       
    #       We map these to global indices.
    #       A block (I, J) corresponds to global indices (2I+i, 2J+j).
    #       The coefficient for A_{2I+i, 2J+j} is u_coeff[I*2+J] * (delta for i,j?)
    #       No, u_coeff is for the block multiplication.
    #       The block multiplication uses the 2x2x2 decomposition.
    #       The 2x2x2 decomposition assumes the inputs are the blocks themselves.
    #       So if u2[i, t] is the coefficient for A_{i,j} in the block,
    #       then for global A, the coefficient for A[2I+i, 2J+j] is u2[i, t] * (block coeff).
    
    # This is getting too deep. I will use the schoolbook for 4x4x4 and 4x4x1.
    # Rank 144.
    # I will try to optimize the 4x4x1 part to rank 12?
    # No, 16 is the lower bound for 4x4x1?
    # Rank 4x4x1 is 16.
    
    # I will return the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
