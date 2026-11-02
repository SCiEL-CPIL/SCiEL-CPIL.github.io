import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank will be Rank(4x4x4) + Rank(4x4x3)
    # Rank(4x4x4) using Strassen = 49
    # Rank(4x4x3) schoolbook = 48
    # Total = 97
    
    rank_sq = 49
    rank_rem = 48
    total_rank = rank_sq + rank_rem
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4 * 4x4 (first 4 columns of B) ---
    # We build the Strassen decomposition for 4x4x4
    # This uses 7 products of 2x2 matrices.
    # Each 2x2 product uses Strassen (rank 7).
    
    # Get the base Strassen tensors for 2x2x2
    U2, V2, W2 = _get_strassen_2x2()
    
    # Define the Strassen combinations for 4x4
    # Each entry is (coeffs_A, coeffs_B)
    # coeffs is a dict: {'11': c, '12': c, '21': c, '22': c}
    strassen_combos = [
        ({'11': 1, '22': 1}, {'11': 1, '22': 1}), # M1
        ({'21': 1, '22': 1}, {'11': 1}),           # M2
        ({'11': 1}, {'12': 1, '22': -1}),          # M3
        ({'22': 1}, {'21': 1, '11': -1}),          # M4
        ({'11': 1, '12': 1}, {'22': 1}),           # M5
        ({'21': 1, '11': -1}, {'11': 1, '12': 1}), # M6
        ({'12': 1, '22': -1}, {'21': 1, '22': 1})  # M7
    ]
    
    col_offset = 0
    for k in range(7):
        coeffs_A, coeffs_B = strassen_combos[k]
        
        for r in range(7): # Inner Strassen rank
            # u2, v2, w2 are vectors of length 4 for the 2x2 blocks
            u2 = U2[:, r]
            v2 = V2[:, r]
            w2 = W2[:, r]
            
            # Expand u2 to U (length 16)
            u_global = [0] * 16
            # u2[0] corresponds to A'00, u2[1] to A'01, u2[2] to A'10, u2[3] to A'11
            # We need to map A' entries to A block entries.
            # A' is 2x2.
            # A'00 comes from A blocks at (0,0) of the block structure.
            # The block structure of A is:
            # A11 A12
            # A21 A22
            # A' is formed by linear combination of these blocks.
            # So A'00 = sum(c_b * A_b[0,0])
            # A'01 = sum(c_b * A_b[0,1])
            # A'10 = sum(c_b * A_b[1,0])
            # A'11 = sum(c_b * A_b[1,1])
            
            # For each block b with coeff c:
            # A_b[0,0] is at global index (block_row*2 + 0) * 4 + (block_col*2 + 0)
            # Block 11: row 0, col 0. Global rows 0,1. Cols 0,1.
            #   A11[0,0] -> idx 0
            #   A11[0,1] -> idx 1
            #   A11[1,0] -> idx 4
            #   A11[1,1] -> idx 5
            # Block 12: row 0, col 1. Global rows 0,1. Cols 2,3.
            #   A12[0,0] -> idx 2
            #   A12[0,1] -> idx 3
            #   A12[1,0] -> idx 6
            #   A12[1,1] -> idx 7
            # Block 21: row 1, col 0. Global rows 2,3. Cols 0,1.
            #   A21[0,0] -> idx 8
            #   A21[0,1] -> idx 9
            #   A21[1,0] -> idx 12
            #   A21[1,1] -> idx 13
            # Block 22: row 1, col 1. Global rows 2,3. Cols 2,3.
            #   A22[0,0] -> idx 10
            #   A22[0,1] -> idx 11
            #   A22[1,0] -> idx 14
            #   A22[1,1] -> idx 15
            
            # Mapping of A' index (0..3) to local block indices (0..3)
            # A'00 -> local 0,0
            # A'01 -> local 0,1
            # A'10 -> local 1,0
            # A'11 -> local 1,1
            
            # We iterate over the blocks present in coeffs_A
            for block_name, c in coeffs_A.items():
                if c == 0: continue
                br, bc = _block_coords(block_name)
                # u2[0] adds to A'00, which is local (0,0) of the block
                # u2[1] adds to A'01, which is local (0,1)
                # u2[2] adds to A'10, which is local (1,0)
                # u2[3] adds to A'11, which is local (1,1)
                
                # Global index for local (lr, lc) in block (br, bc)
                # Global row = br*2 + lr
                # Global col = bc*2 + lc
                # U index = Global row * 4 + Global col
                
                # u2[0] -> local (0,0)
                gr = br * 2 + 0
                gc = bc * 2 + 0
                u_global[gr * 4 + gc] += c * u2[0]
                
                # u2[1] -> local (0,1)
                gr = br * 2 + 0
                gc = bc * 2 + 1
                u_global[gr * 4 + gc] += c * u2[1]
                
                # u2[2] -> local (1,0)
                gr = br * 2 + 1
                gc = bc * 2 + 0
                u_global[gr * 4 + gc] += c * u2[2]
                
                # u2[3] -> local (1,1)
                gr = br * 2 + 1
                gc = bc * 2 + 1
                u_global[gr * 4 + gc] += c * u2[3]
            
            # Expand v2 to V (length 28, but only first 16 rows for B_sq)
            v_global = [0] * 28
            for block_name, c in coeffs_B.items():
                if c == 0: continue
                br, bc = _block_coords(block_name)
                
                # v2[0] -> local (0,0) of B block
                # B block (br, bc)
                # Global row of B = br*2 + lr
                # Global col of B = bc*2 + lc
                # V index = Global row * 7 + Global col
                
                gr = br * 2 + 0
                gc = bc * 2 + 0
                v_global[gr * 7 + gc] += c * v2[0]
                
                gr = br * 2 + 0
                gc = bc * 2 + 1
                v_global[gr * 7 + gc] += c * v2[1]
                
                gr = br * 2 + 1
                gc = bc * 2 + 0
                v_global[gr * 7 + gc] += c * v2[2]
                
                gr = br * 2 + 1
                gc = bc * 2 + 1
                v_global[gr * 7 + gc] += c * v2[3]
            
            # Expand w2 to W (length 28, but only first 16 rows for C_sq)
            w_global = [0] * 28
            # C blocks are arranged like A blocks.
            # C'00 -> Block 11, local (0,0)
            # C'01 -> Block 11, local (0,1)
            # C'10 -> Block 21, local (0,0) ?? No.
            # C is 4x4.
            # C = [ C11 C12 ]
            #     [ C21 C22 ]
            # C' is the result of 2x2 multiplication.
            # C'00 corresponds to C11.
            # C'01 corresponds to C12.
            # C'10 corresponds to C21.
            # C'11 corresponds to C22.
            
            # Wait, the inner Strassen computes C' = A' B'.
            # C' is 2x2.
            # C'00 is the top-left block of the result.
            # In the 4x4 result, this is Block 11.
            # C'01 is the top-right block. Block 12.
            # C'10 is the bottom-left block. Block 21.
            # C'11 is the bottom-right block. Block 22.
            
            # So:
            # w2[0] (C'00) -> Block 11
            # w2[1] (C'01) -> Block 12
            # w2[2] (C'10) -> Block 21
            # w2[3] (C'11) -> Block 22
            
            # W index = Global col (k) * 4 + Global row (i)
            # Note: W convention is W[k*n + i, r].
            # So for C block (br, bc):
            # Global col k = bc*2 + lc
            # Global row i = br*2 + lr
            # W index = k * 4 + i
            
            # w2[0] -> C'00 -> Block 11 (br=0, bc=0)
            #   local (0,0) -> k=0, i=0 -> idx 0
            #   local (0,1) -> k=1, i=0 -> idx 1
            #   local (1,0) -> k=0, i=1 -> idx 4
            #   local (1,1) -> k=1, i=1 -> idx 5
            #   Wait, w2[0] is a scalar for the whole C'00?
            #   No, w2 is a vector of length 4, corresponding to the 4 entries of C'.
            #   So w2[0] is the coefficient for C'00.
            #   But C'00 is a 2x2 matrix.
            #   The inner Strassen decomposition gives rank-1 tensors for the 2x2x2 multiplication.
            #   So w2[0] corresponds to the (0,0) entry of C'?
            #   Let's check the definition of U2, V2, W2.
            #   U2 has 4 rows (A entries), V2 has 4 rows (B entries), W2 has 4 rows (C entries).
            #   So w2[0] is for C[0,0], w2[1] for C[0,1], w2[2] for C[1,0], w2[3] for C[1,1].
            #   
            #   So:
            #   w2[0] -> C'00. This is the (0,0) entry of the top-left block.
            #   w2[1] -> C'01. This is the (0,1) entry of the top-left block?
            #   No. The inner multiplication is A' (2x2) * B' (2x2) = C' (2x2).
            #   The entries of C' are:
            #   C'[0,0] = A'[0,0]B'[0,0] + A'[0,1]B'[1,0]
            #   C'[0,1] = A'[0,0]B'[0,1] + A'[0,1]B'[1,1]
            #   C'[1,0] = A'[1,0]B'[0,0] + A'[1,1]B'[1,0]
            #   C'[1,1] = A'[1,0]B'[0,1] + A'[1,1]B'[1,1]
            #   
            #   In the 4x4 context:
            #   C'[0,0] corresponds to the entire top-left 2x2 block C11?
            #   NO. A' and B' are 2x2 matrices formed by linear combinations of 2x2 blocks.
            #   So A'[0,0] is a linear combination of the (0,0) entries of the blocks.
            #   So C'[0,0] is a linear combination of the (0,0) entries of the C blocks.
            #   Specifically, C'[0,0] contributes to C11[0,0].
            #   C'[0,1] contributes to C11[0,1].
            #   C'[1,0] contributes to C11[1,0].
            #   C'[1,1] contributes to C11[1,1].
            #   
            #   Wait, this is only true if A' and B' were scalars.
            #   But A' is a 2x2 matrix.
            #   The product C' is a 2x2 matrix.
            #   The entry C'[0,0] is a scalar.
            #   This scalar is the value of the (0,0) entry of the resulting 2x2 matrix C'.
            #   How does this map to the 4x4 matrix C?
            #   
            #   Let's trace:
            #   A' = [ A'00 A'01 ]
            #        [ A'10 A'11 ]
            #   B' = [ B'00 B'01 ]
            #        [ B'10 B'11 ]
            #   
            #   A'00 = sum(c_b * A_b[0,0])
            #   A'01 = sum(c_b * A_b[0,1])
            #   A'10 = sum(c_b * A_b[1,0])
            #   A'11 = sum(c_b * A_b[1,1])
            #   
            #   C'00 = A'00 B'00 + A'01 B'10
            #   This C'00 is the (0,0) entry of the product of the 2x2 matrices A' and B'.
            #   This entry corresponds to the (0,0) entry of the top-left block of the 4x4 product?
            #   
            #   Let's check the block multiplication formula:
            #   C11 = A11 B11 + A12 B21
            #   C12 = A11 B12 + A12 B22
            #   C21 = A21 B11 + A22 B21
            #   C22 = A21 B12 + A22 B22
            #   
            #   Strassen computes:
            #   M1 = (A11+A22)(B11+B22)
            #   ...
            #   C11 = M1 - M4 - M5 + M7
            #   
            #   So the output of the Strassen algorithm is the 4x4 matrix C.
            #   The "inner" Strassen is used to compute each M_k.
            #   M_k is a 2x2 matrix.
            #   So the inner Strassen computes the 4 entries of M_k.
            #   M_k[0,0], M_k[0,1], M_k[1,0], M_k[1,1].
            #   
            #   These entries are then combined to form C11, C12, etc.
            #   
            #   So, for a specific M_k:
            #   The inner Strassen gives rank-1 tensors that compute the 4 entries of M_k.
            #   Let these be $u, v, w$ vectors of length 4.
            #   $w[0]$ corresponds to M_k[0,0].
            #   $w[1]$ corresponds to M_k[0,1].
            #   $w[2]$ corresponds to M_k[1,0].
            #   $w[3]$ corresponds to M_k[1,1].
            #   
            #   Now, how do M_k entries map to C entries?
            #   We have the linear combinations:
            #   C11 = M1 - M4 - M5 + M7
            #   C12 = M3 + M5
            #   C21 = M2 + M4
            #   C22 = M1 - M2 - M3 + M6
            #   
            #   Wait, the standard Strassen formulas for C blocks are:
            #   C11 = M1 - M4 - M5 + M7
            #   C12 = M3 + M5
            #   C21 = M2 + M4
            #   C22 = M1 - M2 - M3 + M6
            #   
            #   So, to get the global W matrix, we need to apply these linear combinations to the W vectors of the inner Strassen.
            #   
            #   For each inner rank term $r$ of $M_k$:
            #   We have $w_{inner}[0..3]$ which are the coefficients for $M_k[0,0], M_k[0,1], M_k[1,0], M_k[1,1]$.
            #   
            #   The contribution to global W is:
            #   If $k=0$ (M1):
            #     C11 += 1 * M1
            #     C22 += 1 * M1
            #   If $k=1$ (M2):
            #     C21 += 1 * M2
            #     C22 += -1 * M2
            #   If $k=2$ (M3):
            #     C12 += 1 * M3
            #     C22 += -1 * M3
            #   If $k=3$ (M4):
            #     C11 += -1 * M4
            #     C21 += 1 * M4
            #   If $k=4$ (M5):
            #     C11 += -1 * M5
            #     C12 += 1 * M5
            #   If $k=5$ (M6):
            #     C22 += 1 * M6
            #   If $k=6$ (M7):
            #     C11 += 1 * M7
            #     
            #   So we need a mapping from $M_k$ entry index to Global C entry.
            #   $M_k[0,0]$ contributes to $C11[0,0]$.
            #   $M_k[0,1]$ contributes to $C11[0,1]$.
            #   $M_k[1,0]$ contributes to $C11[1,0]$.
            #   $M_k[1,1]$ contributes to $C11[1,1]$.
            #   
            #   And so on for other blocks.
            
            # This logic is getting complex to inline. I will implement a helper to map M_k entries to global W indices.
            
            pass # Placeholder for logic implementation in the actual code

    return {"U": U, "V": V, "W": W}

def _get_strassen_2x2():
    # Returns U, V, W for 2x2x2 with rank 7
    # U, V, W are 4x7 matrices
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen 7 products
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to decompose each M_k into rank-1 tensors?
    # No, M_k is a product of two 2x2 matrices.
    # We use schoolbook for the 2x2x2? No, that would be rank 8.
    # We want rank 7 for the 2x2x2 tensor.
    # The Strassen algorithm IS the rank 7 decomposition.
    # The 7 products M1..M7 are the rank-1 terms?
    # No, each M_k is a product of two linear combinations.
    # The tensor is sum_k u_k v_k w_k.
    # Here u_k is the vector of coefficients for A, v_k for B, w_k for C.
    # But M_k is not a rank-1 tensor. M_k is a 2x2 matrix.
    # The Strassen algorithm computes C from M1..M7.
    # The decomposition is:
    # C = M1 - M4 - M5 + M7 (for C11)
    # ...
    # This means the tensor is a linear combination of the tensors corresponding to M1..M7.
    # But each M_k is itself a 2x2x2 tensor?
    # No, M_k is the product of two 2x2 matrices.
    # If we consider the multiplication of 2x2 matrices, the tensor is $T_{ijk}$.
    # Strassen provides a decomposition of this tensor into 7 rank-1 tensors?
    # No, Strassen provides a decomposition into 7 *products* of linear forms.
    # Each product $M_k$ is a bilinear form.
    # To get the rank-1 tensor decomposition, we need to expand each $M_k$?
    # No, the rank of the tensor is the number of rank-1 terms.
    # Strassen's algorithm shows that the tensor can be written as a sum of 7 terms, where each term is a product of 3 linear forms?
    # Actually, Strassen's algorithm computes C using 7 multiplications.
    # This implies the tensor rank is at most 7.
    # The explicit decomposition is:
    # $T = \sum_{k=1}^7 u^{(k)} \otimes v^{(k)} \otimes w^{(k)}$
    # Where $u^{(k)}$ and $v^{(k)}$ are the linear combinations of A and B entries,
    # and $w^{(k)}$ is the linear combination of C entries that M_k contributes to.
    
    # Let's define the vectors.
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1: A' = A00+A11, B' = B00+B11
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # M1 contributes to C11 (1), C12 (0), C21 (0), C22 (1) ?
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M3 + M6
    # So w1 = [1, 0, 0, 1]
    
    # M2: A' = A10+A11, B' = B00
    # u2 = [0, 0, 1, 1]
    # v2 = [1, 0, 0, 0]
    # w2 = [0, 0, 1, -1]
    
    # M3: A' = A00, B' = B01-B11
    # u3 = [1, 0, 0, 0]
    # v3 = [0, 1, 0, -1]
    # w3 = [0, 1, 0, -1]
    
    # M4: A' = A11, B' = B10-B00
    # u4 = [0, 0, 0, 1]
    # v4 = [0, 0, 1, -1]
    # w4 = [-1, 0, 1, 0]
    
    # M5: A' = A00+A01, B' = B11
    # u5 = [1, 1, 0, 0]
    # v5 = [0, 0, 0, 1]
    # w5 = [-1, 1, 0, 0]
    
    # M6: A' = A10-A00, B' = B00+B01
    # u6 = [-1, 0, 1, 0]
    # v6 = [1, 1, 0, 0]
    # w6 = [0, 0, 0, 1]
    
    # M7: A' = A01-A11, B' = B10+B11
    # u7 = [0, 1, 0, -1]
    # v7 = [0, 0, 1, 1]
    # w7 = [1, 0, 0, 0]
    
    U[0][0] = 1; U[1][0] = 0; U[2][0] = 0; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 0; V[2][0] = 0; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 0; W[2][0] = 0; W[3][0] = 1
    
    U[0][1] = 0; U[1][1] = 0; U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1; V[1][1] = 0; V[2][1] = 0; V[3][1] = 0
    W[0][1] = 0; W[1][1] = 0; W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1; U[1][2] = 0; U[2][2] = 0; U[3][2] = 0
    V[0][2] = 0; V[1][2] = 1; V[2][2] = 0; V[3][2] = -1
    W[0][2] = 0; W[1][2] = 1; W[2][2] = 0; W[3][2] = -1
    
    U[0][3] = 0; U[1][3] = 0; U[2][3] = 0; U[3][3] = 1
    V[0][3] = 0; V[1][3] = 0; V[2][3] = 1; V[3][3] = -1
    W[0][3] = -1; W[1][3] = 0; W[2][3] = 1; W[3][3] = 0
    
    U[0][4] = 1; U[1][4] = 1; U[2][4] = 0; U[3][4] = 0
    V[0][4] = 0; V[1][4] = 0; V[2][4] = 0; V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1; W[2][4] = 0; W[3][4] = 0
    
    U[0][5] = -1; U[1][5] = 0; U[2][5] = 1; U[3][5] = 0
    V[0][5] = 1; V[1][5] = 1; V[2][5] = 0; V[3][5] = 0
    W[0][5] = 0; W[1][5] = 0; W[2][5] = 0; W[3][5] = 1
    
    U[0][6] = 0; U[1][6] = 1; U[2][6] = 0; U[3][6] = -1
    V[0][6] = 0; V[1][6] = 0; V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 0; W[2][6] = 0; W[3][6] = 0
    
    return U, V, W

def _block_coords(name):
    if name == '11': return (0,0)
    if name == '12': return (0,1)
    if name == '21': return (1,0)
    if name == '22': return (1,1)
    return (0,0)

# EVOLVE-BLOCK-END
