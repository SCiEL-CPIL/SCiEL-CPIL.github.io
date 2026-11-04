import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose 4x5x6 into:
    # 1. 4x4x4 (Strassen, Rank 49)
    # 2. 4x4x2 (Schoolbook, Rank 32)
    # 3. 4x1x6 (Rank 1)
    # Total Rank = 49 + 32 + 1 = 82
    
    rank_total = 82
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    c_offset = 0
    
    # Part 1: 4x4x4 using Strassen
    # A_L (4x4) * B_T1 (4x4) -> C11 (4x4)
    # Strassen factors for 4x4x4
    # We generate the 7 factors for 2x2x2 and combine them for 4x4x4?
    # Or just hardcode the 49 factors for 4x4x4?
    # It's easier to generate Strassen for 2x2x2 and use it 4 times? No, 4x4x4 is one block.
    # Strassen for 4x4x4 uses 7^2 = 49 multiplications.
    # We can build it by applying Strassen to 2x2x2 blocks of the 4x4x4 tensor.
    
    # Let's generate the 49 factors for 4x4x4.
    # Indices for 4x4x4: i in 0..3, j in 0..3, k in 0..3.
    # We will use a helper to get Strassen factors for 2x2x2.
    
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # U is 4x7, V is 4x7, W is 4x7
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Strassen's 7 multiplications for 2x2 matrices
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices: U[i,j], V[j,k], W[k,i]
        # i,j in 0..1 for A, j,k in 0..1 for B, k,i in 0..1 for C
        
        # c=0: M1
        # A parts: (1,1) and (2,2) -> indices (0,0) and (1,1) in U
        # B parts: (1,1) and (2,2) -> indices (0,0) and (1,1) in V
        # C parts: (1,1) +=, (1,2) -=, (2,1) -=, (2,2) +=
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[1, 0] = -1; W_s[2, 0] = -1; W_s[3, 0] = 1
        
        # c=1: M2
        # A parts: (2,1) and (2,2) -> indices (1,0) and (1,1) in U
        # B parts: (1,1) -> index (0,0) in V
        # C parts: (2,1) +=, (2,2) +=
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = 1
        
        # c=2: M3
        # A parts: (1,1) -> index (0,0) in U
        # B parts: (2,1) and (2,2) -> indices (1,0) and (1,1) in V
        # C parts: (1,2) +=, (2,2) -=
        U_s[0, 2] = 1
        V_s[2, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = -1
        
        # c=3: M4
        # A parts: (2,2) -> index (1,1) in U
        # B parts: (1,1) and (2,1) -> indices (0,0) and (1,0) in V
        # C parts: (2,1) -=, (2,2) +=
        U_s[3, 3] = 1
        V_s[0, 3] = 1; V_s[2, 3] = -1
        W_s[2, 3] = -1; W_s[3, 3] = 1
        
        # c=4: M5
        # A parts: (1,1) and (1,2) -> indices (0,0) and (0,1) in U
        # B parts: (2,2) -> index (1,1) in V
        # C parts: (1,1) +=, (1,2) +=
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = 1; W_s[1, 4] = 1
        
        # c=5: M6
        # A parts: (2,1) and (1,1) -> indices (1,0) and (0,0) in U
        # B parts: (1,1) and (1,2) -> indices (0,0) and (0,1) in V
        # C parts: (2,1) +=, (2,2) -=
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[2, 5] = 1; W_s[3, 5] = -1
        
        # c=6: M7
        # A parts: (1,2) and (2,2) -> indices (0,1) and (1,1) in U
        # B parts: (2,1) and (2,2) -> indices (1,0) and (1,1) in V
        # C parts: (1,1) +=, (1,2) -=
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1; W_s[1, 6] = -1
        
        return U_s, V_s, W_s
    
    U_2, V_2, W_2 = strassen_2x2x2()
    
    # Build 4x4x4 factors from 2x2x2
    # 4x4 matrix is 4 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A B
    # We apply Strassen to the 4 blocks.
    # This requires 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7 (using Strassen again? Or schoolbook 4?).
    # To get rank 49, we use Strassen for the 2x2 multiplications too.
    # So we have 7 terms, each contributing 7 sub-terms.
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U_4 index: block_row * 4 + row_in_block * 2 + col_in_block
            # U_2 index: row * 2 + col
            # We need to map the block structure.
            # U_2[c1] gives coefficients for A blocks.
            # The indices in U_2 are (i, j) where i, j in 0..1 (block indices? No, element indices).
            # Wait, U_2 is for 2x2x2.
            # When we use it for 4x4x4, the "elements" of the 2x2 matrix A are themselves 2x2 matrices.
            # So U_2[i, j, c1] tells us how to combine the 2x2 blocks A_{ij}.
            # But we are decomposing the 4x4 tensor.
            # The indices for U_4 are (i, j) where i, j in 0..3.
            # Block index bi = i // 2, bj = j // 2.
            # Local index li = i % 2, lj = j % 2.
            
            # U_2[bi * 2 + bj, c1] is the coefficient for the block A_{bi, bj}.
            # We need to distribute this to all elements in that block.
            # But we also need to combine with the internal Strassen decomposition of the block multiplication?
            # No, the 4x4x4 Strassen decomposition computes 7 products of 2x2 matrices.
            # Each product is $P_c = ( \sum U_{block} A_{block} ) ( \sum V_{block} B_{block} )$.
            # Then $C = \sum W_{block} P_c$.
            # To get the full 4x4 tensor, we expand the 2x2 products using Strassen again.
            
            # Let's just iterate over all i, j, k in 0..3.
            # For a given c1, c2:
            # The term is $U_{4, i, j, c} V_{4, j, k, c} W_{4, k, i, c}$.
            # This term comes from the product of the c1-th Strassen term of 4x4 and c2-th Strassen term of the internal 2x2.
            
            # Actually, it's easier to just say:
            # The 4x4x4 Strassen decomposition is the Kronecker product of the 2x2x2 Strassen decomposition with itself?
            # No, it's the composition.
            
            # Let's use the property that if we have a decomposition for $M_{n,m,p}$, we can get one for $M_{2n, 2m, 2p}$ by Kronecker products.
            # $U_{2n, 2m, 2p} = U_n \otimes U_n$?
            # Yes, if we use the same decomposition for the blocks.
            # So $U_4 = U_2 \otimes U_2$.
            
    U_4 = np.kron(U_2, U_2)
    V_4 = np.kron(V_2, V_2)
    W_4 = np.kron(W_2, W_2)
    
    # Verify shapes: U_2 is 4x7. U_4 is 16x49. Correct.
    
    # Add to global U, V, W
    # Mapping for Part 1:
    # A_L is A[:, 0:4]. So i in 0..3, j in 0..3.
    # B_T1 is B[0:4, 0:4]. So j in 0..3, k in 0..3.
    # C11 is C[0:4, 0:4]. So k in 0..3, i in 0..3.
    
    for i in range(4):
        for j in range(4):
            idx_u = i * 5 + j
            for c in range(49):
                U[idx_u][c_offset + c] = U_4[i * 4 + j, c]
                
    for j in range(4):
        for k in range(4):
            idx_v = j * 6 + k
            for c in range(49):
                V[idx_v][c_offset + c] = V_4[j * 4 + k, c]
                
    for k in range(4):
        for i in range(4):
            idx_w = k * 4 + i
            for c in range(49):
                W[idx_w][c_offset + c] = W_4[k * 4 + i, c]
                
    c_offset += 49
    
    # Part 2: 4x4x2 Schoolbook
    # A_L (4x4) * B_T2 (4x2) -> C12 (4x2)
    # Rank 32
    for i in range(4):
        for j in range(4):
            for k in range(2):
                c = c_offset + (i * 4 + j) * 2 + k
                U[i * 5 + j][c] = 1
                V[j * 6 + (4 + k)][c] = 1 # B_T2 is B[0:4, 4:6]
                W[(4 + k) * 4 + i][c] = 1 # C12 is C[0:4, 4:6]
                
    c_offset += 32
    
    # Part 3: 4x1x6 Rank 1
    # A_R (4x1) * B_B (1x6) -> C2 (4x6)
    # C_{ik} = A_{i4} B_{4k}
    # U_{i, 4, c} = 1
    # V_{4, k, c} = 1
    # W_{k, i, c} = 1
    # But we need to ensure the delta condition.
    # For this specific block, the tensor is rank 1.
    # U[i, 4, c] = 1 for all i?
    # V[4, k, c] = 1 for all k?
    # W[k, i, c] = 1?
    # Let's check if this satisfies the identity.
    # Sum_c U[i', 4, c] V[4, k', c] W[k, i, c]
    # = 1 * 1 * 1 = 1.
    # We need delta[i, i'] delta[4, 4] delta[k, k'] = delta[i, i'] delta[k, k'].
    # The constant 1 does not give deltas.
    # So Rank 1 is NOT sufficient for the tensor identity?
    # The tensor $M_{4,1,6}$ has rank 1?
    # $C_{ik} = A_{i0} B_{0k}$.
    # The tensor is $T_{i, 0, k} = 1$.
    # If we use $U_{i, 0, 0}=1, V_{0, k, 0}=1, W_{k, i, 0}=1$,
    # The reconstructed tensor is $R_{i, 0, k, i', 0, k'} = 1$.
    # But we need $\delta_{i,i'} \delta_{k,k'}$.
    # So we need $W$ to provide the deltas.
    # $W_{k, i, c}$ must be such that $\sum_c W_{k, i, c} \dots = \delta_{i,i'} \delta_{k,k'}$.
    # This requires rank at least $4 \times 6 = 24$?
    # Or can we do it with fewer?
    # $C_{ik} = A_{i0} B_{0k}$.
    # This is a rank-1 matrix.
    # The tensor rank of $M_{n,1,p}$ is $\min(n, p)$? No.
    # It is $n \times p$?
    # Let's check $2 \times 1 \times 2$.
    # $C_{00} = A_{00} B_{00}$
    # $C_{01} = A_{00} B_{01}$
    # $C_{10} = A_{10} B_{00}$
    # $C_{11} = A_{10} B_{01}$
    # We need 4 terms?
    # $C_{00} = U_{0,0,c} V_{0,0,c} W_{0,0,c} \dots$
    # If we use schoolbook, rank is 4.
    # Can we do better?
    # $C_{ik} = A_{i0} B_{0k}$.
    # This is an outer product.
    # The tensor rank is 1?
    # No, the tensor rank is the number of terms in the decomposition.
    # $C_{ik} = \sum_c U_{i,0,c} V_{0,k,c} W_{k,i,c}$.
    # If $R=1$, $C_{ik} = U_{i,0} V_{0,k} W_{k,i}$.
    # We need this to be $A_{i0} B_{0k}$?
    # No, $U, V, W$ are constants. The inputs $A, B$ are multiplied by them.
    # The formula is $C_{ik} = \sum_{j, c} A_{ij} B_{jk} \dots$?
    # The code's condition is on the identity tensor.
    # For $2 \times 1 \times 2$, the identity tensor has 4 non-zero entries.
    # So rank is 4?
    # Yes, schoolbook rank is $n \times m \times p = 4$.
    # So for $4 \times 1 \times 6$, rank is $4 \times 1 \times 6 = 24$.
    
    # So Part 3 Rank is 24.
    # Total Rank = 49 + 32 + 24 = 105.
    
    # Let's add Part 3 with Rank 24.
    for i in range(4):
        for k in range(6):
            c = c_offset + i * 6 + k
            U[i * 5 + 4][c] = 1
            V[4 * 6 + k][c] = 1
            W[k * 4 + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
