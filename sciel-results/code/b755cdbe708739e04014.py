import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    # Split B into B1 (4x4) and B2 (4x3)
    # Part 1: A * B1 (4x4x4) using Strassen (Rank 28)
    # Part 2: A * B2 (4x4x3) using Trivial (Rank 48)
    
    rank = 28 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    
    # --- Part 1: 4x4x4 Strassen ---
    # Strassen terms for 2x2 blocks of A and B
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to the global 4x4 indices.
    # Block (I,J) of A corresponds to rows 2I..2I+1, cols 2J..2J+1.
    # Element (i,j) in block (I,J) is global (2I+i, 2J+j).
    
    # For each Strassen term, we have a 2x2 matrix product.
    # We decompose the 2x2 product into 4 scalar multiplications.
    
    def add_strassen_term(U_coeff, V_coeff, W_add, t_offset):
        # U_coeff: 4x4 matrix of coefficients for A
        # V_coeff: 4x4 matrix of coefficients for B
        # W_add: 4x4 matrix of coefficients for C (result of this term)
        # This function adds 4 rank-1 tensors for the trivial 2x2x2 decomposition of this block product
        # Wait, the "block product" is 2x2 x 2x2 -> 2x2.
        # The trivial decomposition of 2x2x2 has rank 4.
        # We iterate over the 4 scalar multiplications in the 2x2 product.
        
        # The term is sum_{j=0}^1 L_A[i,j] L_B[j,k]
        # L_A and L_B are 2x2 matrices.
        # We need to expand L_A and L_B into global A and B indices.
        
        # This function is too abstract. I will just generate the 28 tensors directly.
        pass

    # Direct generation of 28 tensors for 4x4x4 Strassen
    # Indices: i, j, k in 0..3
    # u_idx = i*4 + j
    # v_idx = j*4 + k  (Note: B is 4x4 here, so j is row, k is col)
    # w_idx = k*4 + i
    
    # Strassen definitions for block indices I, J, K in {0,1}
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # For each M_t, we compute 2x2 product.
    # Let's define the linear combinations for each M_t.
    # A_comb: list of (I, J, coeff)
    # B_comb: list of (J, K, coeff)
    
    strassen_terms = [
        # M1
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)]
        },
        # M2
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)]
        },
        # M3
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)]
        },
        # M4
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)]
        },
        # M5
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)]
        },
        # M6
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)]
        },
        # M7
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)]
        }
    ]
    
    # For each term, we have a 2x2x2 tensor product.
    # We decompose the 2x2x2 tensor trivially (rank 4).
    # The 2x2x2 tensor T_{ijk} = A_{ij} B_{jk} C_{ik}.
    # The trivial decomposition is sum_{j} A_{ij} B_{jk} C_{ik}.
    # But here A and B are linear combos of blocks.
    
    # Let's expand the linear combos.
    for term in strassen_terms:
        A_comb = term['A']
        B_comb = term['B']
        
        # We need to form the 2x2 matrices L_A and L_B.
        # L_A[i,j] = sum_{(I,J,c) in A_comb, I==i, J==j} c
        # L_B[j,k] = sum_{(J,K,c) in B_comb, J==j, K==k} c
        
        L_A = [[0]*2 for _ in range(2)]
        for I, J, c in A_comb:
            L_A[I][J] += c
            
        L_B = [[0]*2 for _ in range(2)]
        for J, K, c in B_comb:
            L_B[J][K] += c
            
        # Now decompose L_A * L_B using trivial 2x2x2 (rank 4)
        # For each intermediate index j_block in {0,1}:
        for j_block in range(2):
            # The term is sum_{i_block, k_block} L_A[i_block, j_block] L_B[j_block, k_block] C[i_block, k_block]
            # We add a rank-1 tensor for this j_block?
            # No, the trivial decomposition is sum_{j} (A_{ij}) (B_{jk}) (C_{ik}).
            # Here, the "scalar" multiplication is L_A[i,j] * L_B[j,k].
            # This contributes to C[i,k].
            # So for each i_block, k_block, we have a term L_A[i_block, j_block] * L_B[j_block, k_block] * C[i_block, k_block].
            # This is a rank-1 tensor in the 2x2x2 space.
            # We need to map this to the global 4x4x4 space.
            
            for i_block in range(2):
                for k_block in range(2):
                    u_val = L_A[i_block][j_block]
                    v_val = L_B[j_block][k_block]
                    if u_val == 0 or v_val == 0:
                        continue
                    
                    # This term contributes to the global C at block (i_block, k_block).
                    # The global C element is C[2*i_block + i_local, 2*k_block + k_local].
                    # The global A element is A[2*i_block + i_local, 2*j_block + j_local].
                    # The global B element is B[2*j_block + j_local, 2*k_block + k_local].
                    
                    # Wait, the linear combination L_A[i_block, j_block] involves specific A elements.
                    # We need to expand the linear combination into individual rank-1 tensors?
                    # No, the linear combination is part of the U and V matrices.
                    # U vector will have non-zero entries for all A elements involved in L_A.
                    # V vector will have non-zero entries for all B elements involved in L_B.
                    # W vector will have 1 at the C position.
                    
                    # But we are adding ONE rank-1 tensor for the whole product L_A * L_B?
                    # No, L_A * L_B is a matrix.
                    # The decomposition of the matrix product $C = AB$ is $\sum_{j} A_{ij} B_{jk} C_{ik}$.
                    # Here, $A$ and $B$ are the linear combinations.
                    # So for a fixed $j_block$, we have a term $L_A[:, j_block] L_B[j_block, :] C$.
                    # This is an outer product of the column of L_A and row of L_B, multiplied by C?
                    # No.
                    # The scalar multiplication is $x \cdot y$ where $x = L_A[i,j]$ and $y = L_B[j,k]$.
                    # This is not a single scalar multiplication.
                    # $L_A[i,j]$ is a linear combo of 4 scalars.
                    # $L_B[j,k]$ is a linear combo of 4 scalars.
                    # Their product is a sum of 16 scalar products.
                    # So we need 16 rank-1 tensors per $i, j, k$?
                    # That would be $7 \times 4 \times 16 = 448$. Too high.
                    
                    # I am overcomplicating.
                    # The rank of $[4,4,4]$ using Strassen is 28.
                    # This means there are 28 rank-1 tensors.
                    # Each rank-1 tensor corresponds to ONE scalar multiplication.
                    # The scalar multiplication is $a \cdot b$ where $a$ is an element of $A$ and $b$ is an element of $B$.
                    # The Strassen algorithm replaces 4 block multiplications with 7.
                    # Each block multiplication is $2 \times 2 \times 2$.
                    # If we use the trivial $2 \times 2 \times 2$ (rank 4), we get $7 \times 4 = 28$ scalar multiplications.
                    # So yes, there are 28 scalar multiplications.
                    
                    # How to identify them?
                    # For each Strassen term $t$, and for each of the 4 scalar multiplications in the inner $2 \times 2 \times 2$ tensor:
                    # The inner $2 \times 2 \times 2$ tensor is $T_{ijk} = L_A_{ij} L_B_{jk} C_{ik}$.
                    # The trivial decomposition is $\sum_{j=0}^1 L_A_{ij} L_B_{jk} C_{ik}$.
                    # Wait, this sum is over $j$.
                    # So for a fixed $t$, we have 4 terms (for $j=0,1$ and $i,k$? No).
                    # The trivial decomposition of $2 \times 2 \times 2$ has 4 terms:
                    # $j=0: i=0,k=0; i=0,k=1; i=1,k=0; i=1,k=1$? No.
                    # The tensor is $T_{ijk}$.
                    # The rank-1 tensors are $e_i \otimes e_j \otimes e_k$? No.
                    # The trivial decomposition is $\sum_{i,j,k} A_{ij} B_{jk} C_{ik}$.
                    # There are $2 \times 2 \times 2 = 8$ such terms? No.
                    # The rank of $[2,2,2]$ is 4? No, it's 7 (Strassen) or 8 (Trivial).
                    # Trivial rank is 8.
                    # So $7 \times 8 = 56$.
                    # My previous rank 28 was wrong.
                    # Rank of $[2,2,2]$ is 8 (trivial).
                    # So Strassen 4x4x4 using trivial inner blocks has rank $7 \times 8 = 56$.
                    # 56 + 48 = 104.
                    # Still better than 112.
                    
                    # Let's use rank 56 for the first part.
                    pass

    # I will just use the trivial 4x4x4 (rank 64) and trivial 4x4x3 (rank 48).
    # Total 112.
    # Wait, I can use Strassen for 4x4x4 with rank 49 (recursive Strassen).
    # 49 + 48 = 97.
    # I will implement rank 49 for 4x4x4.
    
    # Reset
    rank = 49 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    t = 0
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # This is the tensor product of two 2x2x2 Strassen tensors?
    # No, it's a block structure.
    # I will just use the trivial 4x4x4 (rank 64) to be safe and fast.
    # 64 + 48 = 112.
    # I will try to implement the 56 rank version (Strassen outer, Trivial inner).
    
    rank = 56 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    t = 0
    
    strassen_terms = [
        { 'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)] },
        { 'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)] },
        { 'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)] },
        { 'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)] },
        { 'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)] },
        { 'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)] },
        { 'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)] }
    ]
    
    for term in strassen_terms:
        A_comb = term['A']
        B_comb = term['B']
        
        # Expand linear combinations to find non-zero coefficients for each A and B element
        # A elements are indexed by (I, J, i, j) -> global (2I+i, 2J+j)
        # B elements are indexed by (J, K, j, k) -> global (2J+j, 2K+k)
        
        # For each scalar multiplication in the inner 2x2x2 trivial decomposition:
        # The inner tensor is L_A * L_B.
        # Trivial decomposition: sum_{i,j,k} L_A_{ij} L_B_{jk} C_{ik}.
        # There are 8 terms (i,j,k in 0..1).
        
        for i_block in range(2):
            for j_block in range(2):
                for k_block in range(2):
                    # Coeff of L_A[i_block, j_block]
                    u_coeff = 0
                    for I, J, c in A_comb:
                        if I == i_block and J == j_block:
                            u_coeff += c
                    if u_coeff == 0: continue
                    
                    # Coeff of L_B[j_block, k_block]
                    v_coeff = 0
                    for J, K, c in B_comb:
                        if J == j_block and K == k_block:
                            v_coeff += c
                    if v_coeff == 0: continue
                    
                    # This term contributes to C[i_block, k_block]
                    # The global A elements involved are in block (i_block, j_block)
                    # The global B elements involved are in block (j_block, k_block)
                    # The global C element is in block (i_block, k_block)
                    
                    # We need to add a rank-1 tensor for EACH combination of local indices?
                    # No, the linear combination L_A_{ij} is a sum of 4 scalars.
                    # We cannot combine them into one rank-1 tensor.
                    # We must expand the linear combination.
                    
                    # This means for each (i_block, j_block, k_block), we add up to 16 rank-1 tensors.
                    # Total rank would be 7 * 8 * 16 = 896. Too high.
                    
                    # I am stuck on the construction. I will fallback to Trivial 112.
                    pass

    # Fallback to Trivial
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
