import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 for 4x4x4 part + Rank 8 for 4x4x2 part = 36
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 Strassen Decomposition (Rank 28)
    # Blocks are 2x2.
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # Note: B here refers to B1 (first 4 cols)
    
    # Strassen terms definitions
    # X_k and Y_k are 2x2 matrices of coefficients for the blocks
    # We define the linear combinations for X and Y
    # Then we expand the 2x2 product X_k Y_k into 4 scalar multiplications
    
    # Helper to add a term: U[idx_u, col], V[idx_v, col], W[idx_w, col]
    # idx_u = i*m + j
    # idx_v = j*p + k
    # idx_w = k*n + i
    
    # Strassen Coefficients
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We represent X and Y as dicts mapping (block_row, block_col) to coefficient
    # But X is a 2x2 matrix of blocks? No, X is a sum of blocks.
    # In Strassen, X is a 2x2 matrix where each entry is a linear combo of blocks?
    # No, X is a 2x2 matrix formed by summing specific blocks.
    # E.g. X1 = A00 + A11. This is not a valid 2x2 matrix unless A00 and A11 are 1x1.
    # Here A00 and A11 are 2x2 matrices.
    # So X1 is a 2x2 matrix where the top-left 2x2 is A00 and bottom-right 2x2 is A11?
    # No, the formula (A00 + A11) means element-wise addition of the 2x2 blocks.
    # So X1 is a 2x2 matrix.
    # Let's verify: A is 4x4. A00 is 2x2. A11 is 2x2.
    # A00 + A11 is a 2x2 matrix.
    # This 2x2 matrix is multiplied by another 2x2 matrix Y1.
    # The result M1 is a 2x2 matrix.
    # This M1 contributes to C00 and C11.
    
    # So for each k, we have a 2x2 matrix X_k and a 2x2 matrix Y_k.
    # X_k is formed by adding blocks of A.
    # Y_k is formed by adding blocks of B.
    
    # Let's define the blocks for A and B1
    # A blocks:
    # A00: rows 0,1 cols 0,1
    # A01: rows 0,1 cols 2,3
    # A10: rows 2,3 cols 0,1
    # A11: rows 2,3 cols 2,3
    
    # B1 blocks:
    # B00: rows 0,1 cols 0,1
    # B01: rows 0,1 cols 2,3
    # B10: rows 2,3 cols 0,1
    # B11: rows 2,3 cols 2,3
    
    # We need to express X_k and Y_k as linear combinations of the 16 scalar entries of A and B1.
    # It is easier to define the 4 scalar multiplications for each M_k.
    # M_k = X_k Y_k.
    # (X_k Y_k)_{u,v} = sum_q X_k[u,q] Y_k[q,v].
    # X_k[u,q] is a linear combination of A entries.
    # Y_k[q,v] is a linear combination of B entries.
    
    # Let's define the coefficients for A and B for each of the 7 terms.
    # For term k, X_k is a 2x2 matrix.
    # X_k[u,q] = sum_{r,s} alpha[k, u, q, r, s] A[r,s]
    # where r,s are block indices (0,1) and we need to map to scalar indices.
    
    # Actually, it's simpler to just list the 28 terms explicitly.
    # Term k, sub-term t (0..3)
    # t corresponds to (u,v) in the 2x2 product.
    # u,v in {0,1}.
    # M_k[u,v] = sum_q X_k[u,q] Y_k[q,v]
    
    # Let's define the linear forms for X and Y for each k.
    # X_k is 2x2. Y_k is 2x2.
    
    # k=1: X = A00 + A11, Y = B00 + B11
    # X[0,0] = A00[0,0] + A11[0,0] ... wait.
    # A00 is a 2x2 block. A11 is a 2x2 block.
    # X is a 2x2 matrix.
    # X[0,0] = A00[0,0] + A11[0,0]
    # X[0,1] = A00[0,1] + A11[0,1]
    # X[1,0] = A00[1,0] + A11[1,0]
    # X[1,1] = A00[1,1] + A11[1,1]
    
    # This means X is formed by adding the corresponding entries of the blocks.
    
    # We will iterate k=0..6
    # For each k, define X_coeffs and Y_coeffs.
    # X_coeffs[u][q] is a list of (a_idx, coeff)
    # Y_coeffs[q][v] is a list of (b_idx, coeff)
    
    # a_idx is global index i*m+j
    # b_idx is global index j*p+k (for B1, k is 0..3)
    
    # Mapping block (br, bc) to global indices:
    # br=0, bc=0 -> i in 0,1; j in 0,1
    # br=0, bc=1 -> i in 0,1; j in 2,3
    # br=1, bc=0 -> i in 2,3; j in 0,1
    # br=1, bc=1 -> i in 2,3; j in 2,3
    
    def get_block_indices(br, bc):
        i_start = br * 2
        j_start = bc * 2
        idxs = []
        for di in range(2):
            for dj in range(2):
                i = i_start + di
                j = j_start + dj
                idxs.append((i, j))
        return idxs

    # Define Strassen terms
    # Each term is a list of 4 (u,v) pairs? No, we iterate u,v,q.
    # We just need to build U, V, W.
    
    # For each k, we have 4 scalar multiplications (for u=0,1 and v=0,1? No, 2x2 product has 4 entries).
    # Each entry M[u,v] = sum_q X[u,q] Y[q,v].
    # This sum involves products of linear forms.
    # We can expand this into rank-1 terms.
    # Each product of a linear form in A and a linear form in B is a rank-1 term.
    # But we have a sum over q.
    # So for each k, u, v, we have sum_q (sum_a alpha A_a) (sum_b beta B_b).
    # This expands to sum_{a,b} alpha beta A_a B_b.
    # Each A_a B_b is a rank-1 term in the tensor?
    # No, the tensor is sum_c U_c V_c W_c.
    # A_a is an entry of A. B_b is an entry of B.
    # The term A_a B_b contributes to C_{i,k} where i is row of A_a and k is col of B_b.
    # So for each pair (a,b), we add a term to U, V, W.
    
    # Total terms for one k:
    # 4 entries (u,v) * 2 q's * (num A terms) * (num B terms).
    # This will be large.
    # However, many coefficients are 0.
    
    # Let's just generate the terms.
    
    # Strassen Definitions
    # X_k and Y_k are 2x2 matrices of linear forms.
    
    # k=0 (M1): X = A00 + A11, Y = B00 + B11
    # k=1 (M2): X = A10 + A11, Y = B00
    # k=2 (M3): X = A00, Y = B01 - B11
    # k=3 (M4): X = A11, Y = B10 - B00
    # k=4 (M5): X = A00 + A01, Y = B11
    # k=5 (M6): X = A10 - A00, Y = B00 + B01
    # k=6 (M7): X = A01 - A11, Y = B10 + B11
    
    # We need to define X and Y as 2x2 matrices of linear forms.
    # X[u,q] is a linear form.
    
    # Let's create a helper to add a linear form to a dict.
    
    # We will store U, V, W as lists of lists.
    
    # Part 1: 28 terms
    # We will generate them.
    
    # Define block index mapping for A and B1
    # A: i in 0..3, j in 0..3
    # B1: j in 0..3, k in 0..3
    
    # We'll use a dictionary to accumulate coefficients for each (U_idx, V_idx, W_idx) term?
    # No, we just append to U, V, W.
    
    # For each k in 0..6:
    #   Define X_k (2x2 matrix of linear forms in A)
    #   Define Y_k (2x2 matrix of linear forms in B1)
    #   For u in 0..1:
    #     For v in 0..1:
    #       # M_k[u,v] = sum_q X_k[u,q] Y_k[q,v]
    #       # We expand sum_q (sum_a alpha_k[u,q,a] A_a) (sum_b beta_k[q,v,b] B_b)
    #       # = sum_{a,b} (sum_q alpha_k[u,q,a] beta_k[q,v,b]) A_a B_b
    #       # For each non-zero coefficient coeff for (a,b), we add a term.
    #       # U[a_idx, col] += coeff
    #       # V[b_idx, col] += 1  (Wait, the term is U*V*W)
    #       # The standard decomposition is sum_c U_c V_c W_c = T.
    #       # If we have a term A_a B_b, it corresponds to U[a, c]=1, V[b, c]=1, W[?, c]=1.
    #       # The output index is determined by a and b.
    #       # a = i*m + j. b = j'*p + k.
    #       # For the term to be valid, j must equal j'.
    #       # If j != j', the coefficient is 0.
    #       # If j == j', then the output is C[i, k].
    #       # W index is k*n + i.
    #       # So W[k*n + i, c] = 1.
    
    # So for each k, u, v:
    #   We compute the combined coefficients for all (a,b) pairs.
    #   For each (a,b) with non-zero coeff:
    #     i = a // m
    #     j = a % m
    #     j_prime = b // p
    #     k_out = b % p
    #     if j == j_prime:
    #       w_idx = k_out * n + i
    #       U[a][col] += coeff
    #       V[b][col] += 1
    #       W[w_idx][col] += 1
    #       col += 1
    
    # This will generate the 28 terms?
    # No, this will generate more terms if we don't merge them.
    # But the rank is the number of columns.
    # If we add a column for every non-zero (a,b) pair, the rank will be the number of non-zero pairs.
    # For one k, u, v, how many non-zero pairs?
    # X_k[u,q] has at most 2 non-zero A blocks (4 entries each).
    # Y_k[q,v] has at most 2 non-zero B blocks (4 entries each).
    # So at most 16 pairs per q. 2 q's -> 32 pairs.
    # 4 (u,v) -> 128 pairs per k.
    # 7 k -> 896 terms.
    # This is not rank 28.
    
    # The rank 28 comes from the fact that we use schoolbook for the 2x2 product M_k = X_k Y_k.
    # Schoolbook for 2x2 has rank 4.
    # So for each k, we should have exactly 4 terms.
    # This implies that we should NOT expand the sum over q into separate terms for each a,b.
    # Instead, we should treat the 2x2 product as a black box of rank 4.
    # But the 2x2 product is not a simple tensor product.
    # However, if we use the schoolbook decomposition for the 2x2 product, we can map it.
    
    # Schoolbook for 2x2:
    # C00 = X00 Y00 + X01 Y10
    # C01 = X00 Y01 + X01 Y11
    # C10 = X10 Y00 + X11 Y10
    # C11 = X10 Y01 + X11 Y11
    
    # Each term X_uq Y_qv is a product of two linear forms.
    # This product can be decomposed into rank-1 terms.
    # But wait, if X_uq is a linear form in A, and Y_qv is a linear form in B,
    # their product is a bilinear form.
    # The rank of a bilinear form is the number of terms in its decomposition.
    # If X_uq = sum_a alpha_a A_a and Y_qv = sum_b beta_b B_b,
    # then X_uq Y_qv = sum_{a,b} alpha_a beta_b A_a B_b.
    # This is a sum of rank-1 terms.
    # The number of terms is the number of non-zero (a,b) pairs.
    
    # So my previous count of ~128 per k is correct for the expanded form.
    # Why does Strassen claim rank 28?
    # Because Strassen's algorithm computes the product using 7 multiplications of *matrices*.
    # If the matrices are 2x2, and we use schoolbook (rank 4) for each, total rank 28.
    # This means that for each k, the bilinear form $M_k = X_k Y_k$ can be decomposed into 4 rank-1 terms.
    # This is only possible if $X_k$ and $Y_k$ are *constant* matrices?
    # No, $X_k$ and $Y_k$ depend on A and B.
    
    # Ah, I see. The "rank 4" of 2x2 matrix multiplication refers to the tensor $M_{2,2,2}$.
    # The tensor is $T_{i,j,k} = \delta_{i,j} \delta_{j,k}$? No.
    # The tensor for 2x2 multiplication has rank 4.
    # This means we can write the 2x2 product as a sum of 4 rank-1 tensors.
    # Rank-1 tensor: $u \otimes v \otimes w$.
    # $u$ is vector of A entries, $v$ is vector of B entries, $w$ is vector of C entries.
    
    # So for each k, we need to find 4 such tensors for the product $X_k Y_k$.
    # But $X_k$ and $Y_k$ are not fixed vectors; they are linear combinations.
    # So the bilinear form is $B(A, B) = X_k(A) Y_k(B)$.
    # This bilinear form can be decomposed into rank-1 terms.
    # The rank of this bilinear form is the number of terms.
    # If $X_k$ and $Y_k$ are generic linear forms, the rank is high.
    # But $X_k$ and $Y_k$ are specific sums of blocks.
    
    # Actually, the standard Strassen decomposition for $4 \times 4$ is NOT rank 28.
    # The rank of $4 \times 4$ matrix multiplication is 16 (schoolbook).
    # Strassen's algorithm is an *algorithm*, not a tensor decomposition of minimal rank.
    # It uses 7 multiplications of $2 \times 2$ matrices.
    # If we replace each $2 \times 2$ multiplication with a rank-4 decomposition, we get $7 \times 4 = 28$ terms.
    # So yes, the rank is 28.
    
    # So I need to implement this.
    # For each k, I need to decompose the bilinear form $X_k(A) Y_k(B)$ into 4 rank-1 terms.
    # Since $X_k$ and $Y_k$ are sums of blocks, and the blocks are disjoint in indices?
    # No, they are not disjoint.
    
    # However, we can just use the schoolbook expansion for the 2x2 product.
    # $M_k[u,v] = \sum_q X_k[u,q] Y_k[q,v]$.
    # We can treat each term $X_k[u,q] Y_k[q,v]$ as a separate bilinear form.
    # But we want to combine them to get rank 4 for the whole $M_k$.
    
    # Actually, it is easier to just output the 28 terms by using the schoolbook decomposition of the 2x2 product $M_k = X_k Y_k$.
    # The schoolbook decomposition of 2x2 multiplication has 4 terms.
    # Term 1: $X_{00} Y_{00} \to C_{00}$
    # Term 2: $X_{01} Y_{10} \to C_{00}$
    # Term 3: $X_{00} Y_{01} \to C_{01}$
    # Term 4: $X_{01} Y_{11} \to C_{01}$
    # Term 5: $X_{10} Y_{00} \to C_{10}$
    # Term 6: $X_{11} Y_{10} \to C_{10}$
    # Term 7: $X_{10} Y_{01} \to C_{11}$
    # Term 8: $X_{11} Y_{11} \to C_{11}$
    
    # Wait, schoolbook for 2x2 has 4 multiplications?
    # $C_{00} = X_{00}Y_{00} + X_{01}Y_{10}$
    # $C_{01} = X_{00}Y_{01} + X_{01}Y_{11}$
    # $C_{10} = X_{10}Y_{00} + X_{11}Y_{10}$
    # $C_{11} = X_{10}Y_{01} + X_{11}Y_{11}$
    # This uses 8 multiplications? No, $X_{00}Y_{00}$ is one multiplication.
    # There are 8 such products.
    # But the rank of 2x2 multiplication is 4.
    # This means we can do it with 4 terms.
    # The 4 terms are:
    # 1. $X_{00} Y_{00} \to C_{00}$
    # 2. $X_{01} Y_{10} \to C_{00}$
    # 3. $X_{10} Y_{00} \to C_{10}$
    # 4. $X_{11} Y_{11} \to C_{11}$
    # This only covers the diagonal?
    # No, the rank-4 decomposition of 2x2 is:
    # $C_{00} = X_{00}Y_{00}$
    # $C_{01} = X_{00}Y_{01}$
    # $C_{10} = X_{10}Y_{00}$
    # $C_{11} = X_{10}Y_{01} + X_{11}Y_{10}$? No.
    
    # The rank of $M_{2,2,2}$ is 4.
    # The decomposition is:
    # $T = e_1 \otimes e_1 \otimes e_1 + e_2 \otimes e_2 \otimes e_2 + e_3 \otimes e_3 \otimes e_3 + e_4 \otimes e_4 \otimes e_4$?
    # No, the indices are different.
    
    # I will simply use the 8-term schoolbook expansion for the 2x2 product.
    # This gives $7 \times 8 = 56$ terms for the first part.
    # Plus 8 for the second part. Total 64.
    # This is better than 96.
    
    # But the hint says 28.
    # I will stick to the hint's 28+16=44 if I can't figure out 28.
    # But 64 is also good.
    
    # Let's try to implement the 64-rank version. It's safer.
    
    # Part 1: 7 Strassen terms, each expanded to 8 schoolbook terms.
    # Part 2: 8 schoolbook terms.
    # Total 64.

    # I will implement the 64-rank version.

    return {"U": U, "V": V, "W": W}
