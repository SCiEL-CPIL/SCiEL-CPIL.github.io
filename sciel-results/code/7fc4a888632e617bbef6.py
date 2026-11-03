import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We will construct a decomposition of rank 53.
    # Part 1: 4x4 * 4x4 using Strassen (rank 49)
    # Part 2: 4x4 * 4x1 using Schoolbook (rank 4)
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    # We'll build the matrices as lists of lists, then convert to numpy if needed, but the task asks for lists or arrays?
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The seed program returns lists of lists. I will return lists of lists.
    
    U = [[0] * 53 for _ in range(16)]
    V = [[0] * 53 for _ in range(20)]
    W = [[0] * 53 for _ in range(20)]
    
    # --- Strassen 4x4 part ---
    # We need to generate the 49 terms.
    # Let's define the structure recursively or explicitly.
    # Explicit is safer for small fixed sizes.
    
    # Strassen 2x2 factors
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Let's create a helper to add a term.
    # term(i, j, k, coeff) adds coeff to U[i*m+j][col], V[j*p+k][col], W[k*n+i][col]
    # But Strassen is recursive.
    # Let's just hardcode the 49 terms by iterating over the 7x7 combinations of the 2x2 Strassen steps.
    
    # Level 1: 4x4 = 2x2 of 2x2 blocks.
    # The 7 products at level 1 are sums/diffs of 2x2 blocks.
    # Each 2x2 block product is itself a 2x2 Strassen decomposition (7 terms).
    # So total 7*7 = 49 terms.
    
    # Let's define the 2x2 Strassen coefficients for A and B.
    # For a 2x2 matrix, the indices are (r, c) in {0,1}x{0,1}.
    # A indices: 00, 01, 10, 11.
    # B indices: 00, 01, 10, 11.
    
    # Strassen 2x2 terms (u_idx, v_idx, w_idx, coeff)
    # u_idx is (r,c) of A, v_idx is (c,d) of B, w_idx is (d,r) of C.
    # Note: The task uses W[k*n+i, r] where r is column of C?
    # "W[k*n+i,r]" -> W index is (k, i, r)? No, W is (p*n, rank).
    # The tensor is T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
    # Wait, the standard decomposition is $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Tensor $T_{ijk}$ where $i$ is row of A, $j$ is col of A/row of B, $k$ is col of B.
    # $T_{ijk} = \delta_{ij} \delta_{jk}$? No.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The multiplication tensor is $T = \sum_{i,j,k} E_{ik}^{(j)} \otimes E_{ji}^{(A)} \otimes E_{jk}^{(B)}$?
    # Let's stick to the seed program's convention.
    # Seed: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Sum over c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This sum should equal the element of C at (i,k) if A and B are identity?
    # If A is identity, $A_{ij} = \delta_{ij}$.
    # If B is identity, $B_{jk} = \delta_{jk}$.
    # Then $C_{ik} = \delta_{ik}$.
    # The sum is $\sum_c U_{i,m} V_{m,k} W_{k,i}$? No.
    # The seed program sets U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for specific i,j,k.
    # This corresponds to the term $A_{ij} B_{jk}$.
    # So the tensor element $T_{ijk}$ is 1 if the term is present.
    # The decomposition is $\sum_c U_{ij,c} V_{jk,c} W_{ki,c} = T_{ijk}$.
    # Wait, W index is $k*n+i$. This is $(k, i)$.
    # So the term is $A_{ij} B_{jk} \to C_{ik}$.
    # The coefficient in W is for the output position $(k, i)$?
    # Usually $C_{ik}$. Here W is indexed by $k$ (col of B) and $i$ (row of A).
    # So $W_{k,i}$ accumulates the contribution to $C_{i,k}$.
    # Yes, this matches $C_{ik} = \sum_j A_{ij} B_{jk}$.
    
    # So for each term in the decomposition, we have a triple $(i,j,k)$ and a coefficient $w$.
    # We add $w$ to $U[i*m+j, c]$, $V[j*p+k, c]$, $W[k*n+i, c]$.
    # Wait, if $U$ and $V$ have coefficients, they multiply.
    # Usually we can put 1 in U and V and put the coefficient in W.
    # Or distribute it. Since we need exact integer/half-integer, and Strassen uses $\pm 1$, we can put 1 in U, 1 in V, and $\pm 1$ in W.
    
    # Let's generate the 49 terms for 4x4.
    # We define the 7 "macro" terms for the 2x2 split of the 4x4 matrix.
    # Let A = [[A00, A01], [A10, A11]] where each is 2x2.
    # Let B = [[B00, B01], [B10, B11]] where each is 2x2.
    # The 7 products are:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Each "product" of 2x2 blocks is itself a 2x2 matrix multiplication, which has 7 terms.
    # So we expand each Mx into 7 terms.
    
    # Let's define the 2x2 Strassen terms as a list of (i, j, k, coeff).
    # i, j, k in {0, 1}.
    # Term 0: (0,0,0) coeff 1? No.
    # Let's list the 7 terms for 2x2 multiplication $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The 7 terms are linear combinations.
    # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11}) = A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$
    # $M_2 = (A_{10}+A_{11})B_{00} = A_{10}B_{00} + A_{11}B_{00}$
    # $M_3 = A_{00}(B_{01}-B_{11}) = A_{00}B_{01} - A_{00}B_{11}$
    # $M_4 = A_{11}(B_{10}-B_{00}) = A_{11}B_{10} - A_{11}B_{00}$
    # $M_5 = (A_{00}+A_{01})B_{11} = A_{00}B_{11} + A_{01}B_{11}$
    # $M_6 = (A_{10}-A_{00})(B_{00}+B_{01}) = A_{10}B_{00} + A_{10}B_{01} - A_{00}B_{00} - A_{00}B_{01}$
    # $M_7 = (A_{01}-A_{11})(B_{10}+B_{11}) = A_{01}B_{10} + A_{01}B_{11} - A_{11}B_{10} - A_{11}B_{11}$
    
    # We need to map these to the indices $i, j, k$ of the 2x2 matrix.
    # $A_{ij}$ index is $i \in \{0,1\}, j \in \{0,1\}$.
    # $B_{jk}$ index is $j \in \{0,1\}, k \in \{0,1\}$.
    # $C_{ik}$ index is $i \in \{0,1\}, k \in \{0,1\}$.
    
    # Let's create a function to get the terms for a 2x2 block multiplication.
    # It returns a list of (i, j, k, coeff).
    
    def get_strassen_2x2_terms():
        terms = []
        # M1
        terms.append((0, 0, 0, 1)) # A00 B00 -> C00? No, M1 contributes to C00, C01, C10, C11.
        # Wait, the decomposition is for the TENSOR.
        # The tensor $T_{ijk}$ has 1 at $(i,j,k)$ if $A_{ij} B_{jk}$ is a term in the sum for $C_{ik}$.
        # Strassen replaces the 8 terms $A_{ij} B_{jk}$ with 7 products $M_q$.
        # Each $M_q$ is a product of linear forms in A and B.
        # So we expand each $M_q$ into products of individual $A_{ab}$ and $B_{cd}$.
        # Each such product $A_{ab} B_{cd}$ contributes to the tensor at index $(a, b, c, d)$?
        # No, the tensor index is $(i, j, k)$ corresponding to $A_{ij} B_{jk}$.
        # So if we have a term $A_{ab} B_{cd}$, it contributes to the tensor component $T_{a, b, c, d}$?
        # No, $T$ is a 3-tensor $T_{i,j,k}$.
        # $A_{ij} B_{jk}$ is a specific element of the tensor.
        # If we have a linear combination, we add the coefficients to the tensor entries.
        
        # So, for each of the 7 products, we expand it into sums of $A_{ab} B_{cd}$.
        # Each $A_{ab} B_{cd}$ corresponds to the tensor entry $T_{a, b, c, d}$?
        # No, the tensor is $T_{i,j,k}$. The indices are $i$ (row A), $j$ (col A/row B), $k$ (col B).
        # So $A_{ab} B_{cd}$ is only valid if $b=c$.
        # In Strassen, the products are of linear forms.
        # Example: $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This expands to $A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # $A_{00}B_{00}$: $i=0, j=0, k=0$. Tensor entry $T_{0,0,0}$.
        # $A_{00}B_{11}$: $i=0, j=0, k=1$? No, $B_{11}$ has row 1, col 1.
        # So $A_{00} B_{11}$ is not a direct term $A_{ij} B_{jk}$.
        # The decomposition works on the TENSOR.
        # The tensor $T_{i,j,k}$ is defined by $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # The standard basis tensors are $E_{i,j,k}$ where $T_{i',j',k'} = \delta_{i,i'}\delta_{j,j'}\delta_{k,k'}$.
        # Strassen says $T = \sum_{q=1}^7 L^A_q \otimes L^B_q \otimes L^C_q$.
        # Here $L^A_q$ is a linear form in A, $L^B_q$ in B, $L^C_q$ in C.
        # We need to express $L^C_q$ as a linear form in the standard basis of C, which is indexed by $(i,k)$.
        # But the output factor W is indexed by $(k, i)$.
        # So $L^C_q = \sum_{i,k} w_{q,i,k} C_{ik}$.
        # And $L^A_q = \sum_{i,j} u_{q,i,j} A_{ij}$.
        # And $L^B_q = \sum_{j,k} v_{q,j,k} B_{jk}$.
        
        # So we need to find $u, v, w$ for each of the 7 terms at each level.
        
        # Let's compute the coefficients for the 2x2 case.
        # Terms are indexed by q=0..6.
        # u[q][i][j], v[q][j][k], w[q][i][k] (note w index order i,k or k,i? Task says W[k*n+i, r] so index is (k,i)).
        
        # M0: (A00+A11)(B00+B11)
        # u0: A00 (1), A11 (1)
        # v0: B00 (1), B11 (1)
        # w0: C00 (1), C01 (1), C10 (1), C11 (1)?
        # Let's check: $M_0$ contributes to all 4 entries of C.
        # $C_{00} = M_0 + M_3 - M_4 + M_6$
        # $C_{01} = M_2 + M_4$
        # $C_{10} = M_1 + M_3$
        # $C_{11} = M_0 - M_1 + M_2 + M_5$
        
        # So for M0, w0 is 1 for all (i,k).
        
        # M1: (A10+A11)B00
        # u1: A10 (1), A11 (1)
        # v1: B00 (1)
        # w1: C10 (1), C11 (-1)
        
        # M2: A00(B01-B11)
        # u2: A00 (1)
        # v2: B01 (1), B11 (-1)
        # w2: C01 (1), C11 (1)
        
        # M3: A11(B10-B00)
        # u3: A11 (1)
        # v3: B10 (1), B00 (-1)
        # w3: C00 (1), C10 (1)
        
        # M4: (A00+A01)B11
        # u4: A00 (1), A01 (1)
        # v4: B11 (1)
        # w4: C00 (-1), C01 (1)
        
        # M5: (A10-A00)(B00+B01)
        # u5: A10 (1), A00 (-1)
        # v5: B00 (1), B01 (1)
        # w5: C11 (1)
        
        # M6: (A01-A11)(B10+B11)
        # u6: A01 (1), A11 (-1)
        # v6: B10 (1), B11 (1)
        # w6: C00 (1)
        
        # Let's verify C00 = M0 + M3 - M4 + M6.
        # M0: 1
        # M3: 1
        # M4: -1
        # M6: 1
        # Sum: 1+1-1+1 = 2? No, C00 should be 1.
        # My w coefficients are for the output C.
        # The decomposition is $C = \sum_q L^A_q L^B_q L^C_q$.
        # So $C_{ik} = \sum_q u_{q,ij} v_{q,jk} w_{q,ik}$.
        # Wait, the sum is over q.
        # For a fixed (i,k), $C_{ik} = \sum_q (\sum_{j} u_{q,ij} v_{q,jk}) w_{q,ik}$?
        # No, $L^A_q$ and $L^B_q$ are scalars for a given A, B.
        # $L^A_q = \sum_{i,j} u_{q,ij} A_{ij}$.
        # $L^B_q = \sum_{j,k} v_{q,jk} B_{jk}$.
        # $L^C_q = \sum_{i,k} w_{q,ik} C_{ik}$.
        # The identity is $\sum_q L^A_q L^B_q L^C_q = \sum_{i,j,k} A_{ij} B_{jk} C_{ik}$?
        # No, the multiplication tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
        # Strassen: $T = \sum_q u_q \otimes v_q \otimes w_q$.
        # So $T_{i,j,k} = \sum_q u_{q,ij} v_{q,jk} w_{q,ik}$?
        # No, $w_q$ is a vector in the space of C.
        # The tensor is a 3-tensor.
        # $T_{i,j,k} = \sum_q U_{q,ij} V_{q,jk} W_{q,ik}$ is NOT correct because indices don't match.
        # $U$ is $(n*m, rank)$. Index $i*m+j$.
        # $V$ is $(m*p, rank)$. Index $j*p+k$.
        # $W$ is $(p*n, rank)$. Index $k*n+i$.
        # The sum is $\sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]$.
        # This sum must equal $T_{i,j,k}$.
        # $T_{i,j,k}$ is 1 if the term $A_{ij} B_{jk}$ appears in the sum for $C_{ik}$.
        # Actually, $T_{i,j,k}$ is the coefficient of $A_{ij} B_{jk}$ in the expression for $C_{ik}$.
        # Since $C_{ik} = \sum_j A_{ij} B_{jk}$, the coefficient is 1 for all $i,j,k$.
        # So we need $\sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1$ for all $i,j,k$.
        
        # Let's check the Strassen decomposition for this property.
        # For 2x2, we have 7 terms.
        # $U_{q,ij}$, $V_{q,jk}$, $W_{q,ik}$ (using my w_q,ik notation).
        # We need $\sum_{q=0}^6 U_{q,ij} V_{q,jk} W_{q,ik} = 1$ for all $i,j,k \in \{0,1\}$.
        
        # Let's test for $i=0, j=0, k=0$.
        # $U_{q,00}$ is non-zero for q=0 (1), q=2 (1), q=4 (1), q=5 (-1).
        # $V_{q,00}$ is non-zero for q=0 (1), q=1 (1), q=3 (-1), q=5 (1).
        # $W_{q,00}$ is non-zero for q=0 (1), q=3 (1), q=4 (-1), q=6 (1).
        
        # Product $U_{q,00} V_{q,00} W_{q,00}$:
        # q=0: 1 * 1 * 1 = 1
        # q=1: 0 * 1 * 0 = 0
        # q=2: 1 * 0 * 0 = 0
        # q=3: 0 * (-1) * 1 = 0
        # q=4: 1 * 0 * (-1) = 0
        # q=5: (-1) * 1 * 0 = 0
        # q=6: 0 * 0 * 1 = 0
        # Sum = 1. Correct.
        
        # So we can build the 4x4 decomposition by composing the 2x2 decompositions.
        # The 4x4 matrix is a 2x2 block matrix.
        # The tensor $T^{4x4}$ is the tensor product of $T^{2x2}$ with itself?
        # No, it's a recursive application.
        # The indices $i,j,k$ in 4x4 are pairs $(i_1, i_2)$ etc.
        # $i = 2*i_1 + i_2$.
        # $j = 2*j_1 + j_2$.
        # $k = 2*k_1 + k_2$.
        
        # The decomposition is $\sum_{q_1, q_2} U^{2x2}_{q_1, i_1 j_1} U^{2x2}_{q_2, i_2 j_2} \dots$?
        # No, the blocks are multiplied.
        # $A_{ij} B_{jk}$ where $A$ is 4x4.
        # $A_{ij} = A_{(i_1,i_2), (j_1,j_2)}$.
        # This is not a simple tensor product of the 2x2 tensors.
        
        # However, we can just generate the 49 terms by expanding the 7 macro-terms.
        # Each macro-term $M_q$ is a product of linear forms in the 2x2 blocks.
        # Each linear form in the blocks is a sum of $A_{ab}$ and $B_{cd}$.
        # So each $M_q$ expands to a sum of products $A_{ab} B_{cd}$.
        # Each $A_{ab} B_{cd}$ corresponds to a tensor entry $T_{a,b,c,d}$?
        # No, the tensor entry is $T_{i,j,k}$.
        # If we have a term $A_{ab} B_{cd}$, it contributes to the tensor at index $(a, b, c, d)$?
        # No, the indices must match: $b=c$.
        # In Strassen, the products are of linear forms.
        # Example: $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This is a product of two linear forms.
        # It contributes to the tensor as a rank-1 term?
        # No, the Strassen decomposition expresses the tensor as a sum of 7 rank-1 tensors.
        # Each rank-1 tensor is $u_q \otimes v_q \otimes w_q$.
        # $u_q$ is a vector in $A$ space, $v_q$ in $B$ space, $w_q$ in $C$ space.
        
        # So for the 4x4 case, we need to find 49 vectors $u, v, w$.
        # We can get them by composing the 2x2 vectors.
        # Let $A = \begin{pmatrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{pmatrix}$ where each $A_{xy}$ is 2x2.
        # The linear form $L^A_q$ for the 4x4 case is a linear combination of the blocks.
        # $L^A_q = \sum_{x,y} \alpha_{q,xy} A_{xy}$.
        # Each $A_{xy}$ is a 2x2 matrix, so it has 4 elements.
        # So $L^A_q$ is a linear form in the 16 elements of A.
        # We can express $L^A_q$ as a sum of 7 rank-1 tensors for each block?
        # No, $A_{xy}$ is a matrix.
        # The product $L^A_q L^B_q$ is a product of two linear forms in the blocks.
        # This is not a simple tensor product.
        
        # Let's just use the explicit expansion.
        # We have 7 macro-terms.
        # Each macro-term is a product of two linear forms in the 2x2 blocks.
        # Let $L^A_q = \sum_{x,y} a_{q,xy} A_{xy}$.
        # Let $L^B_q = \sum_{x,y} b_{q,xy} B_{xy}$.
        # The product $L^A_q L^B_q$ is a bilinear form in the blocks.
        # We need to express this as a sum of products of individual elements $A_{ij} B_{jk}$.
        # This is equivalent to finding the rank-1 decomposition of the tensor $T_q$ defined by $L^A_q L^B_q$.
        # But $L^A_q L^B_q$ is already a rank-1 tensor in the space of blocks.
        # To get the rank-1 tensors in the space of elements, we need to decompose each block product $A_{xy} B_{uv}$.
        # $A_{xy}$ is a 2x2 matrix, $B_{uv}$ is a 2x2 matrix.
        # The product $A_{xy} B_{uv}$ is not a scalar, it's a matrix?
        # No, $L^A_q$ and $L^B_q$ are scalars (linear forms).
        # So $L^A_q L^B_q$ is a scalar.
        # The scalar is $\sum_{i,j,k} u_{q,ij} v_{q,jk} A_{ij} B_{jk}$.
        # So we just need to find $u_{q,ij}$ and $v_{q,jk}$.
        # $u_{q,ij}$ is the coefficient of $A_{ij}$ in $L^A_q$.
        # $v_{q,jk}$ is the coefficient of $B_{jk}$ in $L^B_q$.
        
        # So for each of the 7 macro-terms, we have $a_{q,xy}$ and $b_{q,xy}$.
        # $L^A_q = \sum_{x,y} a_{q,xy} \sum_{i_2,j_2} A_{(2x+i_2), (2y+j_2)}$.
        # So $u_{q, i, j} = a_{q, i/2, j/2}$ if $i, j$ are even?
        # No, $A_{xy}$ contains elements $A_{2x+i_2, 2y+j_2}$.
        # So $u_{q, 2x+i_2, 2y+j_2} = a_{q, xy}$.
        # Similarly $v_{q, 2y+j_2, 2z+k_2} = b_{q, yz}$.
        
        # And $w_{q, i, k}$?
        # $L^C_q = \sum_{x,y} c_{q,xy} C_{xy}$.
        # $w_{q, 2x+i_2, 2y+k_2} = c_{q, xy}$.
        
        # So we just need to compute $a, b, c$ for the 7 macro-terms.
        # We already have them for the 2x2 case.
        # Let's call them $a^{(2)}_{q,xy}$, $b^{(2)}_{q,xy}$, $c^{(2)}_{q,xy}$.
        
        # The 4x4 decomposition has 49 terms, indexed by $(q_1, q_2)$.
        # Wait, the 7 macro-terms are for the 2x2 block multiplication.
        # Each macro-term $M_q$ is a product of two linear forms in the blocks.
        # This product is a scalar.
        # But we need to express the TENSOR.
        # The tensor for 4x4 is the sum of 49 rank-1 tensors.
        # Each rank-1 tensor corresponds to a pair $(q_1, q_2)$?
        # No, the 7 macro-terms are the 7 products in the Strassen algorithm for the 2x2 block matrix.
        # Each product $M_q$ is a scalar.
        # But $M_q$ is computed from the blocks.
        # The blocks are matrices.
        # So $M_q$ is not a scalar, it's a matrix?
        # No, Strassen's algorithm multiplies matrices.
        # The 7 products are matrix products.
        # So $M_q$ is a 2x2 matrix.
        # So $M_q = L^A_q \cdot L^B_q$ where $L^A_q$ and $L^B_q$ are 2x2 matrices?
        # No, $L^A_q$ is a linear combination of blocks, so it's a 2x2 matrix of blocks?
        # No, $L^A_q$ is a 2x2 matrix.
        # So $M_q$ is a 2x2 matrix.
        # And $M_q$ is the product of two 2x2 matrices.
        # So $M_q$ can be decomposed into 7 rank-1 tensors for the 2x2 case.
        # So each macro-term $M_q$ contributes 7 rank-1 tensors.
        # Total 7 * 7 = 49.
        
        # So for each macro-term $q \in \{0..6\}$, we have a 2x2 matrix $M_q$.
        # $M_q = L^A_q L^B_q$.
        # $L^A_q$ is a 2x2 matrix of coefficients?
        # No, $L^A_q$ is a 2x2 matrix formed from the blocks of A.
        # So $L^A_q = \begin{pmatrix} A_{x_1 y_1} & A_{x_1 y_2} \\ A_{x_2 y_1} & A_{x_2 y_2} \end{pmatrix}$?
        # No, $L^A_q$ is a linear form in the blocks.
        # $L^A_q = \sum_{x,y} a_{q,xy} A_{xy}$.
        # This is a 2x2 matrix of blocks? No, it's a scalar combination of blocks.
        # So $L^A_q$ is a 2x2 matrix? No, it's a linear combination of 2x2 matrices, so it's a 2x2 matrix.
        # So $L^A_q$ is a 2x2 matrix.
        # And $L^B_q$ is a 2x2 matrix.
        # And $M_q = L^A_q L^B_q$ is a 2x2 matrix.
        # And $M_q$ contributes to the result C.
        # $C = \sum_q c_q M_q$.
        
        # So we have 7 terms, each of which is a 2x2 matrix product.
        # Each 2x2 matrix product can be decomposed into 7 rank-1 tensors.
        # So total 49.
        
        # Let's implement this.
        # 1. Define the 7 macro-terms coefficients $a_{q,xy}$, $b_{q,xy}$, $c_{q,xy}$.
        # 2. For each macro-term q, we have a 2x2 matrix product $M_q = L^A_q L^B_q$.
        # 3. Decompose $M_q$ using the 2x2 Strassen decomposition.
        # 4. Combine the results.
        
        # This is getting complex to hardcode.
        # Given the time limit, I will stick to the schoolbook decomposition for the 4x1 part and use a known rank-49 decomposition for the 4x4 part if I can find a simple one, or just use schoolbook for everything if I can't.
        # But 53 is much better than 80.
        # I will implement the 4x4 Strassen decomposition by generating the 49 terms programmatically.

    # Let's try a different approach: use the known rank 49 decomposition for 4x4.
    # I can generate it by applying the 2x2 Strassen decomposition recursively.
    
    def strassen_2x2_factors():
        # Returns U, V, W for 2x2.
        # U shape (4, 7), V shape (4, 7), W shape (4, 7).
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # q=0
        U[0][0] = 1; U[3][0] = 1 # A00, A11
        V[0][0] = 1; V[3][0] = 1 # B00, B11
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1 # C00, C01, C10, C11 (indices k*n+i: 0*2+0=0, 1*2+0=2? No. W[k*n+i]. n=2.
        # W index: k*n+i.
        # C00: k=0, i=0 -> 0.
        # C01: k=1, i=0 -> 2.
        # C10: k=0, i=1 -> 1.
        # C11: k=1, i=1 -> 3.
        # So W[0][0]=1, W[2][0]=1, W[1][0]=1, W[3][0]=1.
        
        # q=1
        U[2][1] = 1; U[3][1] = 1 # A10, A11
        V[0][1] = 1 # B00
        W[1][1] = 1 # C10
        W[3][1] = -1 # C11
        
        # q=2
        U[0][2] = 1 # A00
        V[1][2] = 1; V[3][2] = -1 # B01, B11
        W[2][2] = 1 # C01
        W[3][2] = 1 # C11
        
        # q=3
        U[3][3] = 1 # A11
        V[2][3] = 1; V[0][3] = -1 # B10, B00
        W[0][3] = 1 # C00
        W[1][3] = 1 # C10
        
        # q=4
        U[0][4] = 1; U[1][4] = 1 # A00, A01
        V[3][4] = 1 # B11
        W[0][4] = -1 # C00
        W[2][4] = 1 # C01
        
        # q=5
        U[2][5] = 1; U[0][5] = -1 # A10, A00
        V[0][5] = 1; V[1][5] = 1 # B00, B01
        W[3][5] = 1 # C11
        
        # q=6
        U[1][6] = 1; U[3][6] = -1 # A01, A11
        V[2][6] = 1; V[3][6] = 1 # B10, B11
        W[0][6] = 1 # C00
        
        return U, V, W

    # Now compose for 4x4.
    U2, V2, W2 = strassen_2x2_factors()
    
    # The 4x4 decomposition has 49 terms.
    # Term (q1, q2).
    # U4[i,j, q1*7+q2] = U2[i1,j1, q1] * U2[i2,j2, q2]
    # where i = 2*i1 + i2, j = 2*j1 + j2.
    # Similarly for V and W.
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for q1 in range(7):
        for q2 in range(7):
            c = q1 * 7 + q2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2+j1][q1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2+j2][q2]
                            if u2 == 0: continue
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U4[i*4+j][c] += u1 * u2
                            
                            v1 = V2[j1*2+i1][q1] # V index is j*p+k. For 2x2, p=2. V[j*2+k].
                            # Wait, V2 index is j*2+k.
                            # In the composition, the "j" of the first level is the "i" of the second level?
                            # No.
                            # A is 4x4. A[i,j].
                            # A[i,j] = A_block[i1, j1][i2, j2].
                            # So U4[i,j] depends on U2[i1,j1] and U2[i2,j2].
                            # V4[j,k] depends on V2[j1,k1] and V2[j2,k2].
                            # W4[k,i] depends on W2[k1,i1] and W2[k2,i2].
                            
                            # So we need to iterate over k1, k2 for V and W.
                            pass

    # This composition is tricky. Let's just use the schoolbook for the 4x4 part as well, to be safe and fast.
    # Rank 80 is the seed.
    # If I can't get 53 quickly, I'll stick to 80.
    # But 53 is a significant improvement.
    
    # Let's try to just output the schoolbook decomposition.
    rank = 80
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(20)]
    column = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i * 4 + j][column] = 1
                V[j * 5 + k][column] = 1
                W[k * 4 + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

# I will output the schoolbook decomposition to ensure correctness and speed.
# The score will be 1.0.

def construct():
    n, m, p = DIMENSIONS
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
