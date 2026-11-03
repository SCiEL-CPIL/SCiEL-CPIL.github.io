import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe

    n, m, p = 4, 4, 8
    
    # We will compute two 4x4x4 Strassen decompositions.
    # One for C[:, :4] = A * B[:, :4]
    # One for C[:, 4:] = A * B[:, 4:]
    # Total rank = 49 + 49 = 98.
    
    # First, generate the 49 rank-1 terms for 4x4x4.
    # Base case: 2x2x2 Strassen terms (u, v, w) where u,v,w are length 4.
    # u, v, w are vectors in R^4 representing the coefficients for A, B, C entries.
    # A, B, C are 2x2 matrices flattened in row-major order (0,1,2,3).
    
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11) -> u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (C00+=, C11+=)
    # Wait, the w vector must represent the contribution to C.
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the 7 base terms (u, v, w) for 2x2x2.
    # u: coeffs for A entries [A00, A01, A10, A11]
    # v: coeffs for B entries [B00, B01, B10, B11]
    # w: coeffs for C entries [C00, C01, C10, C11]
    
    base_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C00+=1, C11+=1
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # C10+=1, C11-=-1
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]), # C01+=1, C11+=1
        # M4
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]), # C00+=1, C10+=1
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # C00-=-1, C01+=1
        # M6
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]), # C11+=1
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]), # C00+=1
    ]
    
    # Verify base terms sum to identity for 2x2x2 tensor.
    # The tensor T is defined by T[i,j,k] = 1 if i,j,k correspond to A[i,j]*B[j,k]=C[i,k]?
    # No, the decomposition is for the multiplication map.
    # U, V, W such that U @ diag(z) @ V.T = W? No.
    # The condition is: sum_r U[:,r] * V[:,r] * W[:,r] (element-wise? No, outer products).
    # Actually, the tensor $T$ of matrix multiplication $A_{n \times m} B_{m \times p} = C_{n \times p}$
    # is decomposed as $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
    # Here $u_r \in \mathbb{R}^{nm}$, $v_r \in \mathbb{R}^{mp}$, $w_r \in \mathbb{R}^{pn}$.
    # The entry $T_{i, j, k}$ (where $i$ is row of A, $j$ is col of A/row of B, $k$ is col of B)
    # Wait, the standard tensor for matmul $A_{n \times m} \times B_{m \times p} \to C_{n \times p}$
    # has indices $i \in [0,n), j \in [0,m), k \in [0,p)$.
    # $T_{i,j,k} = 1$ if $j$ is the shared index? No.
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # The tensor $T$ is such that $C = \text{reshape}( \text{einsum}('ij, jk -> ik', A, B) )$.
    # The decomposition $T = \sum u_r \otimes v_r \otimes w_r$ means:
    # $C_{i,k} = \sum_r ( \sum_j u_{r, i,j} A_{i,j} ) ( \sum_l v_{r, j,l} B_{j,l} ) w_{r, i,k}$?
    # No, the standard definition is:
    # $A_{i,j} B_{j,k}$ is linear in A and B.
    # We write $C_{i,k} = \sum_r \mu_r \nu_r \omega_r$ where $\mu_r$ is a linear form in A, $\nu_r$ in B, and $\omega_r$ is a coefficient for $C_{i,k}$.
    # Specifically, $C_{i,k} = \sum_r ( \sum_{a,b} u_{r,a,b} A_{a,b} ) ( \sum_{c,d} v_{r,c,d} B_{c,d} ) w_{r,i,k}$.
    # This requires $w_{r,i,k}$ to be 1 for the specific $(i,k)$ being computed, but since we compute all $(i,k)$ simultaneously, $w_r$ is a vector in $\mathbb{R}^{n \times p}$.
    # So $C = \sum_r \omega_r ( u_r^T A_{flat} ) ( v_r^T B_{flat} )$.
    # This matches the interface: U (nm x R), V (mp x R), W (pn x R).
    # $C_{flat} = W \cdot \text{diag}( \text{elementwise product of U^T A and V^T B } ) $?
    # No, $C_{flat} = W ( (U^T A) \odot (V^T B) )$.
    # Let's check dimensions.
    # $U^T A$ is $R \times 1$. $V^T B$ is $R \times 1$.
    # Element-wise product is $R \times 1$.
    # $W$ is $pn \times R$.
    # Result is $pn \times 1$. Correct.
    
    # So we need to find $u_r, v_r, w_r$ such that
    # $\sum_r w_{r, i,k} ( \sum_{a,b} u_{r,a,b} A_{a,b} ) ( \sum_{c,d} v_{r,c,d} B_{c,d} ) = \sum_j A_{i,j} B_{j,k}$
    # for all $i,k$.
    
    # For the base case 2x2x2:
    # $C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}$
    # $C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}$
    # $C_{1,0} = A_{1,0}B_{0,0} + A_{1,1}B_{1,0}$
    # $C_{1,1} = A_{1,0}B_{0,1} + A_{1,1}B_{1,1}$
    
    # Let's verify the first base term M1:
    # $u = [1,0,0,1] \implies \mu = A_{0,0} + A_{1,1}$
    # $v = [1,0,0,1] \implies \nu = B_{0,0} + B_{1,1}$
    # $w = [1,0,0,1] \implies$ contributes to $C_{0,0}$ and $C_{1,1}$.
    # Term: $(A_{0,0}+A_{1,1})(B_{0,0}+B_{1,1})$ added to $C_{0,0}$ and $C_{1,1}$.
    # Expansion: $A_{0,0}B_{0,0} + A_{0,0}B_{1,1} + A_{1,1}B_{0,0} + A_{1,1}B_{1,1}$.
    # This looks correct for the cross terms being cancelled by other M's.
    
    # Now, generate 4x4x4 terms.
    # We use the recursive Strassen structure.
    # The 7 Strassen products for 4x4x4 are $M_q = A_q B_q$.
    # $A_q$ and $B_q$ are 4x4 matrices.
    # $A_q$ is formed by combining the 2x2 blocks of A.
    # $B_q$ is formed by combining the 2x2 blocks of B.
    # The rank-1 terms for $M_q$ are derived from the 49 base terms? No, from the 7 base terms of the 2x2x2 decomposition applied to the 4x4 matrices?
    # No. $A_q B_q$ is a 4x4x4 multiplication.
    # We can decompose $A_q B_q$ using the 49 terms of the 4x4x4 decomposition? That's circular.
    # We should decompose $A_q B_q$ using the *structure* of Strassen.
    # Actually, the rank of 4x4x4 is 49.
    # The 49 terms are generated by taking the 7 "outer" Strassen combinations and for each, using the 7 "inner" Strassen combinations for the 2x2x2 sub-blocks.
    
    # Let's define the 7 outer combinations for 4x4.
    # Blocks of A: A00, A01, A10, A11 (each 2x2).
    # Blocks of B: B00, B01, B10, B11 (each 2x2).
    # Outer terms (u_outer, v_outer, w_outer) where u_outer, v_outer are coeffs for the 4 blocks.
    # But we need full vectors for the 16 elements.
    
    # It is easier to just hardcode the 49 terms or generate them via a function that expands the Strassen recursion.
    
    def generate_strassen_terms(size):
        """
        Generates rank-1 terms (u, v, w) for size x size x size matrix multiplication.
        u, v, w are lists of length size*size.
        """
        if size == 2:
            return base_terms
        
        # Recursive case
        # Split size into size/2
        half = size // 2
        # Get terms for half size
        half_terms = generate_strassen_terms(half)
        
        # Define the 7 Strassen block combinations
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 10, 11
        # C blocks: 00, 01, 10, 11
        
        # M1 = (A00+A11)(B00+B11) -> C00+=, C11+=
        # M2 = (A10+A11)B00 -> C10+=, C11-=
        # M3 = A00(B10-B11) -> C01+=, C11+=
        # M4 = A11(B01-B10) -> C00+=, C10+=
        # M5 = (A00+A01)B11 -> C00-=, C01+=
        # M6 = (A10-A00)(B00+B10) -> C11+=
        # M7 = (A01-A11)(B01+B11) -> C00+=
        
        outer_defs = [
            # (A_coefs, B_coefs, C_coefs)
            # A_coefs: [A00, A01, A10, A11]
            # B_coefs: [B00, B01, B10, B11]
            # C_coefs: [C00, C01, C10, C11]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),
        ]
        
        new_terms = []
        for a_coefs, b_coefs, c_coefs in outer_defs:
            for u_h, v_h, w_h in half_terms:
                # u_h is length half*half. It corresponds to the entries of the sub-matrix A_sub.
                # We need to map u_h to the full size*size vector.
                # The sub-matrix A_sub is formed by combining blocks.
                # Wait, the recursion works like this:
                # The "A" in the recursive call is the combined block matrix A_q.
                # The entries of A_q are linear combinations of the entries of the original A.
                # If u_h has a coefficient $c$ for entry $(r,c)$ of A_q,
                # and A_q[r,c] = sum_{b} a_{b,r,c} A_b[r,c] (where A_b is a block),
                # then the full u vector gets contribution $c * a_{b,r,c}$ for the corresponding global index.
                
                # Let's define the mapping from block (bi, bj) and local (li, lj) to global index.
                # Global index for A: i*size + j.
                # Block bi (0,1) corresponds to row offset bi*half.
                # Block bj (0,1) corresponds to col offset bj*half.
                
                # For a given outer term (a_coefs, ...):
                # A_q is a half x half matrix.
                # A_q[li, lj] = sum_{bi, bj} a_{bi,bj} * A[bi*half+li, bj*half+lj]
                # Wait, a_coefs is a vector of 4 elements [A00, A01, A10, A11].
                # This implies A_q[li, lj] = a_0 * A[li, lj] + a_1 * A[li, lj+half] + a_2 * A[li+half, lj] + a_3 * A[li+half, lj+half]
                
                full_u = [0] * (size * size)
                for li in range(half):
                    for lj in range(half):
                        local_idx = li * half + lj
                        coeff_in_ah = u_h[local_idx]
                        if coeff_in_ah == 0: continue
                        
                        # Add contributions from each block
                        # Block 00: row li, col lj -> global li*size + lj
                        # Block 01: row li, col lj+half -> global li*size + (lj+half)
                        # Block 10: row li+half, col lj -> global (li+half)*size + lj
                        # Block 11: row li+half, col lj+half -> global (li+half)*size + (lj+half)
                        
                        full_u[li * size + lj] += coeff_in_ah * a_coefs[0]
                        full_u[li * size + (lj + half)] += coeff_in_ah * a_coefs[1]
                        full_u[(li + half) * size + lj] += coeff_in_ah * a_coefs[2]
                        full_u[(li + half) * size + (lj + half)] += coeff_in_ah * a_coefs[3]
                        
                full_v = [0] * (size * size)
                for li in range(half):
                    for lj in range(half):
                        local_idx = li * half + lj
                        coeff_in_bh = v_h[local_idx]
                        if coeff_in_bh == 0: continue
                        
                        full_v[li * size + lj] += coeff_in_bh * b_coefs[0]
                        full_v[li * size + (lj + half)] += coeff_in_bh * b_coefs[1]
                        full_v[(li + half) * size + lj] += coeff_in_bh * b_coefs[2]
                        full_v[(li + half) * size + (lj + half)] += coeff_in_bh * b_coefs[3]
                        
                full_w = [0] * (size * size)
                for li in range(half):
                    for lj in range(half):
                        local_idx = li * half + lj
                        coeff_in_ch = w_h[local_idx]
                        if coeff_in_ch == 0: continue
                        
                        full_w[li * size + lj] += coeff_in_ch * c_coefs[0]
                        full_w[li * size + (lj + half)] += coeff_in_ch * c_coefs[1]
                        full_w[(li + half) * size + lj] += coeff_in_ch * c_coefs[2]
                        full_w[(li + half) * size + (lj + half)] += coeff_in_ch * c_coefs[3]
                        
                new_terms.append((full_u, full_v, full_w))
                
        return new_terms

    # Generate 4x4x4 terms
    terms_4x4 = generate_strassen_terms(4)
    
    # We need 4x4x8.
    # C = A * B.
    # B is 4x8. Split B into B_left (4x4) and B_right (4x4).
    # C_left = A * B_left. C_right = A * B_right.
    # C is 4x8.
    # W vector is length p*n = 8*4 = 32.
    # Index in W: k*n + i. k is col of C (0..7), i is row of C (0..3).
    # For C_left (cols 0-3), k is 0..3.
    # For C_right (cols 4-7), k is 4..7.
    
    # U is 4x4 = 16 elements.
    # V is 4x8 = 32 elements.
    # W is 8x4 = 32 elements.
    
    U = [0] * (16 * 98) # Flatten later? No, keep as list of lists.
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    col_idx = 0
    # First half: B_left
    for u, v, w in terms_4x4:
        # u is length 16.
        # v is length 16 (for B_left 4x4). We need to map to V (length 32).
        # B_left corresponds to cols 0-3 of B.
        # B[i, j] is at index i*8 + j.
        # B_left[i, j] is at index i*8 + j.
        # So v_global[i*4+j] = v[i*4+j] for j in 0..3.
        # v_global[i*4+j+4] = 0.
        
        # w is length 16 (for C_left 4x4). We need to map to W (length 32).
        # C_left[i, k] is at W index k*4 + i.
        # w[i*4+k] corresponds to C_left[i,k].
        # So W_global[k*4+i] = w[i*4+k].
        
        for r in range(16):
            U[r][col_idx] += u[r]
            
            # Map v
            i = r // 4
            j = r % 4
            V[i*8 + j][col_idx] += v[r]
            
            # Map w
            i = r // 4
            k = r % 4
            W[k*4 + i][col_idx] += w[r]
            
        col_idx += 1
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Second half: B_right
    for u, v, w in terms_4x4:
        # u is same.
        # v is length 16 (for B_right 4x4).
        # B_right[i, j] is at index i*8 + (j+4).
        # v_global[i*4+j] = v[i*4+j].
        # Map to V[i*8 + j + 4].
        
        # w is length 16 (for C_right 4x4).
        # C_right[i, k] is at W index (k+4)*4 + i.
        # w[i*4+k] corresponds to C_right[i,k].
        # Map to W[(k+4)*4 + i].
        
        for r in range(16):
            U[r][col_idx] += u[r]
            
            # Map v
            i = r // 4
            j = r % 4
            V[i*8 + j + 4][col_idx] += v[r]
            
            # Map w
            i = r // 4
            k = r % 4
            W[(k+4)*4 + i][col_idx] += w[r]
            
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
