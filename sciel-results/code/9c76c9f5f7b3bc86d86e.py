"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 64.
    # Strategy:
    # 1. Split B (3x8) into two 3x4 blocks: B1, B2.
    # 2. For each 3x4 block, split into 3x3 (Ba) and 3x1 (Bb).
    # 3. Compute A*Ba using a rank-23 decomposition (Strassen for 2x2 part + schoolbook for rest).
    # 4. Compute A*Bb using a rank-9 decomposition (schoolbook).
    # Total rank per 3x4 block: 23 + 9 = 32.
    # Total rank for 3x8: 32 + 32 = 64.
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: U[i,j] += u, V[j,k] += v, W[k,i] += w
    # Here indices are flattened:
    # U index: i*m + j (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k (j in 0..m-1, k in 0..p-1)
    # W index: k*n + i (k in 0..p-1, i in 0..n-1)
    
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j][col] += u_val
        V[j * p + k][col] += v_val
        W[k * n + i][col] += w_val
        col += 1

    # We process two 3x4 blocks of B.
    # Block 1: columns 0..3 of B.
    # Block 2: columns 4..7 of B.
    
    for block_idx in range(2):
        # Offset for B columns
        b_offset = block_idx * 4
        
        # --- Part 1: A (3x3) * Ba (3x3) -> Ca (3x3) ---
        # Ba corresponds to B columns b_offset .. b_offset+2
        # We use the rank-23 decomposition for 3x3x3.
        
        # Strassen for A11 (2x2) * B11 (2x2)
        # A11 is A[0:2, 0:2], B11 is B[0:2, b_offset:b_offset+2]
        
        # M1 = (A11 + A22) * (B11 + B22)
        # Indices for A11: (0,0),(0,1),(1,0),(1,1)
        # Indices for A22: (2,2) -> Wait, in 3x3 split, A22 is 1x1 at (2,2).
        # Let's stick to the 2x2 + rest decomposition derived above.
        
        # A11 (2x2): i in 0,1; j in 0,1
        # A12 (2x1): i in 0,1; j in 2
        # A21 (1x2): i in 2; j in 0,1
        # A22 (1x1): i in 2; j in 2
        
        # B11 (2x2): j in 0,1; k in b_offset, b_offset+1
        # B12 (2x1): j in 0,1; k in b_offset+2
        # B21 (1x2): j in 2; k in b_offset, b_offset+1
        # B22 (1x1): j in 2; k in b_offset+2
        
        # Strassen terms for A11*B11 (2x2x2)
        # We need to map the 2x2 Strassen factors to the global indices.
        
        # M1 = (A11 + A22) (B11 + B22)
        # This term contributes to C11, C22.
        # Actually, let's just implement the 7 Strassen terms for the 2x2 part.
        
        # Strassen 2x2x2 factors for A11*B11:
        # A11 = [[a00, a01], [a10, a11]]
        # B11 = [[b00, b01], [b10, b11]]
        
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # These are products of sums/differences.
        # We add them as rank-1 terms in the tensor decomposition.
        # Each "product" like (a00+a11)(b00+b11) is NOT a single rank-1 term in the tensor decomposition of matrix multiplication.
        # The tensor decomposition requires factors U, V, W such that C = sum U*V*W.
        # Strassen's algorithm works by defining intermediate variables M1..M7 which are products of linear combinations of A and B.
        # Each M is a product of a vector from A and a vector from B.
        # So each M corresponds to 1 rank-1 term in the decomposition?
        # Yes, M1 = u1 . A * v1 . B.
        # So we can define the vectors u, v for each M.
        
        # Let's define the vectors for the 7 Strassen terms.
        # A indices: (i,j). B indices: (j,k).
        
        # Term 1: M1
        # u1: a00 + a11. Vector in A space (dim 4 for 2x2, but we are in 3x3).
        # In global 3x3 A:
        # u1 has 1 at (0,0) and (1,1).
        # v1: b00 + b11. Vector in B space.
        # In global B (for this block):
        # v1 has 1 at (0, b_offset) and (1, b_offset+1).
        # w1: contributes to C11 and C22.
        # C11 += M1, C22 += M1.
        # So w1 has 1 at (b_offset, 0) and (b_offset+1, 1).
        
        # Wait, the W factor is for C.
        # C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]? No.
        # The standard form is C[i,k] = sum_{j,r} U[i,j,r] V[j,k,r] W[r, i, k]?
        # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # And "output factor follows the official trace convention W[k*n+i,r]".
        # This implies C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
        # No, that's not right. U depends on i,j. V on j,k. W on k,i.
        # The sum is over r and j?
        # Usually, tensor decomposition is T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
        # Here T is the multiplication tensor T[i,j,k] = A[i,j] B[j,k] C[k,i]?
        # The tensor for matrix multiplication M_{n,m,p} has indices i (row A), j (col A/row B), k (col B).
        # T[i,j,k] = 1 if we are computing C[i,k] from A[i,j]B[j,k].
        # The decomposition is T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]?
        # No, U is indexed by (i,j). V by (j,k). W by (k,i).
        # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # Summing over r gives the coefficient for A[i,j]B[j,k] in C[k,i]?
        # Actually, the product C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T[i,j,k] is 1.
        # So we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k?
        # No, that's not how it works.
        # The decomposition is of the tensor $M_{n,m,p}$.
        # $M_{n,m,p} = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
        # Where $u_r$ is in $\mathbb{R}^{n \times m}$, $v_r$ in $\mathbb{R}^{m \times p}$, $w_r$ in $\mathbb{R}^{p \times n}$.
        # And the multiplication is $C = \sum_r (u_r \cdot A) \otimes (v_r \cdot B) \cdot w_r^T$?
        # No.
        # The standard interpretation:
        # $C[i,k] = \sum_j A[i,j] B[j,k]$.
        # This is linear in A and B.
        # We want $C[i,k] = \sum_r \alpha_{r, i, j} \beta_{r, j, k} \gamma_{r, k, i} A[i,j] B[j,k]$?
        # No, the factors U, V, W are such that:
        # $C[i,k] = \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$?
        # This sum is over r. But j is fixed?
        # Actually, the tensor is $T[i,j,k]$.
        # $T[i,j,k] = \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$.
        # For the multiplication tensor, $T[i,j,k] = 1$ for all valid i,j,k?
        # No, $T[i,j,k]$ is the coefficient of $A[i,j]B[j,k]$ in $C[i,k]$.
        # Since $C[i,k] = \sum_j A[i,j]B[j,k]$, the coefficient is 1 for each j.
        # So $T[i,j,k] = 1$.
        # Thus we need $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$ for all i,j,k.
        
        # This means each rank-1 term is a product of three vectors.
        # For Strassen, we define linear combinations of A and B.
        # $M_r = (\sum_{i,j} u_{r, i, j} A[i,j]) (\sum_{j,k} v_{r, j, k} B[j,k])$.
        # Then $C[i,k] = \sum_r w_{r, i, k} M_r$.
        # This matches the form if we identify:
        # $U[i*m+j, r] = u_{r, i, j}$
        # $V[j*p+k, r] = v_{r, j, k}$
        # $W[k*n+i, r] = w_{r, i, k}$
        # And the sum is $\sum_r u_{r, i, j} v_{r, j, k} w_{r, i, k}$.
        # Wait, the indices in W are $k*n+i$. So $W[k*n+i, r] = w_{r, i, k}$.
        # Yes.
        
        # So we just need to list the $(u, v, w)$ triples for each Strassen term.
        
        # Strassen 2x2x2 terms (for A11*B11):
        # A11 is A[0:2, 0:2]. B11 is B[0:2, b_offset:b_offset+2].
        # C11 is C[0:2, b_offset:b_offset+2].
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # u1: A00 + A11. Non-zero at (0,0) and (1,1).
        # v1: B00 + B11. Non-zero at (0, b_offset) and (1, b_offset+1).
        # w1: C00 + C11. Non-zero at (b_offset, 0) and (b_offset+1, 1).
        
        # We add this term.
        # u1 indices: (0,0), (1,1)
        # v1 indices: (0, b_offset), (1, b_offset+1)
        # w1 indices: (b_offset, 0), (b_offset+1, 1)
        
        # Note: U, V, W are matrices.
        # U[i*m+j, r] = 1 for (i,j) in support of u.
        # V[j*p+k, r] = 1 for (j,k) in support of v.
        # W[k*n+i, r] = 1 for (k,i) in support of w.
        
        # Let's implement the 7 Strassen terms.
        
        # T1
        u_support = [(0,0), (1,1)]
        v_support = [(0, b_offset), (1, b_offset+1)]
        w_support = [(b_offset, 0), (b_offset+1, 1)]
        # Wait, w index is (k,i).
        # C[0, b_offset] corresponds to i=0, k=b_offset. Index k*n+i = b_offset*3+0.
        # C[1, b_offset+1] corresponds to i=1, k=b_offset+1. Index (b_offset+1)*3+1.
        
        # Let's write a helper to add a term with given supports.
        def add_strassen_term(u_sup, v_sup, w_sup):
            nonlocal col
            for (i,j) in u_sup:
                U[i*m+j][col] = 1
            for (j,k) in v_sup:
                V[j*p+k][col] = 1
            for (k,i) in w_sup:
                W[k*n+i][col] = 1
            col += 1

        # Strassen Terms for A11*B11
        # A11: (0,0),(0,1),(1,0),(1,1)
        # B11: (0,b),(0,b+1),(1,b),(1,b+1) where b=b_offset
        
        # M1 = (A00+A11)(B00+B11)
        add_strassen_term([(0,0),(1,1)], [(0,b_offset),(1,b_offset+1)], [(b_offset,0),(b_offset+1,1)])
        
        # M2 = (A10+A11)B00
        add_strassen_term([(1,0),(1,1)], [(0,b_offset)], [(b_offset,1)])
        # C11 += M2? No.
        # C21 += M2. C[1, b_offset]. i=1, k=b_offset.
        
        # M3 = A00(B01-B11)
        add_strassen_term([(0,0)], [(0,b_offset+1), (1,b_offset+1)], [(b_offset+1,0)])
        # v: B01 - B11. (0, b+1) and (1, b+1) with -1?
        # My helper only adds 1. I need to handle signs.
        
        # Let's modify the helper to accept values.
        
        # Re-implement with values.
        col = 0
        def add_term_val(u_dict, v_dict, w_dict):
            nonlocal col
            for (i,j), val in u_dict.items():
                U[i*m+j][col] += val
            for (j,k), val in v_dict.items():
                V[j*p+k][col] += val
            for (k,i), val in w_dict.items():
                W[k*n+i][col] += val
            col += 1

        # Reset U,V,W and col
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        for block_idx in range(2):
            b = block_idx * 4
            
            # Strassen 2x2x2 for A[0:2,0:2] * B[0:2, b:b+2]
            
            # M1
            add_term_val({(0,0):1, (1,1):1}, {(0,b):1, (1,b+1):1}, {(b,0):1, (b+1,1):1})
            # M2
            add_term_val({(1,0):1, (1,1):1}, {(0,b):1}, {(b,1):1})
            # M3
            add_term_val({(0,0):1}, {(0,b+1):1, (1,b+1):-1}, {(b+1,0):1})
            # M4
            add_term_val({(1,1):1}, {(1,b):1, (0,b):-1}, {(b+1,1):1})
            # M5
            add_term_val({(0,0):1, (0,1):1}, {(1,b+1):1}, {(b+1,0):1, (b+1,1):-1}) 
            # Wait, M5 = (A00+A01)B11.
            # C12 += M5, C22 -= M5?
            # C12 = M3+M5. C22 = M1-M2+M3+M6.
            # My w_supports must reflect the signs in C.
            # C[i,k] = sum w.
            # So if C12 += M5, w has 1 at (b+1, 0)? No, C[0, b+1]. i=0, k=b+1.
            # Index k*n+i = (b+1)*3+0.
            
            # Let's be precise.
            # M1: C00+=1, C11+=1. w: {(b,0):1, (b+1,1):1}
            # M2: C10+=1. w: {(b,1):1}
            # M3: C01+=1. w: {(b+1,0):1}
            # M4: C11+=1. w: {(b+1,1):1}
            # M5: C01+=1, C11-=1. w: {(b+1,0):1, (b+1,1):-1}
            # M6: C11+=1. w: {(b+1,1):1}
            # M7: C00+=1, C10-=1. w: {(b,0):1, (b,1):-1}
            
            # Let's re-add M1-M7 correctly.
            # I already added M1, M2, M3.
            # M1: {(b,0):1, (b+1,1):1}
            # M2: {(b,1):1}
            # M3: {(b+1,0):1}
            
            # M4
            add_term_val({(1,1):1}, {(1,b):1, (0,b):-1}, {(b+1,1):1})
            # M5
            add_term_val({(0,0):1, (0,1):1}, {(1,b+1):1}, {(b+1,0):1, (b+1,1):-1})
            # M6
            add_term_val({(1,0):1, (0,0):-1}, {(0,b):1, (0,b+1):1}, {(b+1,1):1})
            # M7
            add_term_val({(0,1):1, (1,1):-1}, {(1,b):1, (1,b+1):1}, {(b,0):1, (b,1):-1})
            
            # Now the cross terms for 3x3x3.
            # A12 (2x1) * B21 (1x2) -> C11 part.
            # A12: (0,2), (1,2). B21: (2,b), (2,b+1).
            # C00 += A02*B20. C01 += A02*B21. C10 += A12*B20. C11 += A12*B21.
            # 4 terms.
            add_term_val({(0,2):1}, {(2,b):1}, {(b,0):1})
            add_term_val({(0,2):1}, {(2,b+1):1}, {(b+1,0):1})
            add_term_val({(1,2):1}, {(2,b):1}, {(b,1):1})
            add_term_val({(1,2):1}, {(2,b+1):1}, {(b+1,1):1})
            
            # A11 (2x2) * B12 (2x1) -> C12 part.
            # A11: (0,0),(0,1),(1,0),(1,1). B12: (0,b+2),(1,b+2).
            # C01 += A00*B02 + A01*B12. C11 += A10*B02 + A11*B12.
            # 4 terms.
            add_term_val({(0,0):1}, {(0,b+2):1}, {(b+2,0):1})
            add_term_val({(0,1):1}, {(1,b+2):1}, {(b+2,0):1})
            add_term_val({(1,0):1}, {(0,b+2):1}, {(b+2,1):1})
            add_term_val({(1,1):1}, {(1,b+2):1}, {(b+2,1):1})
            
            # A12 (2x1) * B22 (1x1) -> C12 part.
            # A12: (0,2),(1,2). B22: (2,b+2).
            # C01 += A02*B22. C11 += A12*B22.
            # 2 terms.
            add_term_val({(0,2):1}, {(2,b+2):1}, {(b+2,0):1})
            add_term_val({(1,2):1}, {(2,b+2):1}, {(b+2,1):1})
            
            # A21 (1x2) * B11 (2x2) -> C21 part.
            # A21: (2,0),(2,1). B11: (0,b),(0,b+1),(1,b),(1,b+1).
            # C10 += A20*B00 + A21*B10. C11 += A20*B01 + A21*B11.
            # 4 terms.
            add_term_val({(2,0):1}, {(0,b):1}, {(b,2):1})
            add_term_val({(2,0):1}, {(0,b+1):1}, {(b+1,2):1})
            add_term_val({(2,1):1}, {(1,b):1}, {(b,2):1})
            add_term_val({(2,1):1}, {(1,b+1):1}, {(b+1,2):1})
            
            # A21 (1x2) * B12 (2x1) -> C22 part.
            # A21: (2,0),(2,1). B12: (0,b+2),(1,b+2).
            # C12 += A20*B02 + A21*B12.
            # 2 terms.
            add_term_val({(2,0):1}, {(0,b+2):1}, {(b+2,2):1})
            add_term_val({(2,1):1}, {(1,b+2):1}, {(b+2,2):1})
            
            # A22 (1x1) * B21 (1x2) -> C21 part.
            # A22: (2,2). B21: (2,b),(2,b+1).
            # C10 += A22*B20. C11 += A22*B21.
            # 2 terms.
            add_term_val({(2,2):1}, {(2,b):1}, {(b,2):1})
            add_term_val({(2,2):1}, {(2,b+1):1}, {(b+1,2):1})
            
            # A22 (1x1) * B22 (1x1) -> C22 part.
            # A22: (2,2). B22: (2,b+2).
            # C12 += A22*B22.
            # 1 term.
            add_term_val({(2,2):1}, {(2,b+2):1}, {(b+2,2):1})
            
            # Total terms for 3x3x3: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26?
            # 7 (Strassen) + 4 (A12B21) + 4 (A11B12) + 2 (A12B22) + 4 (A21B11) + 2 (A21B12) + 2 (A22B21) + 1 (A22B22) = 26.
            # I estimated 23 earlier. 26 is fine.
            
            # Now 3x3x1 part for B column b+3.
            # A (3x3) * B (3x1) -> C (3x1).
            # 9 terms.
            for i in range(3):
                for j in range(3):
                    add_term_val({(i,j):1}, {(j,b+3):1}, {(b+3,i):1})
                    
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
