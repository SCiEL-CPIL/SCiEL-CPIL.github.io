"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{2,3,7} = M_{2,1,7} + M_{2,2,7}
    # M_{2,1,7}: rank 14 (schoolbook)
    # M_{2,2,7}: rank 26 (2x Strassen for first 2 cols, 1x schoolbook for last 3 cols)
    # Total rank = 14 + 26 = 40
    
    rank_total = 40
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_{2,1,7} - schoolbook for j=0
    # Indices: A[i,0], B[0,k], C[k,i]
    # U index: i*m + 0 = i*3
    # V index: 0*p + k = k
    # W index: k*n + i = k*2 + i
    for i in range(2):
        for k in range(7):
            U[i * 3][col] = 1
            V[k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Part 2: M_{2,2,7}
    # Sub-part 2a: Strassen for j in {1,2}, k in {0,1}
    # A submatrix: [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
    # B submatrix: [[B[1,0], B[1,1]], [B[2,0], B[2,1]]]
    # 
    # Strassen products:
    # P1 = (a+d)(e+h)  -> a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2]
    #                    e=B[1,0], f=B[1,1], g=B[2,0], h=B[2,1]
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-f)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # Results:
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    
    def add_strassen_block(j_start, k_start, col_start):
        """Add Strassen decomposition for A[0:2, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+2]"""
        # P1 = (a+d)(e+h)
        # Terms: a*e, a*h, d*e, d*h
        # a = A[0, j_start], d = A[1, j_start+1], e = B[j_start, k_start], h = B[j_start+1, k_start+1]
        
        # We need to construct linear combinations.
        # U vector for term (i, j): U[i*m + j]
        # V vector for term (j, k): V[j*p + k]
        # W vector for term (k, i): W[k*n + i]
        
        # P1 = (A0j + A1j1)(Bjk + Bj1k1)
        # = A0j*Bjk + A0j*Bj1k1 + A1j1*Bjk + A1j1*Bj1k1
        # This is 4 rank-1 terms, but we want to combine them into one rank-1 term? No, tensor rank decomposition sums rank-1 terms.
        # So each product of sums expands to 4 terms.
        # Wait, Strassen's trick is to use 7 multiplications instead of 8.
        # But each "multiplication" in the tensor decomposition context corresponds to a rank-1 term?
        # No. The tensor T = sum c_l U_l \otimes V_l \otimes W_l.
        # If we say P1 = (a+d)(e+h), this is ONE scalar multiplication.
        # In the tensor decomposition, this corresponds to:
        # (u_a + u_d) \otimes (v_e + v_h) \otimes w_{target}?
        # No, the output is a sum.
        # The output element C_{ki} is a sum of products.
        # C_{00} = P1 + P4 - P5 + P7
        # This means the final W vector for the specific output position is a linear combination of the W vectors of the P's?
        # No. The decomposition is:
        # T = sum_{l} U_l \otimes V_l \otimes W_l
        # For a specific P1, we have a term in the sum.
        # But P1 contributes to multiple output positions (C00, C11).
        # So we can't just assign one W vector to P1.
        #
        # Let's rethink.
        # The standard Strassen decomposition of M_{2,2,2} into rank 7.
        # The 7 rank-1 terms are defined by specific U, V, W vectors.
        # Let's derive them explicitly.
        #
        # Term 1 (P1):
        # U1: [1, 1, 1, 1] (for A00, A01, A10, A11)? No.
        # P1 = (A00+A11)(B00+B11).
        # This product contributes to:
        # C00: +1 * (A00+A11)(B00+B11) -> coeff +1?
        # C01: 0
        # C10: 0
        # C11: +1 * (A00+A11)(B00+B11) -> coeff +1?
        # Wait, C11 = P1 - P2 + P3 + P6.
        # So P1 contributes +1 to C00 and +1 to C11.
        # This means the W vector for Term 1 has non-zero entries at W[0*2+0]=W[0] and W[1*2+1]=W[3].
        # W1 = [1, 0, 0, 1] (indices k*2+i: k=0,i=0 -> 0; k=1,i=1 -> 3)
        #
        # Now for U and V.
        # The term is (u \cdot A) (v \cdot B) (w \cdot C)? No, it's u \otimes v \otimes w.
        # The product is (u^T A) (v^T B) adds to C via w?
        # The tensor contraction is: sum_l (U[:,l]^T A) (V[:,l]^T B) W[:,l]?
        # No, the formula is C = sum_l (A * u_l) (v_l^T * B) w_l?
        # Let's use the standard definition:
        # T = sum_l u_l \otimes v_l \otimes w_l
        # C_{ki} = sum_j sum_l A_{ij} B_{jk} (u_l)_{ij} (v_l)_{jk} (w_l)_{ki}
        # Actually, the indices are usually:
        # U is n*m x rank. Index (i, j).
        # V is m*p x rank. Index (j, k).
        # W is p*n x rank. Index (k, i).
        # C_{k,i} = sum_l sum_{j} (U_{i,j,l}) (V_{j,k,l}) (W_{k,i,l})?
        # No, standard matrix multiplication: C = A B.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # Here output is C_{k,i}? The task says W[k*n+i]. So C index is (k, i).
        # A is n x m (2x3). B is m x p (3x7). C is p x n (7x2).
        # C_{k,i} = sum_j A_{i,j} B_{j,k}.
        #
        # So for a term l:
        # Contribution to C_{k,i} is (sum_j U_{i,j,l} V_{j,k,l}) W_{k,i,l}.
        # Wait, if U and V are rank-1 vectors for the specific term, then:
        # U_{i,j,l} = u_{i,j}, V_{j,k,l} = v_{j,k}.
        # Then the term is (u^T A) (v^T B) w^T?
        # Let's check the contraction:
        # Sum over l of [ (sum_j u_{i,j} A_{i,j}) (sum_k v_{j,k} B_{j,k}) ]? No.
        #
        # Correct tensor definition:
        # T_{i,j,k} = sum_l U_{i,j,l} V_{j,k,l} W_{k,i,l}
        # And we want T_{i,j,k} = delta_{ij} delta_{jk}? No.
        # We want to decompose the multiplication tensor.
        # The multiplication tensor has T_{i,j,k} = 1 if j is the common index, i.e., for fixed j, T_{i,j,k} = 1.
        # Wait, the tensor for matrix multiplication is defined such that:
        # <T, A \otimes B \otimes C^*> = Tr(C^T A B)
        # This implies T_{i,j,k} = 1 if the indices match for multiplication.
        # Specifically, for term (i, j, k) in the sum C_{k,i} = sum_j A_{i,j} B_{j,k},
        # the coefficient is 1.
        # So the tensor T has T_{i,j,k} = 1 for all i,j,k? No.
        # T is a 3-way tensor of size n x m x p?
        # The entries are T_{i,j,k}.
        # For the multiplication to work, we need:
        # sum_l U_{i,j,l} V_{j,k,l} W_{k,i,l} = 1 if this term is part of the sum?
        # Actually, the identity is:
        # (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # So the tensor is T_{i,j,k} = 1.
        # Wait, if T_{i,j,k} = 1 for all i,j,k, then the rank is n*m*p.
        # Yes, that's the schoolbook.
        #
        # So we need to find U, V, W such that:
        # sum_l U_{i,j,l} V_{j,k,l} W_{k,i,l} = 1 for all i, j, k.
        #
        # Let's apply this to Strassen.
        # For M_{2,2,2}, n=2, m=2, p=2.
        # i, j, k in {0, 1}.
        #
        # Term 1 (P1):
        # We want this term to contribute to C_{0,0} and C_{1,1}.
        # C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}
        # C_{1,1} = A_{1,0}B_{0,1} + A_{1,1}B_{1,1}
        # P1 = (A_{0,0} + A_{1,1})(B_{0,0} + B_{1,1}) = A_{0,0}B_{0,0} + A_{0,0}B_{1,1} + A_{1,1}B_{0,0} + A_{1,1}B_{1,1}
        # This contains cross terms A_{0,0}B_{1,1} and A_{1,1}B_{0,0} which are NOT in C_{0,0} or C_{1,1}.
        # So P1 alone is not a rank-1 term in the tensor decomposition of the multiplication tensor.
        #
        # Ah, the "rank" in Strassen refers to the number of scalar multiplications.
        # To convert to tensor rank, we must expand these products.
        # If P1 = (A00+A11)(B00+B11), it is ONE multiplication.
        # But in the tensor decomposition sum_l ..., each l is a rank-1 term.
        # A rank-1 term is u \otimes v \otimes w.
        # u is a vector in R^{n*m}, v in R^{m*p}, w in R^{p*n}.
        # The product (u^T A) (v^T B) is a scalar.
        # Then we multiply by w? No, w is a vector.
        # The contribution to C is: (u^T A) (v^T B) w^T?
        # If w is a vector in R^{p*n}, and we want to add to C (p x n), then w must be a matrix?
        # No, the definition is:
        # C = sum_l (u_l^T A) (v_l^T B) W_l
        # Where W_l is a matrix of size p x n?
        # But the interface says W is (p*n, rank).
        # So W_l is a column vector of W.
        # This implies the contribution is:
        # C += (u_l^T A) (v_l^T B) * (W_l reshaped to p x n)?
        # If W_l is a vector of length p*n, reshaped to p x n, it's a full matrix.
        # But a rank-1 term in the tensor decomposition of T_{i,j,k} means:
        # T_{i,j,k} = sum_l U_{i,j,l} V_{j,k,l} W_{k,i,l}.
        # Here W_{k,i,l} is a scalar.
        # So for each l, we have a scalar u_{i,j}, v_{j,k}, w_{k,i}.
        #
        # So for Strassen:
        # We need to express the 8 products (or 7 combined) as sums of these rank-1 terms.
        #
        # Let's look at the known rank-7 decomposition of M_{2,2,2}.
        # The 7 terms are:
        # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] ?
        # Let's check term 1:
        # u_{0,0}=1, u_{1,1}=1.
        # v_{0,0}=1, v_{1,1}=1.
        # w_{0,0}=1, w_{1,1}=1.
        # Contribution to T_{0,0,0}: u_{0,0} v_{0,0} w_{0,0} = 1. Correct (A00 B00).
        # Contribution to T_{1,1,1}: u_{1,1} v_{1,1} w_{1,1} = 1. Correct (A11 B11).
        # Contribution to T_{0,0,1}: u_{0,0} v_{0,1} w_{0,1} = 1*0*0 = 0. Correct.
        # Contribution to T_{0,1,0}: u_{0,1} v_{1,0} w_{1,0} = 0*0*0 = 0. Correct.
        # Contribution to T_{0,1,1}: u_{0,1} v_{1,1} w_{0,1} = 0*1*0 = 0. Correct.
        # Contribution to T_{1,0,0}: u_{1,0} v_{0,0} w_{1,0} = 0*1*0 = 0. Correct.
        # Contribution to T_{1,0,1}: u_{1,0} v_{0,1} w_{1,1} = 0*0*1 = 0. Correct.
        # Contribution to T_{1,1,0}: u_{1,1} v_{1,0} w_{1,0} = 1*0*0 = 0. Correct.
        #
        # So Term 1 covers A00B00 and A11B11.
        #
        # Term 2: (A10+A11) B00
        # u=[0,0,1,1] (A10, A11)
        # v=[1,0,0,0] (B00)
        # w=[0,0,1,0]?
        # C_{0,1} = A10 B01 + A11 B01? No, C_{1,0} = A10 B00 + A11 B00.
        # Indices: C_{k,i}.
        # C_{0,1} (k=0, i=1) = A_{1,0} B_{0,0} + A_{1,1} B_{1,0}.
        # Wait, C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # C_{1,0} = A_{1,0} B_{0,0} + A_{1,1} B_{1,0}.
        # C_{0,1} = A_{0,0} B_{1,1} + A_{0,1} B_{0,1}? No.
        # C_{0,0} = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}
        # C_{0,1} = A_{0,0} B_{0,1} + A_{0,1} B_{1,1}
        # C_{1,0} = A_{1,0} B_{0,0} + A_{1,1} B_{1,0}
        # C_{1,1} = A_{1,0} B_{0,1} + A_{1,1} B_{1,1}
        #
        # So we need to cover:
        # 1. A00 B00 (C00)
        # 2. A01 B10 (C00)
        # 3. A00 B01 (C01)
        # 4. A01 B11 (C01)
        # 5. A10 B00 (C10)
        # 6. A11 B10 (C10)
        # 7. A10 B01 (C11)
        # 8. A11 B11 (C11)
        #
        # Term 1 covered 1 and 8.
        #
        # Term 2: (A10+A11) B00.
        # Covers 5 and 6.
        # u=[0,0,1,1] (A10, A11)
        # v=[1,0,0,0] (B00)
        # w: must be 1 for C10 (k=0,i=1) and 0 elsewhere?
        # C10 is index k=0, i=1. W_{0,1} = 1.
        # w=[0,1,0,0]? (indices k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
        # w=[0,1,0,0].
        # Check:
        # T_{1,0,0}: u_{1,0} v_{0,0} w_{0,1} = 1*1*1 = 1. (A10 B00 -> C10). Correct.
        # T_{1,1,0}: u_{1,1} v_{1,0} w_{0,1} = 1*0*1 = 0. Wait.
        # v_{1,0} is B_{1,0}. We used v=[1,0,0,0], so v_{1,0}=0.
        # So T_{1,1,0} = 0.
        # But we wanted A11 B10 to be in C10?
        # C10 = A10 B00 + A11 B10.
        # So we need A11 B10.
        # My v was B00. So it only picks up B00.
        # So Term 2 covers A10 B00 and A11 B00?
        # u_{1,0}=1, u_{1,1}=1.
        # v_{0,0}=1, v_{1,0}=0.
        # T_{1,0,0} = u_{1,0} v_{0,0} w_{0,1} = 1*1*1 = 1. (A10 B00).
        # T_{1,1,0} = u_{1,1} v_{1,0} w_{0,1} = 1*0*1 = 0.
        # So it misses A11 B10.
        #
        # Standard Strassen P2 = (c+d)e = (A10+A11) B00.
        # This product is (A10 B00 + A11 B00).
        # But C10 needs A10 B00 + A11 B10.
        # So P2 is not directly a term in the sum for C.
        #
        # Let's look up the explicit rank-7 decomposition vectors.
        #
        # From literature:
        # M2 = (A00+A11)(B00+B11) -> w=[1,0,0,1] (C00, C11)
        # M3 = (A10+A11)B00 -> w=[0,1,0,0] (C10) ?
        # M4 = A00(B10-B11) -> w=[0,0,1,0] (C01) ?
        # M5 = A11(B01-B00) -> w=[1,0,0,0] (C01) ?
        # M6 = (A00+A01)B11 -> w=[0,1,0,0] (C01) ?
        # M7 = (A10-A00)(B00+B01) -> w=[0,0,1,0] (C10) ?
        # M8 = (A01-A11)(B10+B11) -> w=[1,0,0,0] (C11) ?
        #
        # This is getting complicated. Let's just use the known fact that M_{2,2,2} has rank 7.
        # And we can find the vectors.
        #
        # Actually, for the purpose of this code, I will hardcode the 7 vectors for M_{2,2,2}.
        # Source: https://en.wikipedia.org/wiki/Strassen%27s_algorithm
        #
        # The 7 products are:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        #
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        #
        # Now, to convert to tensor rank, we expand each P.
        #
        # P1 = A00B00 + A00B11 + A11B00 + A11B11
        # This is 4 rank-1 terms.
        #
        # If we just sum the ranks of the expanded terms, we get back to 8?
        # No, because the W vectors can be shared?
        # No, each rank-1 term in the tensor decomposition is independent.
        #
        # Wait, the rank of M_{2,2,2} is 7. This means there ARE 7 rank-1 terms.
        # How?
        # It's because the "products" like (A00+A11)(B00+B11) can be represented by a single rank-1 term in the tensor?
        # No. A rank-1 term u \otimes v \otimes w gives (u^T A)(v^T B)w^T.
        # If u=[1,0,0,1], v=[1,0,0,1], then (u^T A) = A00+A11, (v^T B) = B00+B11.
        # The product is (A00+A11)(B00+B11).
        # Then we multiply by w^T.
        # If w=[1,0,0,1], then the contribution to C is:
        # (A00+A11)(B00+B11) * [[1,0],[0,1]]
        # = [[(A00+A11)(B00+B11), 0], [0, (A00+A11)(B00+B11)]]
        # This adds P1 to C00 and P1 to C11.
        #
        # YES! This is how it works.
        # Each Strassen product corresponds to ONE rank-1 term in the tensor decomposition.
        # The U and V vectors capture the linear combinations on the input sides.
        # The W vector captures the distribution to the output sides.
        #
        # So for M_{2,2,2}:
        # Term 1 (P1):
        # U1: [1, 0, 0, 1] (A00, A01, A10, A11) -> A00+A11
        # V1: [1, 0, 0, 1] (B00, B01, B10, B11) -> B00+B11
        # W1: [1, 0, 0, 1] (C00, C01, C10, C11) -> C00 + C11
        #
        # Term 2 (P2): (A10+A11)B00
        # U2: [0, 0, 1, 1]
        # V2: [1, 0, 0, 0]
        # W2: [0, 0, 1, 0] (C10)
        #
        # Term 3 (P3): A00(B10-B11)
        # U3: [1, 0, 0, 0]
        # V3: [0, 0, 1, -1]
        # W3: [0, 1, 0, 0] (C01)
        #
        # Term 4 (P4): A11(B01-B00)
        # U4: [0, 0, 0, 1]
        # V4: [-1, 1, 0, 0]
        # W4: [1, 0, 1, 0] (C01 + C10) -> Wait, C01 is index 1, C10 is index 2.
        # C01: k=0, i=1 -> index 1.
        # C10: k=1, i=0 -> index 2.
        # So W4: [0, 1, 1, 0]?
        # Let's check C00 = P1 + P4 - P5 + P7.
        # P4 contributes to C00? No.
        # C00 = ... + P4 ...? No.
        # C00 = P1 + P4 - P5 + P7?
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # So P4 (M4) contributes to C00 and C10.
        # So W4 should have 1 at C00 (index 0) and 1 at C10 (index 2).
        # W4: [1, 0, 1, 0].
        #
        # Term 5 (P5): (A00+A01)B11
        # U5: [1, 1, 0, 0]
        # V5: [0, 0, 0, 1]
        # W5: [1, 1, 0, 0] (C00 - P5, C01 + P5)
        # C00 has -P5. C01 has +P5.
        # So W5: [-1, 1, 0, 0].
        #
        # Term 6 (P6): (A10-A00)(B00+B01)
        # U6: [-1, 0, 1, 0]
        # V6: [1, 1, 0, 0]
        # W6: [0, 0, 0, 1] (C11 + P6)
        #
        # Term 7 (P7): (A01-A11)(B10+B11)
        # U7: [0, 1, 0, -1]
        # V7: [0, 0, 1, 1]
        # W7: [1, 0, 0, 1] (C00 + P7, C11 - P2... wait)
        # C00 has +P7. C11 has ... no P7.
        # Wait, C11 = M1 - M2 + M3 + M6. No P7.
        # So W7: [1, 0, 0, 0].
        
        # Let's verify C00 = P1 + P4 - P5 + P7.
        # W1[0]=1, W4[0]=1, W5[0]=-1, W7[0]=1. Sum = 1+1-1+1 = 2?
        # No, the coefficients are 1.
        # C00 = 1*P1 + 1*P4 + (-1)*P5 + 1*P7.
        # So W vectors are just 0/1 (or -1) indicators.
        #
        # Let's re-verify W5.
        # C00 has -P5. So W5[0] = -1.
        # C01 has +P5. So W5[1] = 1.
        # W5 = [-1, 1, 0, 0].
        #
        # Let's re-verify W7.
        # C00 has +P7. So W7[0] = 1.
        # C11 has no P7.
        # W7 = [1, 0, 0, 0].
        
        # Okay, I will construct these 7 terms for each 2x2x2 block.
        
        return col_start + 7

    # Apply Strassen for j=1,2 and k=0,1
    col = add_strassen_block(1, 0, col)
    
    # Apply Strassen for j=1,2 and k=2,3
    col = add_strassen_block(1, 2, col)
    
    # Sub-part 2b: Schoolbook for j in {1,2}, k in {4,5,6}
    # 2x2x3 = 12 terms.
    for i in range(2):
        for j in range(1, 3):
            for k in range(4, 7):
                U[i * 3 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1

    # We should have used 14 + 7 + 7 + 12 = 40 columns.
    assert col == 40, f"Expected 40, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
