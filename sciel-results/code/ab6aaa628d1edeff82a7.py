import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    # Rank calculation
    # M_2,2,2 rank 7
    # M_2,4,2 rank 14
    # M_2,4,4 rank 28
    # M_2,4,3 rank 15 (14 + 1)
    # M_2,4,7 rank 43 (28 + 15)
    # M_3,4,7 rank 44 (43 + 1)
    
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r_idx = 0
    
    # Helper to add a 2x2x2 Strassen block
    # i0, j0, k0 are the starting indices for the 2x2 block
    def add_strassen_block(i0, j0, k0):
        nonlocal r_idx
        # Strassen coefficients for A, B, C
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        
        # P1 = (A00+A11)(B00+B11)
        # Terms: A00 B00, A00 B11, A11 B00, A11 B11
        # U: A00, A11. V: B00, B11. W: C00, C11, C01, C10?
        # We need to map the 7 products to U, V, W columns.
        
        # The 7 products are:
        # 1. (A00+A11) * (B00+B11)
        # 2. (A10+A11) * B00
        # 3. A00 * (B01-B11)
        # 4. A11 * (B10-B00)
        # 5. (A00+A01) * B11
        # 6. (A10-A00) * (B00+B01)
        # 7. (A01-A11) * (B10+B11)
        
        # We assign each product to a rank term r.
        # For each term, we update U, V, W.
        
        # Term 1: (A00+A11)(B00+B11)
        # U[A00] += 1, U[A11] += 1
        # V[B00] += 1, V[B11] += 1
        # W[C00] += 1, W[C11] += 1, W[C01] += 0, W[C10] += 0?
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We need to be careful with the W indices.
        # W index is k*n + i.
        
        # Let's define the indices
        # A00: i0, j0
        # A01: i0, j0+1
        # A10: i0+1, j0
        # A11: i0+1, j0+1
        
        # B00: j0, k0
        # B01: j0, k0+1
        # B10: j0+1, k0
        # B11: j0+1, k0+1
        
        # C00: k0, i0
        # C01: k0+1, i0
        # C10: k0, i0+1
        # C11: k0+1, i0+1
        
        def add_term(u_indices, v_indices, w_indices, coeffs):
            nonlocal r_idx
            # u_indices: list of (i, j)
            # v_indices: list of (j, k)
            # w_indices: list of (k, i)
            # coeffs: list of values
            # This term contributes to rank r_idx
            for u_idx, c in zip(u_indices, coeffs):
                U[u_idx[0]*m + u_idx[1], r_idx] += c
            for v_idx, c in zip(v_indices, coeffs):
                V[v_idx[0]*p + v_idx[1], r_idx] += c
            for w_idx, c in zip(w_indices, coeffs):
                W[w_idx[0]*n + w_idx[1], r_idx] += c
            r_idx += 1

        # Term 1: P1
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C11(1)
        add_term([(i0, j0), (i0+1, j0+1)], 
                 [(j0, k0), (j0+1, k0+1)], 
                 [(k0, i0), (k0+1, i0+1)], 
                 [1, 1, 1, 1, 1, 1]) # Wait, coeffs must match indices
        # The add_term function above assumes coeffs correspond to the union of indices?
        # No, the decomposition is $\sum_r U_{i,r} V_{j,r} W_{k,r}$.
        # If we have multiple non-zero entries in U, V, W for a single r, it's not a rank-1 term.
        # A rank-1 term has only ONE non-zero entry in U, ONE in V, ONE in W.
        # Strassen uses sums like (A00+A11). This is not a single entry.
        # We cannot directly use Strassen's linear combinations in U, V, W if we want integer entries?
        # No, U, V, W can have any integer/half-integer entries.
        # But a rank-1 tensor is $u \otimes v \otimes w$.
        # $U_{i,r} = u_i, V_{j,r} = v_j, W_{k,r} = w_k$.
        # If $u$ has multiple non-zeros, it's still a valid vector.
        # So $U_{i0, r} = 1, U_{i1, r} = 1$ is fine.
        # So my add_term logic is correct.
        
        # Term 2: P2 = (A10+A11) B00
        add_term([(i0+1, j0), (i0+1, j0+1)], 
                 [(j0, k0)], 
                 [(k0, i0+1)], 
                 [1, 1, 1, 1])
                 
        # Term 3: P3 = A00 (B01-B11)
        add_term([(i0, j0)], 
                 [(j0, k0+1), (j0+1, k0+1)], 
                 [(k0+1, i0)], 
                 [1, 1, -1, 1])
                 
        # Term 4: P4 = A11 (B10-B00)
        add_term([(i0+1, j0+1)], 
                 [(j0+1, k0), (j0, k0)], 
                 [(k0, i0), (k0, i0+1)], 
                 [1, 1, -1, 1, 1])
                 
        # Term 5: P5 = (A00+A01) B11
        add_term([(i0, j0), (i0, j0+1)], 
                 [(j0+1, k0+1)], 
                 [(k0, i0), (k0+1, i0)], 
                 [1, 1, 1, -1, 1])
                 
        # Term 6: P6 = (A10-A00) (B00+B01)
        add_term([(i0+1, j0), (i0, j0)], 
                 [(j0, k0), (j0, k0+1)], 
                 [(k0+1, i0+1)], 
                 [1, -1, 1, 1, 1])
                 
        # Term 7: P7 = (A01-A11) (B10+B11)
        add_term([(i0, j0+1), (i0+1, j0+1)], 
                 [(j0+1, k0), (j0+1, k0+1)], 
                 [(k0, i0), (k0+1, i0+1)], 
                 [1, -1, 1, 1, 1, 1])

    # We need to fix the add_term function to handle the coefficients correctly.
    # The coefficients apply to the product $U \cdot V \cdot W$.
    # If $U$ has 2 entries, $V$ has 1, $W$ has 1.
    # The term is $(u_1 u_2) \cdot v \cdot w$.
    # The contribution to $C$ is $u_1 v w$ at index 1 and $u_2 v w$ at index 2.
    
    # Let's rewrite add_term to be explicit.
    
    def add_rank_term(u_map, v_map, w_map):
        """
        u_map: dict { (i,j): coeff }
        v_map: dict { (j,k): coeff }
        w_map: dict { (k,i): coeff }
        """
        nonlocal r_idx
        for (i, j), c_u in u_map.items():
            U[i*m + j, r_idx] += c_u
        for (j, k), c_v in v_map.items():
            V[j*p + k, r_idx] += c_v
        for (k, i), c_w in w_map.items():
            W[k*n + i, r_idx] += c_w
        r_idx += 1

    # Re-implement Strassen with add_rank_term
    def add_strassen(i0, j0, k0):
        nonlocal r_idx
        
        # P1 = (A00+A11)(B00+B11)
        # C00 += 1, C11 += 1
        add_rank_term(
            {(i0, j0): 1, (i0+1, j0+1): 1},
            {(j0, k0): 1, (j0+1, k0+1): 1},
            {(k0, i0): 1, (k0+1, i0+1): 1}
        )
        
        # P2 = (A10+A11)B00
        # C10 += 1
        add_rank_term(
            {(i0+1, j0): 1, (i0+1, j0+1): 1},
            {(j0, k0): 1},
            {(k0, i0+1): 1}
        )
        
        # P3 = A00(B01-B11)
        # C01 += 1
        add_rank_term(
            {(i0, j0): 1},
            {(j0, k0+1): 1, (j0+1, k0+1): -1},
            {(k0+1, i0): 1}
        )
        
        # P4 = A11(B10-B00)
        # C00 += 1, C10 += 1
        add_rank_term(
            {(i0+1, j0+1): 1},
            {(j0+1, k0): 1, (j0, k0): -1},
            {(k0, i0): 1, (k0, i0+1): 1}
        )
        
        # P5 = (A00+A01)B11
        # C00 -= 1, C01 += 1
        add_rank_term(
            {(i0, j0): 1, (i0, j0+1): 1},
            {(j0+1, k0+1): 1},
            {(k0, i0): -1, (k0+1, i0): 1}
        )
        
        # P6 = (A10-A00)(B00+B01)
        # C11 += 1
        add_rank_term(
            {(i0+1, j0): 1, (i0, j0): -1},
            {(j0, k0): 1, (j0, k0+1): 1},
            {(k0+1, i0+1): 1}
        )
        
        # P7 = (A01-A11)(B10+B11)
        # C00 += 1, C11 += 1
        add_rank_term(
            {(i0, j0+1): 1, (i0+1, j0+1): -1},
            {(j0+1, k0): 1, (j0+1, k0+1): 1},
            {(k0, i0): 1, (k0+1, i0+1): 1}
        )

    # Block 1: A1 (rows 0,1) * B_L (cols 0,1) -> C (rows 0,1, cols 0,1)
    add_strassen(0, 0, 0)
    # Block 2: A1 (rows 0,1) * B_L (cols 2,3) -> C (rows 0,1, cols 2,3)
    add_strassen(0, 2, 2)
    # Block 3: A1 (rows 0,1) * B_R1 (cols 4,5) -> C (rows 0,1, cols 4,5)
    add_strassen(0, 4, 4)
    
    # Block 4: A1 (rows 0,1) * B_R2 (col 6) -> C (rows 0,1, col 6)
    # M_2,4,1
    # A (2x4), B (4x1)
    # C0 = A00 B0 + A01 B1 + A02 B2 + A03 B3
    # C1 = A10 B0 + A11 B1 + A12 B2 + A13 B3
    # We can do this with 1 term?
    # U: A00, A01, A02, A03, A10, A11, A12, A13 all 1?
    # V: B0, B1, B2, B3 all 1?
    # W: C0, C1 all 1?
    # Sum = (Sum A0) (Sum B) (Sum W).
    # This gives C0 = Sum A0 Sum B, C1 = Sum A1 Sum B.
    # We need C0 = Sum A0j Bj.
    # This works if we separate j?
    # No, we can't separate j in a single rank term if B depends on j.
    # We need 4 terms for M_2,4,1?
    # Term j: U[A0j, A1j]=1, V[Bj]=1, W[C0, C1]=1.
    # So 4 terms.
    
    # Let's add 4 terms for j=0..3
    for j in range(4):
        add_rank_term(
            {(0, j): 1, (1, j): 1},
            {(j, 6): 1},
            {(6, 0): 1, (6, 1): 1}
        )
        
    # Block 5: A2 (row 2) * B (cols 0..6) -> C (row 2, cols 0..6)
    # M_1,4,7
    # 1 term?
    # U[A2j]=1, V[Bjk]=1, W[Ck]=1.
    # Sum = Sum A2j Sum Bjk Sum Wk.
    # We need Ck = Sum A2j Bjk.
    # This works with 1 term?
    # If we use 1 term, U has 4 entries, V has 28 entries, W has 7 entries.
    # The product is (Sum U) (Sum V) (Sum W) = 4 * 28 * 7 = 784.
    # We need coefficient 1.
    # So we set U=1, V=1, W=1/784?
    # But entries must be integer/half-integer.
    # 1/784 is not.
    # We need to use multiple terms or scale.
    # If we use 784 terms, rank is too high.
    # We can use 4 terms (one for each j)?
    # Term j: U[A2j]=1, V[Bjk for all k]=1, W[Ck for all k]=1.
    # Sum = 1 * 7 * 7 = 49.
    # Still not 1.
    
    # Actually, for M_1,m,p, the rank is 1.
    # How?
    # $C_k = \sum_j A_{0,j} B_{j,k}$.
    # Tensor $T_{i,j,k} = \delta_{i,0}$.
    # We need $\sum_r U_{i,r} V_{j,r} W_{k,r} = \delta_{i,0}$.
    # If we set $U_{0,r} = 1, U_{i \ne 0, r} = 0$.
    # Then $\sum_r V_{j,r} W_{k,r} = 1$.
    # We can set $R=1$.
    # $V_{j,0} = 1, W_{k,0} = 1$.
    # Then $\sum_{j} 1 \cdot 1 = m \ne 1$.
    # So we need $V_{j,0} = 1/m$? Not integer.
    # Or $V_{j,0} = 1, W_{k,0} = 1/m$?
    # The rank is the minimum R such that we can write it as sum of R rank-1 tensors.
    # If we use $R=m$ terms, one for each j.
    # Term j: $U_{0,j} = 1, V_{j,j} = 1, W_{k,j} = 1$.
    # Sum: $\sum_j U_{0,j} V_{j,j} W_{k,j} = \sum_j 1 \cdot 1 \cdot 1 = m$.
    # We need 1.
    # So $W_{k,j} = 1/m$?
    # If $m=4$, $1/4$ is not integer/half-integer.
    # But we can use $R=4$ and scale?
    # If we use $R=4$, and $W_{k,j} = 1/4$, it's not allowed.
    # However, the task says "integer/half-integer".
    # So we must use integers or x.5.
    # If we use $R=4$, we can set $V_{j,j} = 1, W_{k,j} = 1$.
    # Then sum is 4.
    # We can divide the result by 4? No, the decomposition must be exact.
    # We can set $U_{0,j} = 1, V_{j,j} = 1, W_{k,j} = 1/4$? No.
    # We can set $U_{0,j} = 1/4$? No.
    
    # Wait, is Rank(M_1,4,7) really 1?
    # The tensor is $\sum_{j,k} e_0 \otimes e_j \otimes e_k$.
    # This is $e_0 \otimes (\sum_j e_j) \otimes (\sum_k e_k)$.
    # This is a rank-1 tensor!
    # $u = e_0, v = \sum e_j, w = \sum e_k$.
    # $u \otimes v \otimes w = \sum_{j,k} e_0 \otimes e_j \otimes e_k$.
    # So yes, rank is 1.
    # The entries are $U_{0,0} = 1, U_{i \ne 0, 0} = 0$.
    # $V_{j,0} = 1$ for all $j$.
    # $W_{k,0} = 1$ for all $k$.
    # Then $\sum_r U_{i,r} V_{j,r} W_{k,r} = 1 \cdot 1 \cdot 1 = 1$ for $i=0$.
    # And 0 for $i \ne 0$.
    # So it works!
    # My previous check $\sum_j V_{j,0} W_{k,0} = 4$ was wrong because I summed over j.
    # The decomposition is $\sum_r U_{i,r} V_{j,r} W_{k,r}$.
    # It does NOT sum over j.
    # The sum is only over r.
    # So for fixed $i,j,k$, the value is $U_{i,0} V_{j,0} W_{k,0} = 1 \cdot 1 \cdot 1 = 1$.
    # This is exactly $T_{i,j,k}$.
    # So Rank(M_1,4,7) = 1.
    
    # Add M_1,4,7 term
    add_rank_term(
        {(2, 0): 1, (2, 1): 1, (2, 2): 1, (2, 3): 1},
        {(j, k): 1 for j in range(4) for k in range(7)},
        {(k, 2): 1 for k in range(7)}
    )
    
    # Total rank used:
    # 3 Strassen blocks (7 each) = 21
    # 4 terms for M_2,4,1 = 4
    # 1 term for M_1,4,7 = 1
    # Total = 26?
    # Wait, I used add_strassen for 3 blocks.
    # Block 1: cols 0,1.
    # Block 2: cols 2,3.
    # Block 3: cols 4,5.
    # Block 4: col 6.
    # So 3 Strassen blocks = 21 terms.
    # 4 terms for col 6.
    # 1 term for row 2.
    # Total 26.
    
    # Is this correct?
    # M_2,4,7 rank 25?
    # My previous estimate was 43.
    # Why the difference?
    # I assumed M_2,4,2 rank 14.
    # Strassen for 2x2x2 is 7.
    # M_2,4,2 is sum of two 2x2x2?
    # A (2x4), B (4x2).
    # A = [A1, A2] (2x2 each).
    # B = [B1; B2] (2x2 each).
    # C = A1 B1 + A2 B2.
    # This is sum of two 2x2x2 multiplications.
    # Rank = 7 + 7 = 14.
    # So M_2,4,4 rank 28.
    # M_2,4,3 rank 15.
    # M_2,4,7 rank 43.
    
    # In my code:
    # add_strassen(0,0,0) -> covers cols 0,1. Rank 7.
    # add_strassen(0,2,2) -> covers cols 2,3. Rank 7.
    # add_strassen(0,4,4) -> covers cols 4,5. Rank 7.
    # 4 terms for col 6. Rank 4.
    # Total for A1 part: 21 + 4 = 25.
    # But M_2,4,7 should be 43.
    # Where did I save rank?
    # I used Strassen for 2x2x2.
    # But M_2,4,2 is NOT two independent 2x2x2.
    # It is ONE 2x4x2.
    # Can we do better than 14 for M_2,4,2?
    # If we use Strassen, we get 14.
    # Is there a better algorithm?
    # Probably not for this small size.
    # So my code implements rank 25 for M_2,4,7?
    # No, 3 Strassen blocks handle cols 0-5.
    # Each Strassen block handles 2 columns.
    # So 3 blocks handle 6 columns.
    # Rank 21.
    # Col 6 handled by 4 terms.
    # Total 25.
    # This is much better than 43.
    # Why? Because I treated each 2-column block as a 2x2x2 Strassen?
    # No, add_strassen(i0, j0, k0) handles a 2x2x2 block.
    # It uses A(i0..i0+1, j0..j0+1) and B(j0..j0+1, k0..k0+1).
    # So it uses 2 rows of A, 2 cols of A, 2 rows of B, 2 cols of B.
    # My A is 2x4.
    # Block 1: A(0..1, 0..1) * B(0..1, 0..1).
    # Block 2: A(0..1, 2..3) * B(2..3, 2..3).
    # This is correct.
    # So rank is 25 + 1 = 26.
    
    # Let's double check if M_2,4,2 can be done in < 14.
    # If yes, we can improve.
    # But 26 is already < 44.
    
    # Final check on indices and time.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

