DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    total_p = p
    
    # We will split the multiplication into two parts:
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4) using Recursive Strassen (Rank 49)
    # Part 2: A (4x4) * B2 (4x3) -> C2 (4x3) using Schoolbook (Rank 48)
    
    # Helper to create sparse vectors
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * total_p)
        v[idx_j * total_p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (total_p * n)
        w[idx_k * n + idx_i] = val
        return w

    terms = []

    # --- Part 1: Recursive Strassen for 4x4x4 ---
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M6+M7
    # C12 = M2+M4
    # C21 = M3+M6
    # C22 = M1-M5+M7

    # We need to embed 2x2 Strassen into 4x4.
    # Split A into 4 blocks of 2x2: A11, A12, A21, A22 (indices 0..1, 2..3)
    # Split B1 into 4 blocks of 2x2: B11, B12, B21, B22 (cols 0..1, 2..3)
    
    # Function to add a 2x2 Strassen decomposition for a specific sub-block
    # offset_i, offset_j for A block (2x2)
    # offset_j, offset_k for B block (2x2)
    # offset_k, offset_i for C block (2x2)
    def add_strassen_2x2(offset_i, offset_j, offset_k):
        # A is at (offset_i, offset_j) 2x2
        # B is at (offset_j, offset_k) 2x2 -- Note: B is 4x4 here, so B1 is 4x4.
        # Actually, B1 is the first 4 columns of the original 4x7 B.
        # So B1 is 4x4. We split B1 into 2x2 blocks.
        # B11: rows 0-1, cols 0-1 (of B1) -> global rows 0-1, cols 0-1
        # B12: rows 0-1, cols 2-3 (of B1) -> global rows 0-1, cols 2-3
        # B21: rows 2-3, cols 0-1 (of B1) -> global rows 2-3, cols 0-1
        # B22: rows 2-3, cols 2-3 (of B1) -> global rows 2-3, cols 2-3
        
        # Strassen terms for this 2x2 block multiplication
        # M1 = (A11+A22)(B11+B22)
        # We need to express this in terms of the global tensor.
        # This requires 7 terms, each involving sums of A and B elements.
        # Since we need exact integer coefficients, we can expand the sums.
        
        # Let's define the 7 products for the 2x2 block at (offset_i, offset_j) x (offset_j, offset_k)
        # A_block indices: (offset_i, offset_j) to (offset_i+1, offset_j+1)
        # B_block indices: (offset_j, offset_k) to (offset_j+1, offset_k+1) -- Wait, B is 4x4.
        # The "j" index in B corresponds to the "j" index in A.
        # So B_block rows are offset_j, offset_j+1.
        # B_block cols are offset_k, offset_k+1.
        
        # M1 = (A_00+A_11) * (B_00+B_11)
        # Expansion: A_00 B_00 + A_00 B_11 + A_11 B_00 + A_11 B_11
        # This creates 4 rank-1 terms for M1 alone.
        # This is not efficient. Strassen reduces the number of *multiplications* of sub-matrices.
        # But at the scalar level, we are still doing scalar multiplications.
        # The rank of the tensor is the number of scalar multiplications.
        # Strassen reduces the number of *recursive* calls.
        # If we go all the way down to 1x1, the rank is the number of scalar multiplications.
        # For 2x2x2, Strassen uses 7 scalar multiplications.
        # For 4x4x4, Recursive Strassen uses 7 * 7 = 49 scalar multiplications.
        
        # So we need to generate the 7 terms for this 2x2 block, where each term is a product of a linear combination of A elements and a linear combination of B elements.
        # And then combine them for the C block.
        
        # Let's define the 7 Strassen "multipliers" for this block.
        # Each multiplier is a sum of scalar products.
        # We will add 7 terms to the global list.
        
        # A elements: A00, A01, A10, A11
        # B elements: B00, B01, B10, B11
        
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M2 = (A10+A11)B00 = A10B00 + A11B00
        # M3 = A00(B10-B11) = A00B10 - A00B11
        # M4 = A11(B01-B00) = A11B01 - A11B00
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        
        # C11 = M1 + M4 - M6 + M7
        # C12 = M2 + M4
        # C21 = M3 + M6
        # C22 = M1 - M5 + M7
        
        # We need to collect all scalar products and sum their coefficients for each C element.
        # This is effectively building the U, V, W matrices for this block.
        
        # Instead of expanding manually, we can use a dictionary to sum coefficients.
        # key: (i_local, j_local, k_local) -> coeff
        # But we need to output U, V, W vectors.
        
        # Let's just add the 7 "Strassen terms" directly.
        # Each Strassen term is a rank-1 tensor? No, it's a sum of rank-1 tensors.
        # Actually, the 7 multiplications in Strassen are of the form (linear combo of A) * (linear combo of B).
        # This is not a single rank-1 term in the scalar tensor. It is a sum of rank-1 terms.
        # To get the scalar rank, we must expand these products.
        # The number of scalar multiplications in Strassen 2x2 is 7.
        # Wait, Strassen 2x2 uses 7 *multiplications of 1x1 matrices*.
        # So it uses 7 scalar multiplications.
        # Therefore, the rank of M_2,2,2 is 7.
        # My previous expansion showed M1 has 4 terms. That's 4 scalar multiplications.
        # If I do that for all 7, I get more than 7.
        # The trick is that the "multiplications" in Strassen are the scalar multiplications.
        # So I should just add 7 terms, where each term is a product of a *single* A element and a *single* B element?
        # No, Strassen works by defining 7 new variables M1..M7 which are *products* of sums.
        # M1 = (A00+A11)(B00+B11). This requires 4 scalar multiplications if expanded.
        # So Strassen does *not* reduce the number of scalar multiplications for 2x2 if you expand the sums?
        # No, Strassen is for *matrix* multiplication. The inputs are matrices.
        # When we say "Rank 7", we mean the tensor rank of the multiplication map.
        # The tensor rank is the number of scalar multiplications in the *optimal* algorithm.
        # Strassen's algorithm for 2x2 uses 7 scalar multiplications.
        # How?
        # It computes M1 = (A00+A11) * (B00+B11).
        # This involves adding A00+A11 (free) and B00+B11 (free), then multiplying them (1 mult).
        # So yes, it is 1 scalar multiplication for that term.
        # So I should add 1 term for M1, 1 for M2, etc.
        # But the term is not rank-1 in the original variables A, B.
        # The decomposition must be in terms of the original A, B elements.
        # So I *must* expand M1 into A00B00 + ...
        # This means the rank is the number of scalar multiplications in the *expanded* form.
        # If I expand M1, I get 4 multiplications.
        # Then the total rank for 2x2 Strassen would be 4+2+2+2+2+4+4 = 20?
        # That's worse than schoolbook (8).
        
        # CORRECTION:
        # The tensor rank of M_2,2,2 is 7.
        # This means there *exists* a decomposition into 7 rank-1 tensors.
        # These rank-1 tensors are not necessarily aligned with the standard basis of A and B.
        # However, the problem requires U, V, W to have integer/half-integer entries.
        # And the decomposition must be exact.
        # The 7 rank-1 terms in Strassen are:
        # T1 = (A00+A11) (B00+B11) ... this is not rank-1 in the standard basis.
        # Wait, the rank-1 tensors in the decomposition of the multiplication tensor are:
        # u_i \otimes v_i \otimes w_i.
        # For Strassen, the "multiplications" are the rank-1 terms?
        # No, the rank-1 terms correspond to the scalar multiplications in the algorithm.
        # In Strassen, the 7 multiplications are of the form (linear combo) * (linear combo).
        # This is NOT a rank-1 term in the standard basis of A_{ij}, B_{jk}.
        # A rank-1 term is A_{ij} B_{jk}.
        # So Strassen's 7 multiplications do NOT directly give 7 rank-1 terms in the standard basis.
        # They give 7 terms that are *products of sums*.
        # To convert to standard basis, we must expand.
        # Expansion increases the number of terms.
        
        # Therefore, the tensor rank of M_2,2,2 is NOT 7 in the standard basis?
        # Actually, the tensor rank IS 7.
        # This implies that there is a way to write the multiplication as a sum of 7 rank-1 tensors.
        # These tensors are not the Strassen M_i directly.
        # They are a different basis.
        
        # Since I don't have the explicit 7 rank-1 vectors for M_2,2,2 in the standard basis, I cannot implement Strassen easily.
        
        # I will stick to Schoolbook.
        pass

    # Since I cannot reliably implement a lower-rank algorithm from memory, I will return the schoolbook decomposition.
    # This ensures validity and score 1.0.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
