def _strassen_terms(n, a_idx, b_idx, c_idx):
    """
    Returns list of (u, v, w, coeff) for n x n * n x n.
    a_idx, b_idx, c_idx are 2D lists of indices.
    """
    if n == 1:
        return [(a_idx[0][0], b_idx[0][0], c_idx[0][0], 1.0)]
    
    h = n // 2
    # Split indices
    # A11: a_idx[0:h, 0:h]
    # A12: a_idx[0:h, h:n]
    # A21: a_idx[h:n, 0:h]
    # A22: a_idx[h:n, h:n]
    
    # Same for B and C.
    
    # We need to pass sub-indices to the recursive call.
    # The recursive call will return terms for the sub-problem.
    # But the sub-problem is a 2x2 * 2x2 (if n=4) or 1x1 * 1x1 (if n=2).
    
    # For n=2, we use Strassen.
    # For n=1, we use schoolbook.
    
    if n == 2:
        # Strassen for 2x2
        # a_idx is 2x2 list of indices.
        # Let's denote a00 = a_idx[0][0], etc.
        a00, a01 = a_idx[0][0], a_idx[0][1]
        a10, a11 = a_idx[1][0], a_idx[1][1]
        b00, b01 = b_idx[0][0], b_idx[0][1]
        b10, b11 = b_idx[1][0], b_idx[1][1]
        c00, c01 = c_idx[0][0], c_idx[0][1]
        c10, c11 = c_idx[1][0], c_idx[1][1]
        
        terms = []
        # M1 = (a00+a11)(b00+b11)
        # This is a product of sums.
        # (a00+a11) is a linear combination of A elements.
        # (b00+b11) is a linear combination of B elements.
        # The product is a bilinear form.
        # We need to expand this into terms of the form u * v * w.
        # But u, v, w are indices.
        # The term is (sum a_i) * (sum b_j).
        # This is not a single rank-1 term.
        # Wait, Strassen works by pre-computing linear combinations.
        # In the tensor decomposition, we don't have "pre-computation" steps.
        # The decomposition is a sum of rank-1 tensors.
        # Strassen's 7 products are not rank-1 tensors in the original A, B, C space.
        # They are products of linear combinations.
        # To express them as a sum of rank-1 tensors, we need to expand the products of sums.
        # (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11.
        # This is 4 terms.
        # So M1 contributes 4 rank-1 terms.
        # Similarly for M2..M7.
        # Total 7 * 4 = 28 terms?
        # No, Strassen reduces the number of *multiplications*.
        # In the tensor decomposition context, "rank" is the number of rank-1 tensors.
        # Schoolbook for 2x2*2x2 has 8 rank-1 terms (4 multiplications, each producing 1 term? No, 8 terms).
        # Wait, C00 = A00*B00 + A01*B10. That's 2 terms.
        # C01 = A00*B01 + A01*B11. 2 terms.
        # C10 = A10*B00 + A11*B10. 2 terms.
        # C11 = A10*B01 + A11*B11. 2 terms.
        # Total 8 terms.
        
        # Strassen says we can do it with 7 multiplications.
        # But each "multiplication" in Strassen is a product of linear combinations.
        # If we expand the linear combinations, we get more than 7 terms.
        # For example, M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11.
        # This is 4 terms.
        # So the total number of rank-1 terms in the expanded form is NOT 7.
        # It is the sum of the number of terms in each Strassen product.
        # M1: 4 terms.
        # M2: (A10+A11)B00 = A10*B00 + A11*B00. 2 terms.
        # M3: A00(B11-B01) = A00*B11 - A00*B01. 2 terms.
        # M4: A11(B10-B00) = A11*B10 - A11*B00. 2 terms.
        # M5: (A00+A01)B11 = A00*B11 + A01*B11. 2 terms.
        # M6: (A10-A00)(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01. 4 terms.
        # M7: (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11. 4 terms.
        
        # Total terms: 4+2+2+2+2+4+4 = 20 terms.
        # This is worse than schoolbook (8 terms)!
        
        # Why is Strassen rank 7 then?
        # Because the "rank" in Strassen refers to the number of *scalar multiplications* required.
        # In the tensor decomposition context, the "rank" is the number of rank-1 tensors.
        # A rank-1 tensor is of the form u v w.
        # If we have a term like (a+b)(c+d), it is not a single rank-1 tensor.
        # It is a sum of 4 rank-1 tensors.
        # So the tensor rank of 2x2*2x2 is 8 (schoolbook), not 7.
        # Strassen does NOT reduce the tensor rank of 2x2*2x2.
        # It reduces the arithmetic complexity.
        
        # Wait, is the tensor rank of 2x2*2x2 8?
        # Yes, the tensor rank of 2x2*2x2 is 8.
        # The tensor rank of 3x3*3x3 is 23 (Strassen).
        # The tensor rank of 4x4*4x4 is 49?
        # Let's check.
        # If 2x2*2x2 has rank 8, then 4x4*4x4 using Strassen would have rank 7 * 8 = 56.
        # Schoolbook is 64.
        # So 56 is an improvement.
        
        # So I should use Strassen at the top level (7 products) and schoolbook at the inner level (8 terms each).
        # Total rank = 7 * 8 = 56 for 4x4*4x4.
        # Then for 4x4*4x9:
        # 4x4*4x4 (rank 56) + 4x4*4x4 (rank 56) + 4x4*4x1 (rank 16) = 128.
        # 128 < 144.
        
        # So I will implement this:
        # 1. Split 4x4*4x9 into two 4x4*4x4 and one 4x4*4x1.
        # 2. Use Strassen (7 products) for the 4x4*4x4 parts.
        # 3. Use schoolbook (8 terms) for the inner 2x2*2x2 products.
        # 4. Use schoolbook (16 terms) for the 4x4*4x1 part.
        
        # Total rank = 56 + 56 + 16 = 128.
        
        return terms
