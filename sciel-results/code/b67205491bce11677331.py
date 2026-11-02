"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Decompose B into 13 rank-1 column matrices B_k = e_k * row_k^T? No.
    # B is 2x13. Let B = sum_{k=0}^{12} B_k where B_k has only column k non-zero.
    # B_k is a 2x13 matrix with column k being [b_0k, b_1k]^T and others 0.
    # Actually, it's easier to view the multiplication C = A B where A is 2x2 and B is 2x13.
    # C is 2x13.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a tensor of size 2x2x13 (indices i, j, k).
    # We can decompose this tensor as a sum of rank-1 tensors.
    # A rank-1 tensor T_{i,j,k} = u_i v_j w_k.
    # We want to express the multiplication tensor M_{i,j,k} = delta_{i,k}? No.
    # The multiplication tensor T_{i,j,k} is defined such that C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}?
    # Standard convention: T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}.
    # Here, the indices are:
    # A: i (row), j (col) -> size n x m = 2 x 2
    # B: j (row), k (col) -> size m x p = 2 x 13
    # C: i (row), k (col) -> size n x p = 2 x 13
    # The tensor T has indices (i, j, k) corresponding to (row_A, col_A/row_B, col_B).
    # Wait, the standard tensor for matrix multiplication (n,m,p) has indices (i, j, k)
    # where i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # The tensor entry T_{i,j,k} is 1 if ...?
    # Actually, the tensor is defined by the structure of the multiplication.
    # For a single scalar multiplication A_{i,j} B_{j,k} contributing to C_{i,k},
    # the term is A_{i,j} B_{j,k} C_{i,k}.
    # So the tensor T_{i,j,k} = 1 for all valid i,j,k? No.
    # The tensor is a 3-way array.
    # For standard matrix multiplication A (n x m) * B (m x p) = C (n x p),
    # the tensor T has size n x m x p.
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In standard multiplication, every term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k?
    # If T_{i,j,k} = 1, then sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,j,k} A_{i,j} B_{j,k}.
    # This is not C.
    # The correct tensor decomposition expresses the *map* from (A,B) to C.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We can write this as a sum of rank-1 terms:
    # C_{i,k} = sum_{r} (sum_{i,j} U_{i,j,r} A_{i,j}) (sum_{j,k} V_{j,k,r} B_{j,k}) W_{i,k,r}?
    # No, the standard form is:
    # C_{i,k} = sum_r U_{i,r} V_{k,r} (sum_{i,j} U'_{i,j,r} A_{i,j})?
    # Let's use the interface provided:
    # U: (n*m, rank) -> U_{idx_A, r} where idx_A = i*m + j
    # V: (m*p, rank) -> V_{idx_B, r} where idx_B = j*p + k
    # W: (p*n, rank) -> W_{idx_C, r} where idx_C = k*n + i
    # The condition is:
    # For all i, k:
    # C_{i,k} = sum_r W_{k*n + i, r} * (sum_{i',j'} U_{i'*m+j', r} A_{i',j'}) * (sum_{j',k'} V_{j'*p+k', r} B_{j',k'})
    # This must equal sum_j A_{i,j} B_{j,k}.
    
    # Let's analyze the structure.
    # Let x_r = sum_{i',j'} U_{i',j',r} A_{i',j'}
    # Let y_r = sum_{j',k'} V_{j',k',r} B_{j',k'}
    # Then C_{i,k} = sum_r W_{i,k,r} x_r y_r.
    
    # We want C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    # Consider the case where we pick specific terms.
    # If we set U_{i,j,r} = delta_{i,a} delta_{j,b}, then x_r = A_{a,b}.
    # If we set V_{j,k,r} = delta_{j,c} delta_{k,d}, then y_r = B_{c,d}.
    # Then the term is W_{i,k,r} A_{a,b} B_{c,d}.
    # For this to contribute to C_{i,k} = sum_j A_{i,j} B_{j,k},
    # we need a=i, b=j, c=j, d=k.
    # So for a fixed term r, if U picks A_{i,j} and V picks B_{j,k},
    # then W must place this product into C_{i,k}.
    # So W_{i,k,r} = 1, and W_{i',k',r} = 0 for other i',k'.
    
    # This suggests that for each (i,j,k), we could have a term.
    # But that would be rank n*m*p = 2*2*13 = 52.
    # However, we can combine terms.
    
    # Let's look at the previous solution's logic which achieved rank 46.
    # It used Strassen for 2x2x2 blocks.
    # The tensor is 2x2x13.
    # We can view this as 13 independent 2x2x1 tensors?
    # For each k, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a 2x2x1 tensor (indices i, j, k fixed).
    # The rank of the 2x2x1 tensor?
    # C_{0,k} = A_{0,0} B_{0,k} + A_{0,1} B_{1,k}
    # C_{1,k} = A_{1,0} B_{0,k} + A_{1,1} B_{1,k}
    
    # Can we do this in rank 2?
    # Term 1: x_1 = A_{0,0} + A_{1,0}? No.
    # Let's try to find a rank 2 decomposition for 2x2x1.
    # We need to represent the pair (C_{0,k}, C_{1,k}) using 2 bilinear forms.
    # Let u_1, u_2 be linear forms in A.
    # Let v_1, v_2 be linear forms in B.
    # C_{0,k} = W_{0,k,1} u_1 v_1 + W_{0,k,2} u_2 v_2
    # C_{1,k} = W_{1,k,1} u_1 v_1 + W_{1,k,2} u_2 v_2
    
    # We have 4 unknowns in W (W_{0,1}, W_{0,2}, W_{1,1}, W_{1,2}) and
    # 4 unknowns in u_1, u_2 (coeffs of A_{0,0}, A_{0,1}, A_{1,0}, A_{1,1}) and
    # 4 unknowns in v_1, v_2 (coeffs of B_{0,k}, B_{1,k}).
    
    # Let's try a simple construction.
    # u_1 = A_{0,0}
    # u_2 = A_{1,0}
    # v_1 = B_{0,k}
    # v_2 = B_{1,k}
    # Then:
    # C_{0,k} = W_{0,1} A_{0,0} B_{0,k} + W_{0,2} A_{1,0} B_{1,k}
    # C_{1,k} = W_{1,1} A_{0,0} B_{0,k} + W_{1,2} A_{1,0} B_{1,k}
    # We need:
    # C_{0,k} = A_{0,0} B_{0,k} + A_{0,1} B_{1,k}
    # C_{1,k} = A_{1,0} B_{0,k} + A_{1,1} B_{1,k}
    # This doesn't match. We are missing A_{0,1} and A_{1,1} terms.
    
    # Let's try:
    # u_1 = A_{0,0} + A_{0,1}
    # u_2 = A_{1,0} + A_{1,1}
    # v_1 = B_{0,k}
    # v_2 = B_{1,k}
    # C_{0,k} = W_{0,1} (A_{0,0}+A_{0,1}) B_{0,k} + W_{0,2} (A_{1,0}+A_{1,1}) B_{1,k}
    # This introduces cross terms like A_{0,0} B_{1,k} if W_{0,2} is non-zero.
    
    # Actually, the rank of the 2x2x1 tensor is known to be 2?
    # Let's check the rank of the matrix M_k where M_k is the 2x2 matrix of coefficients?
    # No, it's a tensor.
    # Flattening along k: It's two 2x2 matrices?
    # Slice k=0 (only column k):
    # C_{0,k} = A_{0,0} B_{0,k} + A_{0,1} B_{1,k}
    # C_{1,k} = A_{1,0} B_{0,k} + A_{1,1} B_{1,k}
    
    # Let's try rank 2 again with different u, v.
    # u_1 = A_{0,0}
    # u_2 = A_{0,1}
    # v_1 = B_{0,k}
    # v_2 = B_{1,k}
    # C_{0,k} = W_{0,1} A_{0,0} B_{0,k} + W_{0,2} A_{0,1} B_{1,k}
    # We need W_{0,1}=1, W_{0,2}=1.
    # C_{1,k} = W_{1,1} A_{0,0} B_{0,k} + W_{1,2} A_{0,1} B_{1,k}
    # We need C_{1,k} = A_{1,0} B_{0,k} + A_{1,1} B_{1,k}.
    # This form cannot produce A_{1,0} or A_{1,1}.
    
    # So we must involve A_{1,0} and A_{1,1} in u.
    # Let's try:
    # u_1 = A_{0,0} + A_{1,0}
    # u_2 = A_{0,1} + A_{1,1}
    # v_1 = B_{0,k}
    # v_2 = B_{1,k}
    # C_{0,k} = W_{0,1} (A_{0,0}+A_{1,0}) B_{0,k} + W_{0,2} (A_{0,1}+A_{1,1}) B_{1,k}
    # We need C_{0,k} = A_{0,0} B_{0,k} + A_{0,1} B_{1,k}.
    # This requires:
    # Coeff of A_{0,0} B_{0,k}: W_{0,1} = 1
    # Coeff of A_{1,0} B_{0,k}: W_{0,1} = 0 -> Contradiction.
    
    # Let's try:
    # u_1 = A_{0,0}
    # u_2 = A_{1,0}
    # v_1 = B_{0,k} + B_{1,k}
    # v_2 = B_{0,k} - B_{1,k}
    # C_{0,k} = W_{0,1} A_{0,0} (B_{0,k}+B_{1,k}) + W_{0,2} A_{1,0} (B_{0,k}-B_{1,k})
    # = (W_{0,1} A_{0,0} + W_{0,2} A_{1,0}) B_{0,k} + (W_{0,1} A_{0,0} - W_{0,2} A_{1,0}) B_{1,k}
    # We need:
    # C_{0,k} = A_{0,0} B_{0,k} + A_{0,1} B_{1,k}
    # This requires A_{0,1} to appear, but it doesn't.
    
    # It seems rank 2 is not sufficient for the 2x2x1 block if we treat it independently?
    # Wait, the previous solution used rank 4 for 2x2x1.
    # And rank 7 for 2x2x2.
    # Total rank 6*7 + 4 = 46.
    
    # Is there a better way?
    # The tensor is 2x2x13.
    # We can view it as 13 slices of 2x2.
    # Or 2 slices of 2x13.
    # Let's look at the structure again.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is essentially 13 independent matrix-vector multiplications (column-wise).
    # Each column k of B is a vector b_k of size 2.
    # C_{:,k} = A b_k.
    # The map A -> C_{:,k} is linear.
    # The whole map (A, B) -> C is bilinear.
    
    # Consider the rank of the tensor T_{i,j,k}.
    # T_{i,j,k} = 1 if ...?
    # Actually, the tensor entries are defined by the indices.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k?
    # No, T is a 3-way array.
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In standard multiplication, A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k.
    # If T_{i,j,k} = 1, then the tensor is the all-ones tensor?
    # No, the indices are coupled.
    # i is from A, j is from A/B, k is from B.
    # The tensor is not all-ones.
    # For a fixed i, k, the sum is over j.
    # The tensor T has non-zero entries at (i, j, k) for all i, j, k.
    # Wait, if T_{i,j,k} = 1 for all i,j,k, then
    # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,j,k} A_{i,j} B_{j,k}.
    # This is not C.
    # The tensor decomposition expresses the *output* C as a sum of products of linear forms in A and B.
    # C_{i,k} = sum_r U_{i,k,r} (linear form in A) (linear form in B)?
    # No, the interface is:
    # C_{i,k} = sum_r W_{idx_C, r} * (sum_{i',j'} U_{idx_A, r} A_{i',j'}) * (sum_{j',k'} V_{idx_B, r} B_{j',k'})
    
    # Let's stick to the known lower bounds.
    # Rank(2x2x2) = 7.
    # Rank(2x2x1) = 4?
    # If we can find a rank 3 decomposition for 2x2x1, total rank would be 13*3 = 39?
    # Or maybe we can combine columns?
    
    # Let's try to construct a rank 3 decomposition for 2x2x1.
    # We need:
    # C_0 = A_{0,0} B_0 + A_{0,1} B_1
    # C_1 = A_{1,0} B_0 + A_{1,1} B_1
    
    # Let u_1, u_2, u_3 be linear forms in A.
    # Let v_1, v_2, v_3 be linear forms in B.
    # C_0 = w_{0,1} u_1 v_1 + w_{0,2} u_2 v_2 + w_{0,3} u_3 v_3
    # C_1 = w_{1,1} u_1 v_1 + w_{1,2} u_2 v_2 + w_{1,3} u_3 v_3
    
    # Try:
    # u_1 = A_{0,0}
    # u_2 = A_{0,1}
    # u_3 = A_{1,0} + A_{1,1}
    # v_1 = B_0
    # v_2 = B_1
    # v_3 = B_0 + B_1
    
    # C_0 = w_{0,1} A_{0,0} B_0 + w_{0,2} A_{0,1} B_1 + w_{0,3} (A_{1,0}+A_{1,1}) (B_0+B_1)
    # We need C_0 = A_{0,0} B_0 + A_{0,1} B_1.
    # So w_{0,1}=1, w_{0,2}=1, w_{0,3}=0.
    
    # C_1 = w_{1,1} A_{0,0} B_0 + w_{1,2} A_{0,1} B_1 + w_{1,3} (A_{1,0}+A_{1,1}) (B_0+B_1)
    # We need C_1 = A_{1,0} B_0 + A_{1,1} B_1.
    # This requires w_{1,1}=0, w_{1,2}=0.
    # Then C_1 = w_{1,3} (A_{1,0}+A_{1,1}) (B_0+B_1) = w_{1,3} (A_{1,0} B_0 + A_{1,0} B_1 + A_{1,1} B_0 + A_{1,1} B_1).
    # We need A_{1,0} B_0 + A_{1,1} B_1.
    # This has extra terms A_{1,0} B_1 and A_{1,1} B_0.
    # So this doesn't work.
    
    # It is known that Rank(2x2x1) = 4.
    # So the current solution of 46 is likely optimal for this block decomposition.
    # However, we can try to optimize the constants or use a different block size.
    # 13 = 2 * 6 + 1.
    # What if we use 1x2 blocks? No.
    # What if we use a different Strassen variant?
    
    # Let's verify the current code's correctness first.
    # The current code uses Strassen for 2x2x2 blocks.
    # Strassen rank is 7.
    # The last column uses 4 terms.
    # Total rank 46.
    
    # Can we do better?
    # Maybe use a 2x1x2 block?
    # Or 1x2x2?
    
    # Let's try to improve the 2x2x1 part.
    # Is there a rank 3 decomposition?
    # According to literature, the border rank of 2x2x1 is 3, but the tensor rank is 4.
    # So 4 is minimal for 2x2x1.
    
    # What about 2x2x3?
    # Rank(2x2x3) <= Rank(2x2x2) + Rank(2x2x1) = 7 + 4 = 11.
    # Is it less?
    # 13 = 2 * 5 + 3.
    # Rank = 5 * 7 + 11 = 35 + 11 = 46. Same.
    
    # 13 = 2 * 4 + 5.
    # Rank(2x2x5) <= 2 * 7 + 4 = 18.
    # Total = 4 * 7 + 18 = 28 + 18 = 46. Same.
    
    # It seems 46 is a robust upper bound.
    # The current code implements this.
    # I will rewrite the code to be cleaner and ensure it is correct.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
    
    # Strassen 2x2x2
    # Indices:
    # U: i*m+j
    # V: j*p+k
    # W: k*n+i
    
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # M1: (A00+A11)(B00+B11)
        # A: 0,3. B: k0, 13+k1. C: 2*k0, 2*k1+1
        add_term({0: 1, 3: 1}, {k0: 1, 13+k1: 1}, {2*k0: 1, 2*k1+1: 1})
        
        # M2: (A10+A11)B00
        # A: 2,3. B: k0. C: 2*k0+1, 2*k1+1 (with -1)
        # C10 = A10 B00 + A11 B00 - A11 B00? No.
        # Strassen:
        # C00 = M1 - M2 + M3 - M4
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M5 + M6 + M7
        
        # Let's check the coefficients in the current code.
        # M1: U{0:1,3:1}, V{k0:1,13+k1:1}, W{2*k0:1, 2*k1+1:1}
        # This contributes to C_{0,k0} (idx 2*k0) and C_{1,k1} (idx 2*k1+1).
        # Wait, W index is k*n+i.
        # For C_{0,k0}, k=k0, i=0 -> idx 2*k0.
        # For C_{1,k1}, k=k1, i=1 -> idx 2*k1+1.
        # M1 = (A00+A11)(B00+B11).
        # This product is added to C00 and C11.
        # In Strassen, M1 is added to C00 and C11. Correct.
        
        # M2: (A10+A11)B00
        # U{2:1,3:1}, V{k0:1}, W{2*k0+1:1, 2*k1+1:-1}
        # Contributes to C_{1,k0} (idx 2*k0+1) and C_{1,k1} (idx 2*k1+1).
        # M2 is added to C10 and subtracted from C11?
        # Strassen: C10 = M2 + M4. C11 = M1 - M5 + M6 + M7.
        # So M2 should be +1 for C10.
        # What about C11? M2 is not in C11.
        # But the code has W{2*k1+1:-1}. This implies M2 is subtracted from C11.
        # This is incorrect for standard Strassen.
        # Let's re-verify Strassen formulas.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M2 + M3 - M4
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M5 + M6 + M7
        
        # Let's check the code's W coefficients against this.
        # M1: W{2*k0:1, 2*k1+1:1}. Correct (C00, C11).
        # M2: W{2*k0+1:1, 2*k1+1:-1}.
        #   C10 (2*k0+1) gets +1. Correct.
        #   C11 (2*k1+1) gets -1. Incorrect. M2 is not in C11.
        #   Wait, is it possible the code is using a different Strassen variant or I am misinterpreting the indices?
        #   Let's check M4.
        # M4: U{3:1}, V{13+k0:1, k0:-1}, W{2*k0:1, 2*k0+1:1}
        #   V: B10 - B00. (13+k0 is B10, k0 is B00). Correct.
        #   W: C00 (2*k0) gets +1. C10 (2*k0+1) gets +1.
        #   Strassen: C00 = ... - M4. So C00 should get -1.
        #   Code has +1.
        #   C10 = ... + M4. So C10 should get +1.
        #   Code has +1.
        #   So M4 has wrong sign for C00.
        
        # It seems the current code might be buggy or using a non-standard assignment.
        # I will rewrite the Strassen part carefully.
        
        # M1: (A00+A11)(B00+B11)
        # U: {0:1, 3:1}
        # V: {k0:1, 13+k1:1}
        # W: {2*k0:1, 2*k1+1:1}
        add_term({0: 1, 3: 1}, {k0: 1, 13+k1: 1}, {2*k0: 1, 2*k1+1: 1})
        
        # M2: (A10+A11)B00
        # U: {2:1, 3:1}
        # V: {k0:1}
        # W: {2*k0+1:1}  # Only C10
        add_term({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1})
        
        # M3: A00(B01-B11)
        # U: {0:1}
        # V: {k1:1, 13+k1:-1}
        # W: {2*k0:1, 2*k1:1} # C00 and C01
        add_term({0: 1}, {k1: 1, 13+k1: -1}, {2*k0: 1, 2*k1: 1})
        
        # M4: A11(B10-B00)
        # U: {3:1}
        # V: {13+k0:1, k0:-1}
        # W: {2*k0:-1, 2*k0+1:1} # C00 (-1), C10 (+1)
        add_term({3: 1}, {13+k0: 1, k0: -1}, {2*k0: -1, 2*k0+1: 1})
        
        # M5: (A00+A01)B11
        # U: {0:1, 1:1}
        # V: {13+k1:1}
        # W: {2*k0:-1, 2*k1:1} # C00 (-1), C01 (+1)
        add_term({0: 1, 1: 1}, {13+k1: 1}, {2*k0: -1, 2*k1: 1})
        
        # M6: (A10-A00)(B00+B01)
        # U: {2:1, 0:-1}
        # V: {k0:1, k1:1}
        # W: {2*k1+1:1} # C11
        add_term({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1})
        
        # M7: (A01-A11)(B10+B11)
        # U: {1:1, 3:-1}
        # V: {13+k0:1, 13+k1:1}
        # W: {2*k1+1:1} # C11
        add_term({1: 1, 3: -1}, {13+k0: 1, 13+k1: 1}, {2*k1+1: 1})
        
    # Last column k=12
    k = 12
    # C0k = A00 B0k + A01 B1k
    # C1k = A10 B0k + A11 B1k
    # 4 terms
    add_term({0: 1}, {k: 1}, {2*k: 1})
    add_term({1: 1}, {13+k: 1}, {2*k: 1})
    add_term({2: 1}, {k: 1}, {2*k+1: 1})
    add_term({3: 1}, {13+k: 1}, {2*k+1: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
